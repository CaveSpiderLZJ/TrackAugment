import os
import time
import tqdm
import shutil
import random
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, List, Dict
from torch.utils.tensorboard import SummaryWriter


import file_utils as fu
from data_process.record import Record
from data_process.dataset import Dataset, DataLoader
from train.generative_model import VAE, VAE2
            

def build_dataloaders(train_users:list, val_users:list) -> Tuple[DataLoader, DataLoader]:
    # load task_list
    task_list_id = 'TLm5wv3uex'
    task_list = fu.load_task_list_with_users(task_list_id)
    assert task_list is not None
    
    class_names = ['Move10', 'Move20', 'Move30', 'Move40']
    train_dataset, val_dataset = Dataset(), Dataset()
    
    # insert positive data
    print(f'### Insert positive data.')
    record_info = []
    for task_name, label in zip(class_names[1:], (1, 2, 3, 4)):
        for task in task_list['tasks']:
            if task['name'] == task_name: break
        assert task['name'] == task_name
        for subtask in task['subtasks']:
            record_dict = subtask['record_dict']
            for user_name, record_id in record_dict.items():
                if user_name not in (train_users + val_users): continue
                record_info.append((task['id'], subtask['id'], record_id, user_name, subtask['times'], label))
    for task_id, subtask_id, record_id, user_name, times, label in tqdm.tqdm(record_info):
        record_path = fu.get_record_path(task_list_id, task_id, subtask_id, record_id)
        try:
            record = Record(record_path, times)
        except:
            print(f'### Error: {record_path}')
            continue
        if user_name in train_users:
            train_dataset.insert_record(record, label)
        elif user_name in val_users:
            val_dataset.insert_record(record, label)
    train_dataset.augment()
    val_dataset.augment()
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    
    return train_dataloader, val_dataloader


def train():
    train_users = ['gsq', 'wxy', 'lc', 'qp', 'hjp', 'qwp', 'cr', 'hz', 'fhy', 'cjy', 'hr', 'lxt', 'lst', 'lyf']
    val_users = ['zqy', 'wxb', 'lzj', 'fqj', 'mfj', 'lhs', 'xq']
    # train_users = ['gsq']
    # val_users = ['zqy']
    train_dataloader, val_dataloader = build_dataloaders(train_users, val_users)
    
    # config parameters
    n_epochs = 200
    learning_rate = 3e-3
    val_steps = 2
    ratio = 0.1     # use how much data to calculate mean performance
    c_kl = 0.001
    model_name = 'vae2_kl0.001'
    log_save_dir = f'../data/log/{model_name}'
    model_save_dir = f'../data/model/{model_name}'
    
    # make dirs
    if os.path.exists(log_save_dir): shutil.rmtree(log_save_dir)
    os.mkdir(log_save_dir)
    if os.path.exists(model_save_dir): shutil.rmtree(model_save_dir)
    os.mkdir(model_save_dir)
    
    # utils
    model = VAE2()
    train_loss_fn = nn.MSELoss(reduction='mean')
    val_loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_steps = len(train_dataloader) * n_epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
        start_factor=1.0, end_factor=1/train_steps, total_iters=train_steps)
    logger = SummaryWriter(log_save_dir)

    min_val_loss = 1e30
    model.train()
    optimizer.zero_grad()
    mean_train_loss = []
    mean_val_loss = [] 
    for epoch in range(n_epochs):
        train_loss = []
        train_reconstruction_loss = []
        for _, batch in enumerate(train_dataloader):
            data = batch['data'].transpose(1, 2)
            output, mu, sigma = model(data)
            reconstruction_loss: torch.Tensor = train_loss_fn(output, data)
            kl_loss = -0.5 * c_kl * torch.mean(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            loss = reconstruction_loss + kl_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss.append(loss.detach().item())
            train_reconstruction_loss.append(reconstruction_loss.detach().item())
        train_loss = np.mean(train_loss)
        train_reconstruction_loss = np.mean(train_reconstruction_loss)
        mean_train_loss.append(train_loss)
        logger.add_scalar('Train Loss', train_loss, epoch)
        logger.add_scalar('Train Reconstruction Loss', train_reconstruction_loss, epoch)
        
        if (epoch + 1) % val_steps == 0: # validation
            model.eval()
            val_loss = []
            val_reconstruction_loss = []
            with torch.no_grad():
                for _, batch in enumerate(val_dataloader):
                    data = batch['data'].transpose(1, 2)
                    output, mu, sigma = model(data)
                    reconstruction_loss: torch.Tensor = val_loss_fn(output, data)
                    kl_loss = -0.5 * c_kl * torch.mean(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
                    loss = reconstruction_loss + kl_loss
                    val_loss.append(loss.detach().item())
                    val_reconstruction_loss.append(reconstruction_loss.detach().item())
            val_loss = np.mean(val_loss)
            val_reconstruction_loss = (np.mean(val_reconstruction_loss))
            mean_val_loss.append(val_loss)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(model.state_dict(), f'{model_save_dir}/best.model')
            logger.add_scalar('Val Loss', val_loss, epoch)
            logger.add_scalar('Val Reconstruction Loss', val_reconstruction_loss, epoch)
            print(f'### epoch: {epoch}, train loss: ({train_loss:.3e},{train_reconstruction_loss:.3e}), ' + \
                f'val_loss: ({val_loss:.3e}, {val_reconstruction_loss:.3e})')
            model.train()
    logger.close()
    cnt = int(len(mean_train_loss)*ratio)
    mean_train_loss = np.mean(mean_train_loss[-cnt:])
    cnt = int(len(mean_val_loss)*ratio)
    mean_val_loss = np.mean(mean_val_loss[-cnt:])
    print(f'### mean train loss: {mean_train_loss:.3e}')
    print(f'### mean val loss: {mean_val_loss:.3e}')
    

def inference():
    train_users = ['gsq', 'wxy',]
    val_users = ['lzj', 'xq']
    train_dataloader, val_dataloader = build_dataloaders(train_users, val_users)
    
    model_state_dict = torch.load('../data/model/vae2_kl0.001/best.model')
    model = VAE2()
    model.load_state_dict(model_state_dict)
    model.eval()
    
    idx = 4
    imu_data = []
    for _, batch in enumerate(val_dataloader):
        data = batch['data'].transpose(1,2)[idx:idx+1,...]
        imu_data.append(data.detach().numpy()[0,:,:])
        for _ in range(7):
            output, mu, sigma = model(data)
            # plt.plot(torch.squeeze(mu).detach().numpy())
            # plt.plot(torch.squeeze(sigma).detach().numpy())
            # plt.show()
            # exit()
            output = torch.squeeze(output).detach().numpy()
            imu_data.append(output)
        break
    for i, imu in enumerate(imu_data):
        idx = i+1 if i < 4 else i+5
        plt.subplot(4, 4, idx)
        for j in range(3): plt.plot(imu[j, :])
        plt.subplot(4, 4, idx + 4)
        for j in range(3): plt.plot(imu[j+3, :])
    plt.show()
    
    
if __name__ == '__main__':
    fu.check_cwd()
    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)
    
    train()
    # inference()
        