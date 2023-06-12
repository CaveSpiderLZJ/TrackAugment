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
from train.generative_model import *
            

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
    for task_name, label in zip(class_names, (0, 1, 2, 3)):
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
    
    
def train_cgan():
    train_users = ['gsq', 'wxy', 'lc', 'qp', 'hjp', 'qwp', 'cr', 'hz', 'cjy',
        'hr', 'lxt', 'lst', 'lyf', 'zqy', 'wxb', 'lzj', 'fqj', 'mfj', 'lhs', 'xq']
    val_users = ['fhy']
    # train_users = ['gsq']
    # val_users = ['zqy']
    train_dataloader, val_dataloader = build_dataloaders(train_users, val_users)
    
    # config parameters
    n_epochs = 100
    learning_rate = 5e-3
    model_name = 'cgan2'
    log_save_dir = f'../data/log/gan/{model_name}'
    model_save_dir = f'../data/model/gan/{model_name}'
    
    # make dirs
    if os.path.exists(log_save_dir): shutil.rmtree(log_save_dir)
    os.makedirs(log_save_dir)
    if os.path.exists(model_save_dir): shutil.rmtree(model_save_dir)
    os.makedirs(model_save_dir)
    
    # utils
    device = torch.device('cpu')
    model_g = ConditionalGenerator2()
    model_d = ConditionalDiscriminator2()
    criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=learning_rate)
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=learning_rate)
    train_steps = len(train_dataloader) * n_epochs
    scheduler_g = torch.optim.lr_scheduler.LinearLR(optimizer_g,
        start_factor=1.0, end_factor=1/train_steps, total_iters=train_steps)
    scheduler_d = torch.optim.lr_scheduler.LinearLR(optimizer_d,
        start_factor=1.0, end_factor=1/train_steps, total_iters=train_steps)
    logger = SummaryWriter(log_save_dir)

    optimizer_g.zero_grad()
    optimizer_d.zero_grad()
    for epoch in range(n_epochs):
        mean_loss_g = []
        mean_loss_d = []
        mean_d_x = []
        mean_d_g_z1 = []
        mean_d_g_z2 = []
        for _, batch in enumerate(train_dataloader):
            data = batch['data'].transpose(1, 2)
            label = batch['label']
            # (1) Update D with real data
            model_d.zero_grad()
            b_size = data.shape[0]
            real_x = data.to(device)
            real_y = torch.zeros((b_size, 4), dtype=torch.float32)
            real_y[range(b_size), label] = 1.0
            output = model_d(real_x, real_y).view(-1)
            target = torch.ones((b_size,), dtype=torch.float32)
            loss_d_real = criterion(output, target)
            loss_d_real.backward()
            d_x = output.mean().item()
            # (2) Update the discriminator with fake data
            noise = torch.randn(b_size, 46, device=device)
            fake_x = model_g(noise, real_y)
            target = target.fill_(0.0)
            output = model_d(fake_x.detach(), real_y).view(-1)
            loss_d_fake = criterion(output, target)
            loss_d_fake.backward()
            d_g_z1 = output.mean().item()
            loss_d = loss_d_real + loss_d_fake
            optimizer_d.step()
            scheduler_d.step()
            # (3) Update the generator with fake data
            model_g.zero_grad()
            output = model_d(fake_x, real_y).view(-1)
            target.fill_(1.0)
            loss_g = criterion(output, target)
            loss_g.backward()
            d_g_z2 = output.mean().item()
            optimizer_g.step()
            scheduler_g.step()
            # log metrics
            mean_loss_g.append(loss_g.item())
            mean_loss_d.append(loss_d.item())
            mean_d_x.append(d_x)
            mean_d_g_z1.append(d_g_z1)
            mean_d_g_z2.append(d_g_z2)
        mean_loss_g = np.mean(mean_loss_g)
        mean_loss_d = np.mean(mean_loss_d)
        mean_d_x = np.mean(mean_d_x)
        mean_d_g_z1 = np.mean(mean_d_g_z1)
        mean_d_g_z2 = np.mean(mean_d_g_z2)
        logger.add_scalar('Loss G', mean_loss_g, epoch)
        logger.add_scalar('Loss D', mean_loss_d, epoch)
        logger.add_scalar('D(x)', mean_d_x, epoch)
        logger.add_scalar('D(G)1', mean_d_g_z1, epoch)
        logger.add_scalar('D(G)2', mean_d_g_z2, epoch)
        print(f'epoch: {epoch}, loss_g: {mean_loss_g:.3e}, loss_d: {mean_loss_d:.3e}, '
            + f'D(x): {mean_d_x:.3e}, D(G(x)): {mean_d_g_z1:.3e} / {mean_d_g_z2:.3e}')
    # save model
    torch.save(model_g.state_dict(), f'{model_save_dir}/generator.model')
    torch.save(model_d.state_dict(), f'{model_save_dir}/discriminator.model')
    
    
def inference_cgan():
    state_dict = torch.load('../data/model/gan/cgan2/generator.model')
    model = ConditionalGenerator2()
    model.load_state_dict(state_dict)
    model.eval()
    
    label = 3
    x = torch.randn(8, 46, dtype=torch.float32)
    y = torch.zeros(8, 4, dtype=torch.float32)
    y[:, label] = 1.0
    output = model(x, y).detach().numpy()

    for i, imu in enumerate(output):
        idx = i+1 if i < 4 else i+5
        plt.subplot(4, 4, idx)
        for j in range(3): plt.plot(imu[j, :])
        plt.ylim(-15, 15)
        plt.subplot(4, 4, idx + 4)
        for j in range(3): plt.plot(imu[j+3, :])
        plt.ylim(-5, 5)
    plt.show()
    
    
if __name__ == '__main__':
    fu.check_cwd()
    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)
    
    # train()
    # inference()
    train_cgan()
    # inference_cgan()
    
    # train_users = ['gsq', 'wxy', 'lc', 'qp', 'hjp', 'qwp', 'cr', 'hz', 'cjy',
    #     'hr', 'lxt', 'lst', 'lyf', 'zqy', 'wxb', 'lzj', 'fqj', 'mfj', 'lhs', 'xq']
    # # train_users = ['gsq']
    # val_users = ['fhy']
    # train_dataloader, val_dataloader = build_dataloaders(train_users, val_users)
    
    # for target_label in range(4):
    #     imu_list = []
    #     for i, batch in enumerate(train_dataloader):
    #         data = batch['data'].numpy().transpose(0,2,1)
    #         label = batch['label'].numpy()
    #         for j in range(label.shape[0]):
    #             if label[j] == target_label:
    #                 imu_list.append(data[j,:,:])
    #         if len(imu_list) >= 8: break
        
    #     for i, imu in enumerate(imu_list[:8]):
    #         idx = i+1 if i < 4 else i+5
    #         plt.subplot(4, 4, idx)
    #         for j in range(3): plt.plot(imu[j, :])
    #         plt.ylim(-15, 15)
    #         plt.subplot(4, 4, idx + 4)
    #         for j in range(3): plt.plot(imu[j+3, :])
    #         plt.ylim(-5, 5)
    #     plt.suptitle(f'Label: {target_label}')
    #     plt.show()
