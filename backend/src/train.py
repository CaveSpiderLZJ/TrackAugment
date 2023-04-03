import os
import gc
import tqdm
import shutil
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import config as cf
import file_utils as fu
from data_process.record import Record
from data_process.dataset import Dataset, DataLoader
from train.model import *


def main():
    # config parameters
    model = Model1()
    task_list_id = 'TLnmdi15b8'
    n_classes = cf.N_CLASSES
    class_names = cf.CLASS_NAMES
    group_id_to_name = {group_id: group_name for
        group_id, group_name in zip(range(n_classes), class_names)}
    n_epochs = cf.N_EPOCHS
    learning_rate = cf.LEARNING_RATE
    batch_size = cf.BATCH_SIZE
    warmup_steps = cf.WARMUP_STEPS
    log_steps = cf.LOG_STEPS
    eval_steps = cf.EVAL_STEPS
    model_save_dir = f'{cf.MODEL_ROOT}/{cf.MODEL_NAME}'
    log_save_dir = f'{cf.LOG_ROOT}/{cf.MODEL_NAME}'
    
    # load task list
    task_lists = fu.load_root_list_info()['tasklists']
    for task_list in task_lists:
        if task_list['id'] == task_list_id: break
    assert task_list['id'] == task_list_id
    for task in task_list['tasks']:
        task_id = task['id']
        for subtask in task['subtasks']:
            subtask_id = subtask['id']
            record_list = fu.load_record_list(task_list_id, task_id, subtask_id)
            record_dict= dict()
            for item in record_list:
                record_dict[item['user_name']] = item['record_id']
            subtask['record_dict'] = record_dict
    
    # build the dataset and dataloader
    train_users = set(cf.USERS[:40])
    test_users = set(cf.USERS[40:])
    train_dataset = Dataset(negative_label=6)
    test_dataset = Dataset(negative_label=6)
    
    # insert Shake, DoubleShake, Flip and DoubleFlip
    for task_id, label in (('TK9fe2fbln', 2), ('TK5rsia9fw', 3), ('TKtvkgst8r', 4), ('TKie8k1h6r', 5)):
        print(f'### {task_id}')
        for task in task_list['tasks']:
            if task['id'] == task_id: break
        assert task['id'] == task_id
        for subtask in task['subtasks']:
            subtask_id = subtask['id']
            print(f'### {subtask_id}')
            record_dict = subtask['record_dict']
            for user_name, record_id in record_dict.items():
                if user_name not in train_users and user_name not in test_users: continue
                record_path = fu.get_record_path(task_list_id, task_id, subtask_id, record_id)
                record = Record(record_path, subtask['times'])
                if user_name in train_users:
                    train_dataset.insert_record(record, label)
                else: test_dataset.insert_record(record, label)
                
    # insert Raise and Drop
    task_id = 'TK7t3ql6jb'
    for task in task_list['tasks']:
        if task['id'] == task_id: break
    assert task['id'] == task_id
    print(f'### {task_id}')
    for subtask in task['subtasks']:
        subtask_id = subtask['id']
        print(f'### {subtask_id}')
        record_dict = subtask['record_dict']
        for user_name, record_id in record_dict.items():
            if user_name not in train_users and user_name not in test_users: continue
            record_path = fu.get_record_path(task_list_id, task_id, subtask_id, record_id)
            record = Record(record_path, subtask['times'])
            if user_name in train_users:
                train_dataset.insert_record_raise_drop(record, raise_label=0, drop_label=1)
            else: test_dataset.insert_record_raise_drop(record, raise_label=0, drop_label=1)
    # insert negetive data
    # task_id = 'TKbszc8ch6'
    # record_paths = glob(f'{fu.get_task_path(task_list_id, task_id)}/ST*/RD*')
    # for record_path in record_paths:
    #     record = Record(record_path)
    #     dataset.insert_record(record, 0)    
    # dataset.shuffle()
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # dataloader.set_split_mode('train')
    train_dataset.shuffle()
    test_dataset.shuffle()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # utils
    if os.path.exists(model_save_dir): shutil.rmtree(model_save_dir)
    os.makedirs(model_save_dir)
    if os.path.exists(log_save_dir): shutil.rmtree(log_save_dir)
    os.makedirs(log_save_dir)
    model = model.to(cf.DEVICE)
    logger = SummaryWriter(log_save_dir)
    train_criterion = nn.CrossEntropyLoss()
    test_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer,
        start_factor=1/warmup_steps, end_factor=1.0, total_iters=warmup_steps)
    train_step = len(train_dataloader) * n_epochs - warmup_steps
    train_scheduler = optim.lr_scheduler.LinearLR(optimizer,
        start_factor=1.0, end_factor=1/train_step, total_iters=train_step)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer,
        schedulers=[warmup_scheduler, train_scheduler], milestones=[warmup_steps])
    
    # train process
    max_acc = -1.0
    train_log = []
    accs = []
    optimizer.zero_grad()
    for epoch in range(n_epochs):
        gc.collect()
        train_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            data: torch.Tensor = batch['data']
            label: torch.Tensor = batch['label'].to(cf.DEVICE)
            output: torch.Tensor = model(data.transpose(1,2))
            loss: torch.Tensor = train_criterion(output, label)
            loss.backward()
            train_loss += loss.detach().item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        train_loss /= len(train_dataloader)
        lr = optimizer.param_groups[0]['lr']
        train_log.append({"epoch": epoch, "lr": lr, "loss": train_loss})
        if epoch % log_steps == 0:
            print(f'epoch: {epoch}, lr: {lr:.4e}, loss: {train_loss:.4e}')
            logger.add_scalar('Loss', train_loss, epoch)
        if epoch % eval_steps == 0:  # test model on testing dataset
            model.eval()
            class_correct = np.zeros(n_classes, dtype=np.int32)
            class_total = np.zeros(n_classes, dtype=np.int32)
            matrix = np.zeros((n_classes, n_classes), dtype=np.int32)
            test_loss = 0.0
            with torch.no_grad():
                for i, batch in enumerate(test_dataloader):
                    data: torch.Tensor = batch['data']
                    label: torch.Tensor = batch['label'].to(cf.DEVICE)
                    output: torch.Tensor = model(data.transpose(1,2))
                    test_loss += test_criterion(output, label).item()
                    _, predicted = torch.max(output, dim=1)
                    c = (predicted == label)
                    for i in range(len(label)):
                        l = label[i]
                        matrix[l.item(), predicted[i].item()] += 1
                        is_correct = c[i]
                        class_correct[l] += is_correct
                        class_total[l] += 1
            test_loss /= len(test_dataloader)
            logger.add_scalar("Test Loss", test_loss, epoch)
            model.train()
            acc = np.diag(matrix).sum() / matrix.sum()
            accs.append(acc)
            if acc > max_acc:
                max_acc = acc
                torch.save(model.state_dict(), os.path.join(model_save_dir, 'best.model'))
            print(f'-'*64)
            for i in range(n_classes):
                print(f'Accuracy of {class_names[i]:12s}: {(100*matrix[i,i]/matrix[i,:].sum()):.1f} %')                        
                logger.add_scalar(f'Accuracy of {class_names[i]}', 100*matrix[i,i]/matrix[i,:].sum(), epoch)
            print(f'-'*64)
            logger.add_scalar('Accuracy', acc, epoch)
    print(f'Genaral acc: {(100*np.mean(accs[-10:])):.1f} %')
    

if __name__ == '__main__':
    np.random.seed(cf.RAND_SEED)
    torch.manual_seed(cf.RAND_SEED)
    fu.check_cwd()
    main()