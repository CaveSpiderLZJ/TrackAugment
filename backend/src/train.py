import os
import gc
import copy
import tqdm
import time
import shutil
import pickle
import random
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
from train.feature import feature2


def worker_init_fn(worker_id:int):
    random.seed(cf.RAND_SEED + worker_id)
    np.random.seed(cf.RAND_SEED + worker_id)
    
    
def build_dataloader_study1() -> Tuple[DataLoader, DataLoader]:
    ''' Build train and test dataloader in study-1.
    '''
    # load task_list
    task_list_id = 'TLnmdi15b8'
    task_list = fu.load_task_list_with_users(task_list_id)
    assert task_list is not None
    
    # build the dataset and dataloader
    users = list(cf.USERS)
    np.random.shuffle(users)
    train_users = set(users[:cf.N_TRAIN_USERS])
    test_users = set(users[cf.N_TRAIN_USERS:cf.N_TRAIN_USERS+cf.N_TEST_USERS])
    # remove left hand users
    if 'lby' in train_users: train_users.remove('lby')
    if 'lby' in test_users: test_users.remove('lby')
    days = list(range(1,75))
    np.random.shuffle(days)
    train_days = days[:cf.N_TRAIN_DAYS]
    test_days = days[cf.N_TRAIN_DAYS:cf.N_TRAIN_DAYS+cf.N_TEST_DAYS]
    train_negative_paths = []
    test_negative_paths = []
    for day in train_days:
        train_negative_paths.extend(glob(f'../data/negative/day{day}/*.pkl'))
    for day in test_days:
        test_negative_paths.extend(glob(f'../data/negative/day{day}/*.pkl'))
    train_dataset = Dataset()
    test_dataset = Dataset()
    
    # insert Shake, DoubleShake, Flip and DoubleFlip
    print(f'### Insert Shake, DoubleShake, Flip and DoubleFlip.')
    record_info = []
    for task_id, label in (('TK9fe2fbln', 3), ('TK5rsia9fw', 4), ('TKtvkgst8r', 5), ('TKie8k1h6r', 6)):
        for task in task_list['tasks']:
            if task['id'] == task_id: break
        assert task['id'] == task_id
        for subtask in task['subtasks']:
            subtask_id = subtask['id']
            record_dict = subtask['record_dict']
            for user_name, record_id in record_dict.items():
                if user_name not in train_users and user_name not in test_users: continue
                record_info.append((task_id, subtask_id, record_id, user_name, subtask['times'], label))
    for task_id, subtask_id, record_id, user_name, times, label in tqdm.tqdm(record_info):
        record_path = fu.get_record_path(task_list_id, task_id, subtask_id, record_id)
        try:
            record = Record(record_path, times)
        except:
            print(f'### Error: {record_path}')
            continue
        if user_name in train_users:
            train_dataset.insert_record(record, label)
        else: test_dataset.insert_record(record, label)
                
    # insert Raise and Drop
    print(f'### Insert Raise and Drop.')
    record_info = []
    task_id = 'TK7t3ql6jb'
    for task in task_list['tasks']:
        if task['id'] == task_id: break
    assert task['id'] == task_id
    for subtask in task['subtasks']:
        subtask_id = subtask['id']
        record_dict = subtask['record_dict']
        for user_name, record_id in record_dict.items():
            if user_name not in train_users and user_name not in test_users: continue
            record_info.append((subtask_id, record_id, user_name, subtask['times']))
    for subtask_id, record_id, user_name, times in tqdm.tqdm(record_info):
        record_path = fu.get_record_path(task_list_id, task_id, subtask_id, record_id)
        try:
            record = Record(record_path, times)
        except:
            print(f'### Error: {record_path}')
            continue
        if user_name in train_users:
            train_dataset.insert_record_raise_drop(record, raise_label=1, drop_label=2)
        else: test_dataset.insert_record_raise_drop(record, raise_label=1, drop_label=2)
            
    # insert negative data
    print(f'### Insert Negative.')
    negative_batch = 100
    W = int(cf.FS_PREPROCESS * cf.CUT_DURATION)
    for i in tqdm.trange(int(np.ceil(len(train_negative_paths)/negative_batch))):
        negative_data = []
        for path in train_negative_paths[i*negative_batch:(i+1)*negative_batch]:
            data = pickle.load(open(path, 'rb'))
            negative_data.append(data[None,:W,:])
        negative_data = np.concatenate(negative_data, axis=0)
        train_dataset.insert_negativa_data(negative_data, label=0)
    for i in tqdm.trange(int(np.ceil(len(test_negative_paths)/negative_batch))):
        negative_data = []
        for path in test_negative_paths[i*negative_batch:(i+1)*negative_batch]:
            data = pickle.load(open(path, 'rb'))
            negative_data.append(data[None,:W,:])
        negative_data = np.concatenate(negative_data, axis=0)
        test_dataset.insert_negativa_data(negative_data, label=0)
    
    train_dataset.augment(method=cf.AUG_METHOD)
    test_dataset.augment(method=None)
    train_dataloader = DataLoader(train_dataset, batch_size=cf.batch_size,
        shuffle=True, pin_memory=True, num_workers=0, worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=cf.batch_size,
        shuffle=True, pin_memory=True, num_workers=0, worker_init_fn=worker_init_fn)
    
    return train_dataloader, test_dataloader


def build_dataloader_pilot_move() -> Tuple[DataLoader, DataLoader]:
    ''' Build train and test dataloader in pilot study for Move.
    '''
     # load task_list
    task_list_id = 'TL3wni1oq3'
    task_list = fu.load_task_list_with_users(task_list_id)
    assert task_list is not None
    
    # build the dataset and dataloader
    train_users = ['lzj']
    test_users = ['lzj2']
    train_days = [1]
    test_days = [2]
    train_negative_paths = []
    test_negative_paths = []
    for day in train_days:
        train_negative_paths.extend(glob(f'../data/negative/day{day}/*.pkl'))
    for day in test_days:
        test_negative_paths.extend(glob(f'../data/negative/day{day}/*.pkl'))
    train_dataset = Dataset()
    test_dataset = Dataset()
    
    # insert positive data
    print(f'### Insert positive data.')
    record_info = []
    for task_name, label in (('Move10cm', 1), ('Move20cm', 2), ('Move30cm', 3), ('Move40cm', 4), ('Move50cm', 5)):
        for task in task_list['tasks']:
            if task['name'] == task_name: break
        assert task['name'] == task_name
        for subtask in task['subtasks']:
            record_dict = subtask['record_dict']
            for user_name, record_id in record_dict.items():
                if user_name not in train_users and user_name not in test_users: continue
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
        else: test_dataset.insert_record(record, label)
        
    # insert negative data
    print(f'### Insert negative data.')
    negative_batch = 100
    W = int(cf.FS_PREPROCESS * cf.CUT_DURATION)
    for i in tqdm.trange(int(np.ceil(len(train_negative_paths)/negative_batch))):
        negative_data = []
        for path in train_negative_paths[i*negative_batch:(i+1)*negative_batch]:
            data = pickle.load(open(path, 'rb'))
            negative_data.append(data[None,:W,:])
        negative_data = np.concatenate(negative_data, axis=0)
        train_dataset.insert_negativa_data(negative_data, label=0)
    for i in tqdm.trange(int(np.ceil(len(test_negative_paths)/negative_batch))):
        negative_data = []
        for path in test_negative_paths[i*negative_batch:(i+1)*negative_batch]:
            data = pickle.load(open(path, 'rb'))
            negative_data.append(data[None,:W,:])
        negative_data = np.concatenate(negative_data, axis=0)
        test_dataset.insert_negativa_data(negative_data, label=0)
    
    train_dataset.augment(method=cf.AUG_METHOD)
    test_dataset.augment(method=None)
    train_dataloader = DataLoader(train_dataset, batch_size=cf.BATCH_SIZE,
        shuffle=True, pin_memory=True, num_workers=0, worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=cf.BATCH_SIZE,
        shuffle=True, pin_memory=True, num_workers=0, worker_init_fn=worker_init_fn)
    
    return train_dataloader, test_dataloader


def main():
    # config parameters
    model = Model4()
    task_list_id = 'TLnmdi15b8'
    n_classes = cf.N_CLASSES
    class_names = cf.CLASS_NAMES
    n_epochs = cf.N_EPOCHS
    learning_rate = cf.LEARNING_RATE
    super_batch = cf.SUPER_BATCH
    batch_size = cf.BATCH_SIZE
    warmup_steps = cf.WARMUP_STEPS
    log_steps = cf.LOG_STEPS
    eval_steps = cf.EVAL_STEPS
    model_save_dir = f'{cf.MODEL_ROOT}/{cf.MODEL_NAME}'
    log_save_dir = f'{cf.LOG_ROOT}/{cf.MODEL_NAME}'
    
    # build dataloaders
    train_dataloader, test_dataloader = build_dataloader_pilot_move()
    
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
    train_steps = len(train_dataloader) * n_epochs - warmup_steps
    train_scheduler = optim.lr_scheduler.LinearLR(optimizer,
        start_factor=1.0, end_factor=1/train_steps, total_iters=train_steps)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer,
        schedulers=[warmup_scheduler, train_scheduler], milestones=[warmup_steps])
    
    # train process
    max_acc = -1.0
    train_log, accs, matrices = [], [], []
    optimizer.zero_grad()
    tic = time.perf_counter()
    
    for epoch in range(n_epochs):
        if (epoch+1) % cf.GC_STEPS == 0: gc.collect()
        train_loss = 0.0
        data, label = [], []
        for i, batch in enumerate(train_dataloader):
            data.append(batch['data'])
            label.append(batch['label'])
            if (i+1) % super_batch != 0 and (i+1) != len(train_dataloader):
                continue
            data = feature2(torch.concat(data,dim=0).transpose(1,2)).to(cf.DEVICE)
            label = torch.concat(label, dim=0).to(cf.DEVICE)
            for j in range(int(np.ceil(data.shape[0]/batch_size))):
                output: torch.Tensor = model(data[j*batch_size:(j+1)*batch_size,:,:])
                loss: torch.Tensor = train_criterion(output, label[j*batch_size:(j+1)*batch_size])
                loss.backward()
                train_loss += loss.detach().item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            data, label = [], []
        
        if cf.AUG_METHOD is not None:
            train_dataloader.augment(method=cf.AUG_METHOD)
        train_loss /= len(train_dataloader)
        lr = optimizer.param_groups[0]['lr']
        train_log.append({"epoch": epoch, "lr": lr, "loss": train_loss})
        
        if (epoch+1) % log_steps == 0:
            print(f'epoch: {epoch}, lr: {lr:.3e}, loss: {train_loss:.3e}')
            logger.add_scalar('Learning Rate', lr, epoch)
            logger.add_scalar('Train Loss', train_loss, epoch)
            
        if (epoch+1) % eval_steps == 0:  # test model on testing dataset
            model.eval()
            class_correct = np.zeros(n_classes, dtype=np.int32)
            class_total = np.zeros(n_classes, dtype=np.int32)
            matrix = np.zeros((n_classes, n_classes), dtype=np.int32)
            test_loss = 0.0
            
            with torch.no_grad():  
                data, label = [], []
                for i, batch in enumerate(test_dataloader):
                    data.append(batch['data'])
                    label.append(batch['label'])
                    if (i+1) % super_batch != 0 and (i+1) != len(test_dataloader):
                        continue
                    data = feature2(torch.concat(data,dim=0).transpose(1,2)).to(cf.DEVICE)
                    label = torch.concat(label, dim=0).to(cf.DEVICE)
                    for j in range(int(np.ceil(data.shape[0]/batch_size))):
                        output: torch.Tensor = model(data[j*batch_size:(j+1)*batch_size,:,:])
                        label_batch = label[j*batch_size:(j+1)*batch_size]
                        test_loss: torch.Tensor = test_criterion(output, label_batch)
                        _, predicted = torch.max(output, dim=1)
                        c = (predicted == label_batch)
                        for k in range(len(label_batch)):
                            l = label_batch[k]
                            matrix[l.item(), predicted[k].item()] += 1
                            is_correct = c[k]
                            class_correct[l] += is_correct
                            class_total[l] += 1
                    data, label = [], []
            
            test_loss /= len(test_dataloader)
            logger.add_scalar("Test Loss", test_loss, epoch)
            model.train()
            matrices.append(matrix)
            acc = np.diag(matrix).sum() / matrix.sum()
            accs.append(acc)
            if acc > max_acc:
                max_acc = acc
                torch.save(model.state_dict(), os.path.join(model_save_dir, 'best.model'))
            torch.save(model.state_dict(), os.path.join(model_save_dir, f'{epoch}.model'))
            fpr = np.sum(matrix[0,1:]) / np.sum(matrix[0,:])
            logger.add_scalar('False Positive Rate', fpr, epoch)
            recall = np.sum(np.diag(matrix)[1:] / np.sum(matrix[1:,:]))
            logger.add_scalar('Accuracy of Positive', recall, epoch)
            print(f'+{"-"*79}+')
            for i in range(n_classes):
                row = matrix[i,:].copy()
                row = 100 * row / np.sum(row)
                print(f'| Accuracy of {class_names[i]:12s}: {row[i]:.2f} % | ', end='')
                print(' '.join([f'{item:.2f}'.rjust(5) for item in row]) + ' |')
                logger.add_scalar(f'Accuracy of {class_names[i]}', row[i], epoch)
            print(f'+{"-"*79}+')
            print(f'| False Positive Rate     : {f"{100*fpr:.3f}".rjust(6)} %' + ' '*44 + '|')
            print(f'| Recall                  : {f"{100*recall:.3f}".rjust(6)} %' + ' '*44 + '|')
            print(f'+{"-"*79}+')
            logger.add_scalar('Accuracy', acc, epoch)
    toc = time.perf_counter()
    general_matrix = np.mean(matrices[-4:], axis=0)
    print(f'General matrix:')
    for i in range(n_classes):
        row = general_matrix[i,:].copy()
        row = 100 * row / np.sum(row)
        print(f'    Accuracy of {class_names[i]:12s}: {row[i]:.2f} % | ', end='')
        print(' '.join([f'{item:.2f}'.rjust(5) for item in row]))
    fpr = np.sum(general_matrix[0,1:]) / np.sum(general_matrix[0,:])
    recall = np.sum(np.diag(general_matrix)[1:] / np.sum(general_matrix[1:,:]))
    print(f'General FPR: {100*fpr:.3f} %')
    print(f'General recall: {100*recall:.3f} %')
    print(f'General acc: {(100*np.mean(accs[-4:])):.1f} %')
    print(f'Training time: {toc-tic:.1f} s')
    

if __name__ == '__main__':
    random.seed(cf.RAND_SEED)
    np.random.seed(cf.RAND_SEED)
    torch.manual_seed(cf.RAND_SEED)
    fu.check_cwd()
    main()
    