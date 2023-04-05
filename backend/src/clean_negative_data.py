import os
import time
import pickle
import numpy as np
import multiprocessing as mp
from glob import glob
from matplotlib import pyplot as plt
from typing import Tuple, List, Dict

import config as cf
from data_process import augment as aug


NEGATIVE_DATA_ROOT = '/data2/qinyue/ProxiMic_Honor/data/data3_dailylife'
MEDIA_ROOT = '../data/media'
CLEAN_NEGATIVE_DATA_ROOT = f'../data/negative'


def clean_negative_data(path:str) -> None:
    ''' Parse raw negative data (75 days imu recording), filter out data without fluctuations,
        and save data to NEGATIVE_ROOT.
    '''
    day = path.split('/')[-2]
    file_name = path.split('/')[-1].split('.')[0]
    save_path = f'{CLEAN_NEGATIVE_DATA_ROOT}/{day}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    acc_t, acc_data = [], []
    gyro_t, gyro_data = [], []
    with open(path, 'r') as f:
        cnt = 0
        for line in f:
            cnt += 1
            tokens = line.split()
            if len(tokens) != 5: continue
            if tokens[0] == 'ACCELEROMETER':
                acc_t.append(int(tokens[1]))
                acc_data.append((float(tokens[2]), float(tokens[3]), float(tokens[4])))
            elif tokens[0] == 'GYROSCOPE':
                gyro_t.append(int(tokens[1]))
                gyro_data.append((float(tokens[2]), float(tokens[3]), float(tokens[4])))
            if cnt % 10000000 == 0:
                print(f'{path}: {cnt}')
    acc_t = np.array(acc_t, dtype=np.int64)
    acc_data = np.array(acc_data, dtype=np.float32)
    gyro_t = np.array(gyro_t, dtype=np.int64)
    gyro_data = np.array(gyro_data, dtype=np.float32)
    acc_fs = acc_t.shape[0] * 1e9 / (acc_t[-1] - acc_t[0])
    gyro_fs = gyro_t.shape[0] * 1e9 / (gyro_t[-1] - gyro_t[0])
    acc_data = aug.resample(acc_data, axis=0, ratio=(cf.FS_PREPROCESS/acc_fs))
    gyro_data = aug.resample(gyro_data, axis=0, ratio=(cf.FS_PREPROCESS/gyro_fs))
    length = min(acc_data.shape[0], gyro_data.shape[0])
    acc_data = acc_data[:length,:]
    gyro_data = gyro_data[:length,:]
    W = cf.FS_PREPROCESS * 3
    for i in range(length // W):
        gyro = gyro_data[i*W:(i+1)*W,:]
        if np.std(gyro) < 0.1: continue
        acc = acc_data[i*W:(i+1)*W,:]
        sample = np.concatenate([acc, gyro], axis=1)
        pickle.dump(sample, open(f'{save_path}/{file_name}_{i}.pkl', 'wb'))
    

if __name__ == '__main__':
    paths = glob(f'{NEGATIVE_DATA_ROOT}/day*/*.sensor')
    tic = time.perf_counter()
    for path in paths:
        try:
            clean_negative_data(path)
        except:
            print(f'### Error: {path}')
            continue
    toc = time.perf_counter()
    print(f'### time cost: {(toc-tic):.3f} s')
    
    
