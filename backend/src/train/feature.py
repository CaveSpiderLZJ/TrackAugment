import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from typing import Tuple, List, Dict

import torch

import config as cf
import file_utils as fu
from data_process.filter import Butterworth
from data_process.record import Record


filters = [Butterworth(fs=cf.FS_TRAIN, cut=[0.22, 8], mode='bandpass'),
    Butterworth(fs=cf.FS_TRAIN, cut=[8, 32], mode='bandpass'),
    Butterworth(fs=cf.FS_TRAIN, cut=32, mode='highpass')]


def feature1(x:np.ndarray) -> np.ndarray:
    ''' Generate input data features.
    args:
        x: np.ndarray[(batch_size, channels, length)], the model input.
    returns:
        np.ndarray, the generated features.
    '''
    res = [x] + [filter.filt(x, axis=2) for filter in filters]
    res = np.concatenate(res, axis=1)
    return torch.from_numpy(res.astype(np.float32))


def feature2(x:np.ndarray) -> np.ndarray:
    ''' Filter the data and put the same axis together.
    args:
        x: np.ndarray[(batch_size, channels, length)], the model input.
    returns:
        np.ndarray, the generated features.
    '''
    n_channel = x.shape[1]
    res = signal.detrend(x, axis=2)
    res = [res] + [filter.filt(res, axis=2) for filter in filters]
    res = np.concatenate(res, axis=1)
    indices = np.concatenate([list(range(i, i+4*n_channel, n_channel)) for i in range(n_channel)])
    return torch.from_numpy(res[:,tuple(indices),:].astype(np.float32))


if __name__ == '__main__':
    fu.check_cwd()
    task_list_id = 'TL13r912je'
    task_id = 'TKfvdarv6k'
    subtask_id = 'STxw6enkhj'
    record_id = 'RD6fu3gmp6'
    record_path = fu.get_record_path(task_list_id, task_id, subtask_id, record_id)
    record = Record(record_path, n_sample=21)
    imu = record.cutted_imu_data
    imu = np.concatenate([imu['acc'], imu['gyro']], axis=2)
    imu = imu.transpose(0,2,1)
    imu = feature2(imu)
    for i in range(4):
        plt.subplot(4,1,i+1)
        plt.plot(imu[0,12+i,:])
    plt.show()