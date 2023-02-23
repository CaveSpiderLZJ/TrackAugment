import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, List, Dict

import torch

import config as cf
from data_process.filter import Butterworth


filters = [Butterworth(fs=cf.FS_TRAIN, cut=8, mode='lowpass'),
    Butterworth(fs=cf.FS_TRAIN, cut=[8, 32], mode='bandpass'),
    Butterworth(fs=cf.FS_TRAIN, cut=32, mode='highpass')]


def feature(x:np.ndarray) -> np.ndarray:
    ''' Generate input data features.
    args:
        x: np.ndarray[(batch_size, channels, length)], the model input.
    returns:
        np.ndarray, the generated features.
    '''
    res = [x] + [filter.filt(x, axis=2) for filter in filters]
    res = np.concatenate(res, axis=1)
    return torch.from_numpy(res.astype(np.float32))


if __name__ == '__main__':
    pass