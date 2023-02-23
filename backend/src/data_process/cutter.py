import numpy as np
from matplotlib import pyplot as plt

import config as cf
from data_process.filter import Butterworth


class PeakCutter:
    
    
    def __init__(self, n_sample:int, window_length:int, noise:int, fs:float, fs_stop:float) -> None:
        ''' Init the cutter parameters.
            Generally, the peak cutter will use a Butterworth low pass filter to find the action
            peak, and cut a window around the action peak, with some noise added.
        args:
            n_sample: int, the number of actions in the data time series.
            window_length: int, the cut window length.
            noise: int, the window position noise.
            fs: float, the data sampling frequency.
            fs_stop: float, the stop frequency of the low pass filter.
        '''
        self.n_sample = n_sample
        self.window_length = window_length
        self.noise = noise
        self.filter = Butterworth(fs=fs, cut=fs_stop, mode='lowpass', order=4)
    
    
    def cut_range(self, data:np.ndarray) -> np.ndarray:
        ''' Cut the data with a specific window length around the action peaks.
            Find the above cut ranges.
        args:
            data: np.ndarray[(N, 3), np.float32], the imu data (acc, gyro or linear_acc).
        returns:
            np.ndarray[(n_sample, 2), np.int32], the start and end indices of the cut ranges.
        '''
        N = data.shape[0]
        window_length = self.window_length
        noise = self.noise
        norm = np.sqrt(np.sum(np.square(data), axis=1))
        filtered_norm = self.filter.filt(norm, axis=0)
        sample_length: float = N / self.n_sample
        assert window_length + 1 < sample_length
        sample_ranges = np.empty((self.n_sample, 2), dtype=np.int32)
        for i in range(self.n_sample):
            sample_ranges[i,:] = int(i*sample_length), int((i+1)*sample_length)
        cut_ranges = np.empty_like(sample_ranges)
        for i in range(self.n_sample):
            start, end = sample_ranges[i,:]
            peak = start + np.argmax(filtered_norm[start:end])
            start_window = peak - window_length//2
            if noise > 0: start_window += np.random.randint(-noise//2, noise//2)
            start_window = max(min(start_window, N-window_length), 0)
            cut_ranges[i,:] = start_window, start_window + window_length
        return cut_ranges
    

if __name__ == '__main__':
    pass