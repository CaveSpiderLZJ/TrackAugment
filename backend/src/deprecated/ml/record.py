from decimal import DecimalTuple
from email.iterators import _structure
from email.policy import strict
import os
import json
import struct
import random
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Set, Tuple


'''
Record is parsed from the continuous collected data which may includes several action instances.
Use this class to parse record data from files.
'''
class Record:
    def __init__(self, motion_path, timestamp_path, record_id,
            group_id:int=0, group_name:str='', description:str='',
            cutter=None, cut_data:bool=True):
        ''' Init some parameters.
        '''
        self.motion_path = motion_path  # xxx/Motion_xxx.bin
        self.timestamp_path = timestamp_path    # xxx/Timestamp_xxx.json
        self.record_id = record_id
        self.group_id = group_id
        self.group_name = group_name
        self.description = description
        self.cutter = cutter
        
        # data
        self.data_labels = ('acc', 'mag', 'gyro', 'linear_acc')
        self.data = None
        self.timestamps = None
        self.cut_data = None

        # self.data, self.timestamps
        self.load_from_file(motion_path, timestamp_path)
        if cut_data:
            self.align_data()
            self.cut()


    def cut(self):
        ''' Use self.cutter to cut the data.
        '''
        self.cut_data = self.cutter.cut(self.data, self.timestamps)


    def align_data(self):
        ''' If the data frequency of all sensors do not match,
            downsample them to align with the lowest frequency.
            Also make sure all sensor data have the same length after aligning.
        '''
        
        data, data_labels = self.data, self.data_labels
        data_t = {label: data[label]['t'] for label in data_labels}
        calc_freq = lambda t: (1e9 * (len(t)-1) / (t[-1]-t[0]))
        data_freq = {label: calc_freq(data_t[label]) for label in data_labels}
        
        freqs = list(data_freq.values())
        min_freq, max_freq = np.min(freqs), np.max(freqs)
        thres = 1.1
        if max_freq / min_freq <= thres:
            print('No need for resampling.')
            return
        
        min_freq = 1e30
        min_freq_label = None
        for label, freq in data_freq.items():
            if freq < min_freq:
                min_freq_label, min_freq = label, freq
                
        for label in data_labels:
            if data_freq[label] / min_freq > thres:
                # downsampling
                data[label] = self.down_sample(data[label], data[min_freq_label])
            
        # after resampling
        data_t = {label: data[label]['t'] for label in data_labels}
        data_freq = {label: calc_freq(data_t[label]) for label in data_labels}
        print(f'### resampled frequencies: ', end='')
        for label, freq in data_freq.items():
            print(f'{freq:.1f} ', end='')
        print('Hz')
                    
        # align to the same length
        min_len = np.min([len(data_t[label]) for label in data_labels])
        for label in data_labels:
            sensor_data = data[label]
            if len(data_t[label]) <= min_len: continue
            sensor_data['x'] = sensor_data['x'][:min_len]
            sensor_data['y'] = sensor_data['y'][:min_len]
            sensor_data['z'] = sensor_data['z'][:min_len]
            sensor_data['t'] = sensor_data['t'][:min_len]
            data[label] = sensor_data
        
        self.data = data
        
    
    def down_sample(self, src:dict, ref:dict) -> dict:
        ''' Downsampling src so that it will have the same frequency as refer.
        args:
            src: dict, like {'x': [...], 'y': [...], 'z': [...], 't': [...]},
                all lists are 1D np.ndarray
            ref: the same as arc, with lower frequency
        return:
            A dict, downsampled src.
        '''
        idxs = []
        src_t, ref_t = src['t'], ref['t']
        src_len, ref_len = len(src_t), len(ref_t)
        # preprocess: ensure srt_t[0] < ref_t[idx_start] < ref_t[idx_end] < src_t[-1]
        idx_start, idx_end = 0, ref_len - 1
        while ref_t[idx_start] <= src_t[0] and idx_start < ref_len - 1:
            idx_start += 1
        while ref_t[idx_end] >= src_t[-1] and idx_end > 0:
            idx_end -= 1
        src_idx, ref_idx = 0, idx_start
        while ref_idx <= idx_end:
            t = ref_t[ref_idx]
            while src_t[src_idx] < t:
                src_idx += 1
            # determine which idx is closer
            if src_t[src_idx] - t < t - src_t[src_idx-1]:
                idxs.append(src_idx)
            else: idxs.append(src_idx - 1)
            ref_idx += 1
        idxs = np.array(idxs)
        return {'x': src['x'][idxs], 'y': src['y'][idxs],
                'z': src['z'][idxs], 't': src['t'][idxs]}


    def load_from_file(self, motion_path:str, timestamp_path:str):
        ''' Parse motion data from motion_path file, and store in self.data.
            Parse timestamps from timestamp_path file, and store in self.timestamps
        args:
            motion_path: str, like 'xxx/Motion_xxx.bin'.
            timestamp_path: str, like 'xxx/Timestamp_xxx.json'
        attrs:
            self.data: Dict[sensor_type:str, Dict[axis:str, np.ndarray]].
                sensor_type in {'acc', 'acc_un', 'gyro', 'gyro_un', 'mag', 'mag_un',
                    'linear_acc', 'gravity', 'rotation'}.
                axis in {'x', 'y', 'z', 'u', 'v', 'w'}, 6D in total, uvw are optional.
            self.timestamps: List[int].
        '''
        assert(motion_path.endswith('.bin'))
        assert(timestamp_path.endswith('.json'))
        
        # data structure configuration
        sensor_types: Tuple[str] = ('acc', 'acc_un', 'gyro', 'gyro_un', 'mag', 'mag_un',
                        'linear_acc', 'gravity', 'rotation')
        axes: Tuple[str] = ('x', 'y', 'z', 'u', 'v', 'w')
        sensor_dimensions: Dict[str, int] = {'acc':3, 'acc_un':6, 'gyro':3, 'gyro_un':6,
                'mag':3, 'mag_un':6, 'linear_acc':3, 'gravity':3, 'rotation':4}
        
        data = {}
                
        with open(motion_path, 'rb') as f:
            for sensor_type in sensor_types:
                dimension = sensor_dimensions[sensor_type]
                sensor_data = {'t': []}
                for i in range(dimension):
                    sensor_data[axes[i]] = []
                size, = struct.unpack('>i', f.read(4))
                for _ in range(size):
                    values = struct.unpack(f'>{"f"*dimension}q', f.read(8+dimension*4))
                    for i in range(dimension):
                        sensor_data[axes[i]].append(values[i])
                    sensor_data['t'].append(values[-1])
                for i in range(dimension):
                    sensor_data[axes[i]] = np.array(sensor_data[axes[i]], dtype=float)
                sensor_data['t'] = np.array(sensor_data['t'], dtype=int)
                data[sensor_type] = sensor_data 
        
        self.data = data
        self.timestamps = json.load(open(timestamp_path, 'r'))
        
        
if __name__ == '__main__':
    pass    