import os
import json
import time
import struct
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from matplotlib import pyplot as plt

import file_utils as fu
import config as cf
from data_process import augment as aug
from data_process.cutter import PeakCutter


class Record:
    
    
    def __init__(self, record_path:str) -> None:
        assert os.path.exists(record_path)
        self.record_path:str = record_path
        self.motion_path:str = None
        self.timestamp_path:str = None
        self.track_path:str = None
        self.track_data:Dict[str, np.ndarray] = None
        self.imu_data:Dict[str, np.ndarray] = None
        for _, _, file_names in os.walk(record_path):
            for file_name in file_names:
                if file_name.startswith('Motion'):
                    self.motion_path = os.path.join(record_path, file_name)
                elif file_name.startswith('Timestamp'):
                    self.timestamp_path = os.path.join(record_path, file_name)
                elif file_name == 'track.csv':
                    self.track_path = os.path.join(record_path, file_name)
        self.load_track_data()
        self.load_imu_data()
        self.align_imu_data_frequency()
        self.align_track_and_imu_data()
        self.cut_data()
        
        
    def load_track_data(self):
        track_path = self.track_path
        if track_path is None or not os.path.exists(track_path): return
        track_data = pd.read_csv(track_path, header=[1,3,4])
        timestamps = track_data[('Unnamed: 1_level_0', 'Unnamed: 1_level_1', 'Time (Seconds)')].to_numpy(np.float32)
        center_pos = [track_data[('SmartPhone', 'Position', axis)].to_numpy(
            np.float32) for axis in ('X', 'Y', 'Z')]
        center_pos = 1e-3 * np.column_stack(center_pos) # mm -> m
        center_pos[np.isnan(center_pos)] = 0.0
        center_rot = [track_data[('SmartPhone', 'Rotation', axis)].to_numpy(
            np.float32) for axis in ('X', 'Y', 'Z', 'W')]
        center_rot = 1e-3 * np.column_stack(center_rot) # mm -> m
        center_rot[np.isnan(center_rot)] = 1.0
        marker_pos = []
        for i in range(1,7):
            pos = [track_data[(f'SmartPhone:Marker{i}', 'Position', axis)].to_numpy(
                np.float32) for axis in ('X', 'Y', 'Z')]
            marker_pos.append(np.column_stack(pos)[np.newaxis,:,:])
        marker_pos = 1e-3 * np.concatenate(marker_pos, axis=0) # mm -> m
        marker_pos[np.isnan(marker_pos)] = 0.0
        self.track_data = {'timestamps': timestamps, 'center_pos': center_pos,
            'center_rot': center_rot, 'marker_pos': marker_pos}
        
    
    def load_imu_data(self):
        motion_path = self.motion_path
        timestamp_path = self.timestamp_path
        if motion_path is None or not os.path.exists(motion_path): return
        if timestamp_path is None or not os.path.exists(timestamp_path): return
        # data structure configuration
        sensor_types: Tuple[str] = ('acc', 'acc_un', 'gyro', 'gyro_un',
            'mag', 'mag_un', 'linear_acc', 'gravity', 'rotation')
        axis_labels: Tuple[str] = ('x', 'y', 'z', 'u', 'v', 'w')
        sensor_dimensions: Dict[str, int] = {'acc':3, 'acc_un':6, 'gyro':3,
            'gyro_un':6, 'mag':3, 'mag_un':6, 'linear_acc':3, 'gravity':3, 'rotation':4}
        self.imu_data = {}
        f = open(motion_path, 'rb')
        for sensor_type in sensor_types:
            dimension = sensor_dimensions[sensor_type]
            sensor_data = {'t': []}
            for i in range(dimension):
                sensor_data[axis_labels[i]] = []
            size, = struct.unpack('>i', f.read(4))
            for _ in range(size):
                values = struct.unpack(f'>{"f"*dimension}q', f.read(8+dimension*4))
                for i in range(dimension):
                    sensor_data[axis_labels[i]].append(values[i])
                sensor_data['t'].append(values[-1])
            for i in range(dimension):
                sensor_data[axis_labels[i]] = np.array(sensor_data[axis_labels[i]], dtype=np.float32)
            sensor_data['t'] = np.array(sensor_data['t'], dtype=np.int64)
            self.imu_data[sensor_type] = sensor_data
        f.close()
        timestamps = json.load(open(timestamp_path, 'r'))
        self.imu_data['timestamps'] = np.array(timestamps, dtype=np.int64)
        
        
    def align_imu_data_frequency(self):
        ''' Align the frequcies of imu sensors to the same as the track data.
            Also, align the number of all imu sensors to the mininum length.
        '''
        imu_data = self.imu_data
        sensor_types: Tuple[str] = ('acc', 'acc_un', 'gyro', 'gyro_un',
            'mag', 'mag_un', 'linear_acc', 'gravity', 'rotation')
        min_length = int(1e10)
        for sensor in sensor_types:
            sensor_data = imu_data[sensor]
            resample_ratio = cf.FS_PREPROCESS / cf.FS_IMU[sensor]
            for axis in sensor_data.keys():
                axis_data = sensor_data[axis]
                resampled_data = aug.resample(axis_data, axis=0, ratio=resample_ratio)
                if resampled_data.shape[0] < min_length:
                    min_length = resampled_data.shape[0]
                sensor_data[axis] = resampled_data
            imu_data[sensor] = sensor_data
        for sensor in sensor_types:
            sensor_data = imu_data[sensor]
            for axis in sensor_data.keys():
                sensor_data[axis] = sensor_data[axis][:min_length]
            imu_data[sensor] = sensor_data
        self.imu_data = imu_data
        
    
    def align_track_and_imu_data(self):
        ''' Align track and imu data in time domain,
            and discard the track data that are out of imu data range.
        '''
        imu_data = self.imu_data
        gyro = imu_data['gyro']
        gyro = np.column_stack([gyro[axis] for axis in ('x','y','z')])
        track_data = self.track_data
        center_pos = track_data['center_pos']
        center_rot = track_data['center_rot']
        marker_pos = track_data['marker_pos']
        timestamps = track_data['timestamps']
        # NOTE: should resample track data if FS_TRACK != FS_PREPROCESS
        axes = aug.calc_local_axes(marker_pos)
        generated_gyro = aug.track_to_gyro(axes, cf.FS_PREPROCESS)
        len_track = generated_gyro.shape[0]
        len_gyro = gyro.shape[0]
        step, length = 10, 1000
        min_err, off = 1e30, 0
        for i in range(0, len_track - len_gyro, step):
            err = np.mean(np.abs(generated_gyro[i:i+length] - gyro[:length]))
            if err < min_err: min_err, off = err, i
        for i in range(max(0, off-step), min(len_track-len_gyro, off+10)):
            err = np.mean(np.abs(generated_gyro[i:i+length] - gyro[:length]))
            if err < min_err: min_err, off = err, i
        self.track_data = {'timestamps': timestamps[off:off+len_gyro], 'center_pos': center_pos[off:off+len_gyro,:],
            'center_rot': center_rot[off:off+len_gyro,:], 'marker_pos': marker_pos[:,off:off+len_gyro,:]}
        
    
    def cut_data(self):
        ''' Cut track and imu data to individual action samples.
        '''
        # calculate cut ranges from gyro
        track_data = self.track_data
        imu_data = self.imu_data
        gyro = np.column_stack([imu_data['gyro'][axis] for axis in ('x','y','z')])
        window_length = int(cf.WINDOW_DURATION * cf.FS_PREPROCESS)
        cutter = PeakCutter(cf.N_SAMPLE, window_length, noise=0,
            fs=cf.FS_PREPROCESS, fs_stop=0.005*cf.FS_PREPROCESS)
        # discard the first sample
        cut_ranges = cutter.cut_range(gyro)[1:]
        # cut track and imu data
        acc = np.column_stack([imu_data['acc'][axis] for axis in ('x','y','z')])
        self.cutted_imu_data = {
            'acc': np.row_stack([acc[None,l:r,:] for l, r in cut_ranges]),
            'gyro': np.row_stack([gyro[None,l:r,:] for l, r in cut_ranges])}
        self.cutted_track_data = {
            'center_pos': np.row_stack([track_data['center_pos'][None,l:r,:] for l, r in cut_ranges]),
            'center_rot': np.row_stack([track_data['center_rot'][None,l:r,:] for l, r in cut_ranges]),
            'marker_pos': np.row_stack([track_data['marker_pos'][None,:,l:r,:] for l, r in cut_ranges])}


if __name__ == '__main__':
    task_list_id = 'TL13r912je'
    task_id = 'TKfvdarv6k'
    subtask_id = 'ST6klid59e'
    record_id = 'RDmb2zdzis'
    record_path = fu.get_record_path(task_list_id, task_id, subtask_id, record_id)
    record = Record(record_path)
    imu_data = record.imu_data
