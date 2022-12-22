import os
import json
import struct
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from matplotlib import pyplot as plt

import file_utils as fu


class Record:
    
    
    def __init__(self, record_path:str) -> None:
        self.record_path:str = record_path
        self.motion_path:str = None
        self.timestamp_path:str = None
        self.track_path:str = None
        self.track_data:Dict[str, np.ndarray] = None
        self.motion_data:Dict[str, np.ndarray] = None
        for _, _, file_names in os.walk(record_path):
            for file_name in file_names:
                if file_name.startswith('Motion'):
                    self.motion_path = os.path.join(record_path, file_name)
                elif file_name.startswith('Timestamp'):
                    self.timestamp_path = os.path.join(record_path, file_name)
                elif file_name == 'track.csv':
                    self.track_path = os.path.join(record_path, file_name)
        self.load_track_data()
        self.load_motion_data()
        
        
    def load_track_data(self):
        track_path = self.track_path
        if track_path is None or not os.path.exists(track_path): return
        track_data = pd.read_csv(track_path, header=[1,3,4])
        timestamps = track_data[('Unnamed: 1_level_0', 'Unnamed: 1_level_1', 'Time (Seconds)')].to_numpy(np.float32)
        phone_pos = [track_data[('SmartPhone', 'Position', axis)].to_numpy(
            np.float32) for axis in ('X', 'Y', 'Z')]
        phone_pos = np.column_stack(phone_pos)
        phone_pos[np.isnan(phone_pos)] = 0.0
        phone_rot = [track_data[('SmartPhone', 'Rotation', axis)].to_numpy(
            np.float32) for axis in ('X', 'Y', 'Z', 'W')]
        phone_rot = np.column_stack(phone_rot)
        phone_rot[np.isnan(phone_rot)] = 1.0
        marker_pos = []
        for i in range(1,7):
            pos = [track_data[(f'SmartPhone:Marker{i}', 'Position', axis)].to_numpy(
                np.float32) for axis in ('X', 'Y', 'Z')]
            marker_pos.append(np.column_stack(pos)[np.newaxis,:,:])
        marker_pos = np.concatenate(marker_pos, axis=0)
        marker_pos[np.isnan(marker_pos)] = 0.0
        self.track_data = {'timestamps': timestamps, 'phone_pos': phone_pos,
            'phone_rot': phone_rot, 'marker_pos': marker_pos}
        
    
    def load_motion_data(self):
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
        self.motion_data = {}
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
            self.motion_data[sensor_type] = sensor_data
        f.close()
        timestamps = json.load(open(timestamp_path, 'r'))
        self.motion_data['timestamps'] = np.array(timestamps, dtype=np.int64)


if __name__ == '__main__':
    task_list_id = 'TL13r912je'
    task_id = 'TKfvdarv6k'
    subtask_id = 'ST6klid59e'
    record_id = 'RDmb2zdzis'
    record_path = fu.get_record_path(task_list_id, task_id, subtask_id, record_id)
    record = Record(record_path)
    motion_data = record.motion_data
