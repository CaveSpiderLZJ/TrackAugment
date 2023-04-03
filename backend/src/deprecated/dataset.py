import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, List, Dict

import torch
import torch.utils.data

import config as cf
import data_process.augment as aug
from data_process.record import Record


class Dataset(torch.utils.data.Dataset):
    
    
    def __init__(self, group_id_to_name:Dict[int, str],
            split_mode:str='train', train_ratio:float=0.8) -> None:
        ''' Init the dataset data structures.
        args:
            group_id_to_name: Dict[int, str], the mapping of group ids to group names.
                If there are N groups in total, the group ids must be [0, ..., N-1]
            split_mode: str, in {'train', 'test'}, defualt = 'train'.
            train_ratio: float, the ratio of training set, default = 0.8.
        '''
        self.size = 0
        self.positive_size = 0
        self.group_id_to_name = group_id_to_name
        self.split_mode = split_mode
        self.train_ratio = train_ratio
        self.track_data = {'center_pos': None, 'center_rot': None, 'marker_pos': None}
        self.imu_data = {'acc': None, 'gyro': None}
        self.labels = None
        self.reindex_map = None
            

    def __len__(self) -> int:
        if self.split_mode == 'train':
            return int(self.size * self.train_ratio)
        return self.size - int(self.size * self.train_ratio) 
    
    
    def __getitem__(self, idx:int) -> Dict[str, np.ndarray]:
        ''' Get an action sample data.
        args:
            idx: int, the index of the action sample.
                the idx will be reindexed before retrieving the actual sample data.]
        '''
        # no track augmentation yet, directly return imu data
        train_size = int(self.size * self.train_ratio)
        if self.split_mode != 'train': idx += train_size
        if self.reindex_map is not None: idx = self.reindex_map[idx]
        # TODO: no track data used yet
        sample = np.concatenate([self.imu_data['acc'][idx,:,:], self.imu_data['gyro'][idx,:,:]], axis=1)
        sample = aug.down_sample_by_step(sample, axis=0, step=(cf.FS_PREPROCESS//cf.FS_TRAIN))
        return {'data': sample, 'label': self.labels[idx]}
    
    
    def insert_records(self, records:List[Record], group_ids:List[int]) -> None: 
        ''' Insert a list of records.
        args:
            records: List[Record], a list or records.
            group_ids: List[int], the group ids of these records,
                must have the same length as records.
        '''
        # a simple version based on insert_record()
        for record, group_id in zip(records, group_ids):
            self.insert_record(record, group_id)
    
    
    def insert_record(self, record:Record, group_id:int) -> None:
        ''' Insert a single record with all samples have the same group id.
            NOTE: always insert negative (group_id == 0) after positive data (group_id != 0).
        args:
            record: Record, the action record.
            group_id: int, the group id of the record.
        '''
        # insert imu data
        imu_data = self.imu_data
        cutted_imu_data = record.cutted_imu_data
        cnt = cutted_imu_data['acc'].shape[0]
        if imu_data['acc'] is None:
            for key in ('acc', 'gyro'):
                imu_data[key] = cutted_imu_data[key]
        else:
            for key in ('acc', 'gyro'):
                imu_data[key] = np.concatenate([imu_data[key], cutted_imu_data[key]], axis=0)
        self.imu_data = imu_data
        # insert track data
        if group_id != 0:
            track_data = self.track_data
            cutted_track_data = record.cutted_track_data
            if track_data['center_pos'] is None:
                for key in ('center_pos', 'center_rot', 'marker_pos'):
                    track_data[key] = cutted_track_data[key]
            else:
                for key in ('center_pos', 'center_rot', 'marker_pos'):
                    track_data[key] = np.concatenate([track_data[key], cutted_track_data[key]], axis=0)
            self.track_data = track_data
            self.positive_size += cnt
        # insert labels
        if self.labels is None:
            self.labels = np.zeros(cnt, dtype=np.int64) + group_id
        else: self.labels = np.concatenate([self.labels, np.zeros(cnt, dtype=np.int64) + group_id])
        self.size += cnt
        
        
    def insert_record_raise_drop(self, record:Record, group_ids:Tuple[int]) -> None:
        ''' An adapter for insert raise and drop data only.
        args:
            record: Record, the action record.
            group_ids: Tuple[int], len == 2, the group ids of raise and drop.
        '''
        assert len(group_ids) == 2
        # insert imu data
        imu_data = self.imu_data
        cutted_imu_data = record.cutted_imu_data
        cnt = cutted_imu_data['acc'].shape[0]
        if imu_data['acc'] is None:
            for key in ('acc', 'gyro'):
                imu_data[key] = cutted_imu_data[key]
        else:
            for key in ('acc', 'gyro'):
                imu_data[key] = np.concatenate([imu_data[key], cutted_imu_data[key]], axis=0)
        self.imu_data = imu_data
        # insert track data
        track_data = self.track_data
        cutted_track_data = record.cutted_track_data
        if track_data['center_pos'] is None:
            for key in ('center_pos', 'center_rot', 'marker_pos'):
                track_data[key] = cutted_track_data[key]
        else:
            for key in ('center_pos', 'center_rot', 'marker_pos'):
                track_data[key] = np.concatenate([track_data[key], cutted_track_data[key]], axis=0)
        self.track_data = track_data
        if self.labels is None:
            self.labels = np.tile(group_ids, cnt//2+1)[:cnt]
        else: self.labels = np.concatenate([self.labels, np.tile(group_ids, cnt//2+1)[:cnt]])
        self.positive_size += cnt
        self.size += cnt
        
        
    def shuffle(self) -> None:
        ''' Shuffle the reindex mapping.
            NOTE: always call shuffle after inserting records.
        '''
        self.reindex_map = np.arange(self.size, dtype=np.int32)
        np.random.shuffle(self.reindex_map)
        
    
    def set_split_mode(self, split_mode:str) -> None:
        ''' Set a new split mode to retrieve different data.
        args:
            split_mode: str, in {'train', 'test'}.
        '''
        self.split_mode = split_mode
        
        
class DataLoader(torch.utils.data.DataLoader):
    
    
    def __init__(self, dataset:Dataset, *args, **kwargs) -> None:
        super(DataLoader, self).__init__(dataset, *args, **kwargs)
        
        
    def set_split_mode(self, split_mode:str) -> None:
        self.dataset.set_split_mode(split_mode)
    
    
if __name__ == '__main__':
    pass