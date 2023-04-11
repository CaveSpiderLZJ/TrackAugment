import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, List, Dict

import torch
import torch.utils.data

import config as cf
import data_process.augment as aug
from data_process.record import Record


class Dataset(torch.utils.data.Dataset):
    
    
    def __init__(self) -> None:
        ''' Init the dataset data structures.
        '''
        self.size = 0
        self.positive_size = 0
        self.track_data = {'center_pos': None, 'center_rot': None, 'marker_pos': None}
        self.imu_data = {'acc': None, 'gyro': None}
        self.labels = None
        self.reindex_map = None
            

    def __len__(self) -> int:
        return self.size
    
    
    def __getitem__(self, idx:int) -> Dict[str, np.ndarray]:
        ''' Get an action sample data.
        args:
            idx: int, the index of the action sample.
                the idx will be reindexed before retrieving the actual sample data.]
        '''
        # TODO: no track augmentation yet, directly return imu data
        # If idx after reindex_map >= positive size, it means the sample at idx is negative data,
        # which does not has track_data. Return imu data directly.
        if self.reindex_map is not None: idx = self.reindex_map[idx]
        w_cut = int(cf.CUT_DURATION * cf.FS_PREPROCESS)
        w_train = int(cf.TRAIN_DURATION * cf.FS_PREPROCESS)
        if cf.RANDOM_SAMPLE_EN: start = np.random.randint(0, w_cut - w_train)
        else: start = (w_cut - w_train) // 2
        sample = np.concatenate([self.imu_data['acc'][idx,start:start+w_train,:],
            self.imu_data['gyro'][idx,start:start+w_train,:]], axis=1)
        sample = aug.down_sample_by_step(sample, axis=0, step=(cf.FS_PREPROCESS//cf.FS_TRAIN))
        return {'data': sample, 'label': self.labels[idx]}
    
    
    def insert_records(self, records:List[Record], labels:List[int]) -> None: 
        ''' Insert a list of records.
        args:
            records: List[Record], a list or records.
            labels: List[int], the labels of these records,
                must have the same length as records.
        '''
        # a simple version based on insert_record()
        for record, label in zip(records, labels):
            self.insert_record(record, label)
    
    
    def insert_record(self, record:Record, label:int) -> None:
        ''' Insert a single record with all samples have the same group id.
            NOTE: always insert negative after positive data.
        args:
            record: Record, the action record.
            label: int, the label of the record.
        '''
        # insert imu data
        clean_mask = record.clean_mask
        cnt = np.sum(clean_mask)
        imu_data = self.imu_data
        cutted_imu_data = record.cutted_imu_data
        if imu_data['acc'] is None:
            imu_data['acc'] = cutted_imu_data['acc'][clean_mask,...]
            imu_data['gyro'] = cutted_imu_data['gyro'][clean_mask,...]
        else:
            imu_data['acc'] = np.concatenate([imu_data['acc'],
                cutted_imu_data['acc'][clean_mask,...]], axis=0)
            imu_data['gyro'] = np.concatenate([imu_data['gyro'],
                cutted_imu_data['gyro'][clean_mask,...]], axis=0)
        self.imu_data = imu_data
        # insert track data
        track_data = self.track_data
        cutted_track_data = record.cutted_track_data
        if track_data['center_pos'] is None:
            track_data['center_pos'] = cutted_track_data['center_pos'][clean_mask,...]
            track_data['center_rot'] = cutted_track_data['center_rot'][clean_mask,...]
            track_data['marker_pos'] = cutted_track_data['marker_pos'][clean_mask,...]    
        else:
            track_data['center_pos'] = np.concatenate([track_data['center_pos'],
                cutted_track_data['center_pos'][clean_mask,...]], axis=0)
            track_data['center_rot'] = np.concatenate([track_data['center_rot'],
                cutted_track_data['center_rot'][clean_mask,...]], axis=0)
            track_data['marker_pos'] = np.concatenate([track_data['marker_pos'],
                cutted_track_data['marker_pos'][clean_mask,...]], axis=0)
        self.track_data = track_data
        self.positive_size += cnt
        # insert labels
        if self.labels is None:
            self.labels = np.zeros(cnt, dtype=np.int64) + label
        else: self.labels = np.concatenate([self.labels, np.zeros(cnt, dtype=np.int64) + label])
        self.size += cnt
        
        
    def insert_record_raise_drop(self, record:Record, raise_label:int, drop_label:int) -> None:
        ''' An adapter for insert raise and drop data only.
        args:
            record: Record, the action record.
            raise_label: int, the label of raise.
            drop_label: int, the label of drop.
        '''
        clean_mask = record.clean_mask
        cnt = np.sum(clean_mask)
        imu_data = self.imu_data
        cutted_imu_data = record.cutted_imu_data
        # insert imu data
        if imu_data['acc'] is None:
            imu_data['acc'] = cutted_imu_data['acc'][clean_mask,...]
            imu_data['gyro'] = cutted_imu_data['gyro'][clean_mask,...]
        else:
            imu_data['acc'] = np.concatenate([imu_data['acc'],
                cutted_imu_data['acc'][clean_mask,...]], axis=0)
            imu_data['gyro'] = np.concatenate([imu_data['gyro'],
                cutted_imu_data['gyro'][clean_mask,...]], axis=0)
        self.imu_data = imu_data
        # insert track data
        track_data = self.track_data
        cutted_track_data = record.cutted_track_data
        if track_data['center_pos'] is None:
            track_data['center_pos'] = cutted_track_data['center_pos'][clean_mask,...]
            track_data['center_rot'] = cutted_track_data['center_rot'][clean_mask,...]
            track_data['marker_pos'] = cutted_track_data['marker_pos'][clean_mask,...]    
        else:
            track_data['center_pos'] = np.concatenate([track_data['center_pos'],
                cutted_track_data['center_pos'][clean_mask,...]], axis=0)
            track_data['center_rot'] = np.concatenate([track_data['center_rot'],
                cutted_track_data['center_rot'][clean_mask,...]], axis=0)
            track_data['marker_pos'] = np.concatenate([track_data['marker_pos'],
                cutted_track_data['marker_pos'][clean_mask,...]], axis=0)
        self.track_data = track_data
        # insert labels
        labels = np.zeros(cutted_imu_data['acc'].shape[0], dtype=np.int64) + raise_label
        labels[1::2] = drop_label
        if self.labels is None:
            self.labels = labels[clean_mask]
        else: self.labels = np.concatenate([self.labels, labels[clean_mask]])
        self.positive_size += cnt
        self.size += cnt
        
        
    def insert_negativa_data(self, negative_data:np.ndarray, label:int):
        ''' Insert negative data in batch.
        args:
            negative_data: np.ndarray[(N, int(FS_PREPROCESS*WINDOW_DURATION), 6), np.float32].
            label: int, the negative label.
        '''
        cnt = negative_data.shape[0]
        imu_data = self.imu_data
        if imu_data['acc'] is None:
            imu_data['acc'] = negative_data[:,:,:3]
            imu_data['gyro'] = negative_data[:,:,3:]
        else:
            imu_data['acc'] = np.concatenate([imu_data['acc'], negative_data[:,:,:3]], axis=0)
            imu_data['gyro'] = np.concatenate([imu_data['gyro'], negative_data[:,:,3:]], axis=0)
        self.imu_data = imu_data
        labels = np.zeros(cnt, dtype=np.int64) + label
        if self.labels is None:
            self.labels = labels
        else: self.labels = np.concatenate([self.labels, labels])
        self.size += cnt     
        
        
    def shuffle(self) -> None:
        ''' Shuffle the reindex mapping.
            NOTE: always call shuffle after inserting records.
        '''
        self.reindex_map = np.arange(self.size, dtype=np.int32)
        np.random.shuffle(self.reindex_map)
        
        
class DataLoader(torch.utils.data.DataLoader):
    
    
    def __init__(self, dataset:Dataset, *args, **kwargs) -> None:
        super(DataLoader, self).__init__(dataset, *args, **kwargs)
    
    
if __name__ == '__main__':
    pass