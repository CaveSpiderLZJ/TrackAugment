import time
import json
import numpy as np
from scipy import interpolate as interp
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
        self.raw_track_data = {'center_pos': [], 'center_rot': [], 'marker_pos': []}    #=\
        self.raw_imu_data = {'acc': [], 'gyro': []}                                     #=| positive data
        self.raw_positive_labels = []                                                   #=/
        self.raw_negative_data = {'acc': [], 'gyro': []}                                #=\ negative data
        self.raw_negative_labels = []                                                   #=/
        self.record_paths = []
        self.augmented_imu_data = None  # (size, window_length, n_channels)
        self.augmented_labels = None    # (size,)
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
        assert self.reindex_map is not None
        idx = self.reindex_map[idx]
        w_cut = int(cf.CUT_DURATION * cf.FS_PREPROCESS)
        w_train = int(cf.TRAIN_DURATION * cf.FS_PREPROCESS)
        if cf.RANDOM_SAMPLE_EN: start = np.random.randint(0, w_cut - w_train)
        else: start = (w_cut - w_train) // 2
        sample = self.augmented_imu_data[idx, start:start+w_train, :]
        sample = aug.down_sample_by_step(sample, axis=0, step=(cf.FS_PREPROCESS//cf.FS_TRAIN))
        return {'data': sample, 'label': self.augmented_labels[idx]}
    
    
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
        ''' Insert a single positive record with all samples have the same group id.
        args:
            record: Record, the action record.
            label: int, the label of the record.
        '''
        clean_mask = record.clean_mask
        cnt = np.sum(clean_mask)
        # insert imu data
        cutted_imu_data = record.cutted_imu_data
        self.raw_imu_data['acc'].append(cutted_imu_data['acc'][clean_mask,:,:])
        self.raw_imu_data['gyro'].append(cutted_imu_data['gyro'][clean_mask,:,:])
        # insert track data
        cutted_track_data = record.cutted_track_data
        self.raw_track_data['center_pos'].append(cutted_track_data['center_pos'][clean_mask,:,:])
        self.raw_track_data['center_rot'].append(cutted_track_data['center_rot'][clean_mask,:,:])
        self.raw_track_data['marker_pos'].append(cutted_track_data['marker_pos'][clean_mask,:,:])
        # insert labels
        self.raw_positive_labels.append(np.zeros(cnt, dtype=np.int64) + label)
        # tmp
        self.record_paths.append(record.record_path)
        
        
    def insert_record_raise_drop(self, record:Record, raise_label:int, drop_label:int) -> None:
        ''' An adapter for insert raise and drop data only.
        args:
            record: Record, the action record.
            raise_label: int, the label of raise.
            drop_label: int, the label of drop.
        '''
        clean_mask = record.clean_mask
        cnt = np.sum(clean_mask)
        # insert imu data
        cutted_imu_data = record.cutted_imu_data
        N = cutted_imu_data['acc'].shape[0]
        raise_indices = list(range(0,N,2))
        drop_indices = list(range(1,N,2))
        raise_mask = clean_mask[raise_indices]
        drop_mask = clean_mask[drop_indices]
        self.raw_imu_data['acc'].append(cutted_imu_data['acc'][raise_indices,:,:][raise_mask,:,:])
        self.raw_imu_data['gyro'].append(cutted_imu_data['gyro'][raise_indices,:,:][raise_mask,:,:])
        self.raw_imu_data['acc'].append(cutted_imu_data['acc'][drop_indices,:,:][drop_mask,:,:])
        self.raw_imu_data['gyro'].append(cutted_imu_data['gyro'][drop_indices,:,:][drop_mask,:,:])
        # insert track data
        cutted_track_data = record.cutted_track_data
        self.raw_track_data['center_pos'].append(cutted_track_data['center_pos'][raise_indices,:,:][raise_mask,:,:])
        self.raw_track_data['center_rot'].append(cutted_track_data['center_rot'][raise_indices,:,:][raise_mask,:,:])
        self.raw_track_data['marker_pos'].append(cutted_track_data['marker_pos'][raise_indices,:,:,:][raise_mask,:,:,:])
        self.raw_track_data['center_pos'].append(cutted_track_data['center_pos'][drop_indices,:,:][drop_mask,:,:])
        self.raw_track_data['center_rot'].append(cutted_track_data['center_rot'][drop_indices,:,:][drop_mask,:,:])
        self.raw_track_data['marker_pos'].append(cutted_track_data['marker_pos'][drop_indices,:,:,:][drop_mask,:,:,:])
        # insert labels
        self.raw_positive_labels.append(np.zeros(np.sum(raise_mask), dtype=np.int64) + raise_label)
        self.raw_positive_labels.append(np.zeros(np.sum(drop_mask), dtype=np.int64) + drop_label)
        # tmp
        self.record_paths.append(record.record_path + '(Raise)')
        self.record_paths.append(record.record_path + '(Drop)')
        
        
    def insert_negativa_data(self, negative_data:np.ndarray, label:int):
        ''' Insert negative data in batch.
        args:
            negative_data: np.ndarray[(N, int(FS_PREPROCESS*WINDOW_DURATION), 6), np.float32].
            label: int, the negative label.
        '''
        cnt = negative_data.shape[0]
        self.raw_negative_data['acc'].append(negative_data[:,:,:3])
        self.raw_negative_data['gyro'].append(negative_data[:,:,3:])
        self.raw_negative_labels.append(np.zeros(cnt, dtype=np.int64) + label)
        
        
    def augment(self, method:str=None) -> None:
        ''' Use raw data to generate augmented imu data.
        args:
            method: str, in {None, 'classic', 'dtw'}
        '''
        augmented_imu_data = []
        augmented_labels = []
        if method == 'classic':     # classic augmentation methods on imu data
            for acc, gyro, labels in zip(self.raw_imu_data['acc'],
                self.raw_imu_data['gyro'], self.raw_positive_labels):
                N = labels.shape[0]
                imu_list = []
                for i in range(N):
                    imu = aug.classic_augment(np.concatenate([acc[i,:,:], gyro[i,:,:]], axis=1), axis=0)
                    imu_list.append(imu[None,:,:])
                augmented_imu_data.append(np.concatenate(imu_list, axis=0))
                augmented_labels.append(labels)
        elif method == 'classic_on_track':
            for center_pos_batch, marker_pos_batch, labels in zip(self.raw_track_data['center_pos'],
                self.raw_track_data['marker_pos'], self.raw_positive_labels):
                N = labels.shape[0]
                imu_list = []
                for i in range(N):
                    imu = aug.classic_augment_on_track(
                        center_pos_batch[i,:,:], marker_pos_batch[i,:,:,:])
                    imu_list.append(imu[None,:,:])
                augmented_imu_data.append(np.concatenate(imu_list, axis=0))
                augmented_labels.append(labels)
        elif method == 'dtw':       # dtw based augmentation on track data
            for center_pos, marker_pos, labels in zip(self.raw_track_data['center_pos'],
                self.raw_track_data['marker_pos'], self.raw_positive_labels):
                N, T = center_pos.shape[0], center_pos.shape[1]
                dcenter_pos = np.diff(center_pos, axis=1, append=center_pos[:,-1:,:])
                imu_list = []
                weight = 0.5
                rand_indices = np.random.randint(0, N-1, size=N)
                rand_indices[rand_indices>=np.arange(N)] += 1
                for i, idx in enumerate(rand_indices):
                    warping_path = aug.dtw_match(dcenter_pos[i,:,:],
                        dcenter_pos[idx,:,:], axis=0, window=int(1.0*cf.FS_PREPROCESS))
                    augmented_center_pos = aug.dtw_augment(center_pos[i,:,:],
                        center_pos[idx,:,:], warping_path, axis=0, weight=weight)
                    augmented_marker_pos = aug.dtw_augment(marker_pos[i,:,:,:],
                        marker_pos[idx,:,:,:], warping_path, axis=1, weight=weight)
                    axes = aug.calc_local_axes(augmented_marker_pos)
                    acc = aug.track_to_acc(augmented_center_pos, axes, fs=cf.FS_PREPROCESS)
                    gyro = aug.track_to_gyro(axes, fs=cf.FS_PREPROCESS)
                    imu_list.append(np.concatenate([acc, gyro], axis=1)[None,:,:])
                augmented_imu_data.append(np.concatenate(imu_list, axis=0))
                augmented_labels.append(labels)
        else:                       # no augmentation, copy imu data directly
            assert method is None
            for acc, gyro, labels in zip(self.raw_imu_data['acc'],
                self.raw_imu_data['gyro'], self.raw_positive_labels):
                augmented_imu_data.append(np.concatenate([acc, gyro], axis=2))
                augmented_labels.append(labels)
        for acc, gyro, labels in zip(self.raw_negative_data['acc'],
            self.raw_negative_data['gyro'], self.raw_negative_labels):
            augmented_imu_data.append(np.concatenate([acc, gyro], axis=2))
            augmented_labels.append(labels)
        self.augmented_imu_data = np.concatenate(augmented_imu_data, axis=0)
        self.augmented_labels = np.concatenate(augmented_labels, axis=0)
        self.size = self.augmented_labels.shape[0]
        self.reindex_map = np.arange(self.size, dtype=np.int32)
        np.random.shuffle(self.reindex_map)
        
        
class DataLoader(torch.utils.data.DataLoader):
    
    
    def __init__(self, dataset:Dataset, *args, **kwargs) -> None:
        super(DataLoader, self).__init__(dataset, *args, **kwargs)
        
        
    def augment(self, *args, **kwargs) -> None:
        self.dataset.augment(*args, **kwargs)
    
    
if __name__ == '__main__':
    pass