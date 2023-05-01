import os
import json
import shutil
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from typing import List

import config as cf
import file_utils as fu
from data_process.record import Record
from data_process import augment as aug

RECORD_ROOT = '../data/record'
TRACK_ROOT = '../data/track'
TASK_LIST_ID = 'TL3wni1oq3'


def check_track_file_names():
    track_file_names: List[str] = glob(f'{TRACK_ROOT}/*.csv')
    user_dict = dict()
    task_dict = dict()
    subtask_dict = dict()
    for name in track_file_names:
        name = name.split('/')[-1].split('.')[0]
        user, task, subtask, _ = name.split('_')
        if user not in user_dict: user_dict[user] = 1
        else: user_dict[user] += 1
        if task not in task_dict: task_dict[task] = 1
        else: task_dict[task] += 1
        if subtask not in subtask_dict: subtask_dict[subtask] = 1
        else: subtask_dict[subtask] += 1
    for user, cnt in user_dict.items():
        print(user, cnt)
    for task, cnt in task_dict.items():
        print(task, cnt)
    for subtask, cnt in subtask_dict.items():
        print(subtask, cnt)
        
        
def copy_track_files():
    # get task list
    task_list = fu.load_task_list_with_users(TASK_LIST_ID)
    task_name_map = {'M10': 'Move10cm', 'M20': 'Move20cm', 'M30': 'Move30cm', 'M40': 'Move40cm', 'M50': 'Move50cm'}
    subtask_name_map = {'F': 'Fast', 'M': 'Medium', 'S': 'Slow'}
    # iterate track file paths
    track_file_paths = glob(f'{TRACK_ROOT}/*.csv')
    for path in track_file_paths:
        name = path.split('/')[-1].split('.')[0]
        user_name, task_name, subtask_name, _ = name.split('_')
        task_name = task_name_map[task_name]
        subtask_name = subtask_name_map[subtask_name]
        for task in task_list['tasks']:
            if task['name'] == task_name: break
        assert task['name'] == task_name
        for subtask in task['subtasks']:
            if subtask['name'] == subtask_name: break
        assert subtask['name'] == subtask_name
        record_dict = subtask['record_dict']
        if user_name in record_dict:
            record_id = record_dict[user_name]
            record_path = fu.get_record_path(TASK_LIST_ID, task['id'], subtask['id'], record_id)
            shutil.copy(path, record_path)
            print(f'Success: {path}')
        else: print(f'Dst path not exist: {path}')
        
        
def check_record_files():
    task_ids = ['TK6qxanwm7', 'TKcgnu2c20', 'TKp59cxeeh', 'TKqqx4moyj', 'TKsql1b33x']
    for task_id in task_ids:
        record_paths = glob(f'{RECORD_ROOT}/{TASK_LIST_ID}/{task_id}/ST*/RD*')
        for record_path in record_paths:
            for _, _, file_paths in os.walk(record_path):
                track_cnt, imu_cnt = 0, 0
                for file_path in file_paths:
                    if file_path.startswith('Motion') and file_path.endswith('.bin'):
                        imu_cnt += 1
                    if file_path.endswith('.csv'):
                        track_cnt += 1
                if track_cnt != 1 or imu_cnt != 1:
                    print(f'{record_path}: {track_cnt}, {imu_cnt}')
    

def check_record_signal():
    task_id = 'TK5rsia9fw'
    subtask_id = 'STl3yde7qb'
    record_id = 'RDqq4s24ot'
    record_path = fu.get_record_path(TASK_LIST_ID, task_id, subtask_id, record_id)
    record = Record(record_path, n_sample=20)
    imu_data = record.imu_data
    track_data = record.track_data
    gyro = imu_data['gyro']
    center_pos = track_data['center_pos']
    center_rot = track_data['center_rot']
    marker_pos = track_data['marker_pos']
    timestamps = track_data['timestamps']
    # NOTE: should resample track data if FS_TRACK != FS_PREPROCESS
    axes = aug.calc_local_axes(marker_pos)
    generated_gyro = aug.track_to_gyro(axes, cf.FS_PREPROCESS)
    for i in range(3):
        plt.plot(gyro[:,i])
    for i in range(3):
        plt.plot(generated_gyro[:,i])
    plt.show()


if __name__ == '__main__':
    fu.check_cwd()
    copy_track_files()
    check_record_files()
    # check_record_signal()
