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
TASK_LIST_ID = 'TLnmdi15b8'


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
    root_list = fu.load_root_list_info()
    for task_list in root_list['tasklists']:
        if task_list['id'] == TASK_LIST_ID:
            break
    # iterate track files
    track_file_paths: List[str] = glob(f'{TRACK_ROOT}/*.csv')
    record_set = set()
    for path in track_file_paths:
        name = path.split('/')[-1].split('.')[0]
        user_name, task_name, subtask_name, _ = name.split('_')
        task_id, subtask_id, record_id = None, None, None
        for task in task_list['tasks']:
            if task['name'] == task_name:
                task_id = task['id']
                break
        assert task_id != None
        for subtask in task['subtasks']:
            if subtask['name'] == subtask_name:
                subtask_id = subtask['id']
                break
        assert subtask_id != None
        record_list = json.load(open(f'{RECORD_ROOT}/{TASK_LIST_ID}/{task_id}/{subtask_id}/recordlist.json', 'r'))
        for record in record_list:
            if record['user_name'] == user_name:
                record_id = record['record_id']
                record_set.add(record_id)
                break
        if record_id == None:
            print(f'Record id not found: {path}, {task_id}, {subtask_id}')
            continue
        dst = f'{RECORD_ROOT}/{TASK_LIST_ID}/{task_id}/{subtask_id}/{record_id}'
        if not os.path.exists(dst):
            print(f'Dst path not exist: {path}')
            continue
        shutil.copy(path, dst)
        print(f'Success: {path}')
        
        
def check_record_files():
    task_ids = ['TK7t3ql6jb', 'TK9fe2fbln', 'TK5rsia9fw', 'TKtvkgst8r', 'TKie8k1h6r']
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
    record = Record(record_path)
    imu_data = record.imu_data
    track_data = record.track_data
    gyro = imu_data['gyro']
    gyro = np.column_stack([gyro[axis] for axis in ('x','y','z')])
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
