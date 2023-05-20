import re
import numpy as np
import pandas as pd
from glob import glob
from scipy import stats
from matplotlib import pyplot as plt

import file_utils as fu


def parse_output_file(file_path:str):
    f = open(file_path, 'r')
    pattern = r'F1-score = (\d+\.\d+)%'
    for _ in range(8): f.readline()
    val_score = float(re.search(pattern,f.readline()).group(1))
    for _ in range(10): f.readline()
    test_score = float(re.search(pattern,f.readline()).group(1))
    return val_score, test_score


def analyze_test_f1_score():
    tasks, data_types, val_scores, test_scores, user_list_ids \
        = [], [], [], [], []
    for task in ('Move', 'Rotate'):
        for user_list_id in range(7):    
            # base
            file_path = glob(f'../data/output/*/*{task}IMU_userlist{user_list_id}_N*.txt')[0]
            val_score, test_score = parse_output_file(file_path)
            tasks.append(task); data_types.append('base'); user_list_ids.append(user_list_id)
            val_scores.append(val_score); test_scores.append(test_score)
            # imu
            file_paths = glob(f'../data/output/*/*{task}IMU_userlist{user_list_id}*.txt')
            best_val, best_test = 0, 0
            for file_path in file_paths:
                if '_N_' in file_path: continue
                val_score, test_score = parse_output_file(file_path)
                if val_score > best_val:
                    best_val, best_test = val_score, test_score
            tasks.append(task); data_types.append('imu'); user_list_ids.append(user_list_id)
            val_scores.append(best_val); test_scores.append(best_test)
            # track
            file_paths = glob(f'../data/output/*/*{task}Track_userlist{user_list_id}*.txt')
            best_val, best_test = 0, 0
            for file_path in file_paths:
                if '_N_' in file_path: continue
                val_score, test_score = parse_output_file(file_path)
                if val_score > best_val:
                    best_val, best_test = val_score, test_score
            tasks.append(task); data_types.append('track'); user_list_ids.append(user_list_id)
            val_scores.append(best_val); test_scores.append(best_test)
    df = pd.DataFrame({'task':tasks, 'data_type':data_types, 'user_list_id':user_list_ids,
        'val_score':val_scores, 'test_score':test_scores})
    move_base = df.query(f'task=="Move"&data_type=="base"')['test_score']
    move_imu = df.query(f'task=="Move"&data_type=="imu"')['test_score']
    move_track = df.query(f'task=="Move"&data_type=="track"')['test_score']
    rotate_base = df.query(f'task=="Rotate"&data_type=="base"')['test_score']
    rotate_imu = df.query(f'task=="Rotate"&data_type=="imu"')['test_score']
    rotate_track = df.query(f'task=="Rotate"&data_type=="track"')['test_score']
    
    print(f'Move NoAug: M={np.mean(move_base):.3f}, SD={np.std(move_base,ddof=1):.3f}')
    print(f'Move IMU: M={np.mean(move_imu):.3f}, SD={np.std(move_imu,ddof=1):.3f}')
    print(f'Move Track: M={np.mean(move_track):.3f}, SD={np.std(move_track,ddof=1):.3f}')
    print(f'Rotate NoAug: M={np.mean(rotate_base):.3f}, SD={np.std(rotate_base,ddof=1):.3f}')
    print(f'Rotate IMU: M={np.mean(rotate_imu):.3f}, SD={np.std(rotate_imu,ddof=1):.3f}')
    print(f'Rotate Track: M={np.mean(rotate_track):.3f}, SD={np.std(rotate_track,ddof=1):.3f}')
    
    print(f'MoveIMU vs MoveTrack:')
    print(stats.ttest_rel(move_imu, move_track))
    print(f'RotateIMU vs RotateTrack:')
    print(stats.ttest_rel(rotate_imu, rotate_track))   
    


if __name__ == '__main__':
    fu.check_cwd()
    analyze_test_f1_score()
    