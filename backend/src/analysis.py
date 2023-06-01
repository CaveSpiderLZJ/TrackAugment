import re
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from scipy import stats
from matplotlib import pyplot as plt

import file_utils as fu


plt.rcParams["font.sans-serif"]=["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"]=False


def parse_output_file(file_path:str) -> dict:
    file_name = file_path.split('/')[-1][:-4]
    f = open(file_path, 'r')
    pattern = r'F1-score = (\d+\.\d+)%'
    for _ in range(8): f.readline()
    val_score = float(re.search(pattern,f.readline()).group(1))
    for _ in range(4): f.readline()
    matrix = np.empty((5,5), dtype=np.float64)
    for i in range(5):
        line = f.readline()
        matrix[i, :] = list(map(float, line.split('|')[-1].split()))
    line = f.readline()
    fpr = float(re.search(r'FPR = (\d+\.\d+)%', line).group(1))
    acc = float(re.search(r'Accuracy = (\d+\.\d+)%', line).group(1))
    line = f.readline()
    recall = float(re.search(r'Recall = (\d+\.\d+)%', line).group(1))
    precision = float(re.search(r'Precision = (\d+\.\d+)%', line).group(1))
    test_score = float(re.search(pattern, line).group(1))
    return {'file_name': file_name, 'val_score': val_score,
        'test_score': test_score, 'matrix': matrix,
        'fpr': fpr, 'acc': acc, 'recall': recall, 'precision': precision}


def analyze_test_f1_score():
    tasks, data_types, val_scores, test_scores, user_list_ids, accs, fprs, \
        recalls, precisions = [], [], [], [], [], [], [], [], []
    for task in ('Move', 'Rotate'):
        for user_list_id in range(10):    
            # base
            file_path = glob(f'../data/output/*/*{task}IMU_userlist{user_list_id}_N*.txt')[0]
            info = parse_output_file(file_path)
            val_score, test_score = info['val_score'], info['test_score']
            tasks.append(task); data_types.append('base'); user_list_ids.append(user_list_id)
            val_scores.append(val_score); test_scores.append(test_score)
            accs.append(info['acc']); fprs.append(info['fpr'])
            recalls.append(info['recall']); precisions.append(info['precision'])
            # imu and track
            for key1, key2 in (('IMU', 'imu'), ('Track', 'track')):
                
                file_paths = glob(f'../data/output/*/*{task}{key1}_userlist{user_list_id}*.txt')
                best_val, best_test, best_acc, best_fpr, best_recall, best_precision \
                    = 0, 0, 0, 0, 0, 0
                for file_path in file_paths:
                    if '_N_' in file_path: continue
                    info = parse_output_file(file_path)
                    val_score, test_score = info['val_score'], info['test_score']
                    if val_score > best_val:
                        best_val, best_test = val_score, test_score
                        best_acc = info['acc']; best_fpr = info['fpr']
                        best_recall = info['recall']; best_precision = info['precision']
                tasks.append(task); data_types.append(key2); user_list_ids.append(user_list_id)
                val_scores.append(best_val); test_scores.append(best_test)
                accs.append(best_acc); fprs.append(best_fpr)
                recalls.append(best_recall); precisions.append(best_precision)
    df = pd.DataFrame({'task':tasks, 'data_type':data_types, 'user_list_id':user_list_ids,
        'val_score':val_scores, 'test_score':test_scores, 'acc': accs,
        'fpr': fprs, 'recall': recalls, 'precision': precisions})
    move_base = df.query(f'task=="Move"&data_type=="base"')['test_score']
    move_imu = df.query(f'task=="Move"&data_type=="imu"')['test_score']
    move_track = df.query(f'task=="Move"&data_type=="track"')['test_score']
    rotate_base = df.query(f'task=="Rotate"&data_type=="base"')['test_score']
    rotate_imu = df.query(f'task=="Rotate"&data_type=="imu"')['test_score']
    rotate_track = df.query(f'task=="Rotate"&data_type=="track"')['test_score']
    
    for key in ('acc', 'fpr', 'recall', 'precision', 'test_score'):
        print(f'\n{key}:')
        move_base = df.query(f'task=="Move"&data_type=="base"')[key]
        move_imu = df.query(f'task=="Move"&data_type=="imu"')[key]
        move_track = df.query(f'task=="Move"&data_type=="track"')[key]
        rotate_base = df.query(f'task=="Rotate"&data_type=="base"')[key]
        rotate_imu = df.query(f'task=="Rotate"&data_type=="imu"')[key]
        rotate_track = df.query(f'task=="Rotate"&data_type=="track"')[key]
        print(f'Move NoAug: M={np.mean(move_base):.2f}, SD={np.std(move_base,ddof=1):.2f}')
        print(f'Move IMU: M={np.mean(move_imu):.2f}, SD={np.std(move_imu,ddof=1):.2f}')
        print(f'Move Track: M={np.mean(move_track):.2f}, SD={np.std(move_track,ddof=1):.2f}')
        print(f'Rotate NoAug: M={np.mean(rotate_base):.2f}, SD={np.std(rotate_base,ddof=1):.2f}')
        print(f'Rotate IMU: M={np.mean(rotate_imu):.2f}, SD={np.std(rotate_imu,ddof=1):.2f}')
        print(f'Rotate Track: M={np.mean(rotate_track):.2f}, SD={np.std(rotate_track,ddof=1):.2f}')
        
    key = 'test_score'
    move_base = df.query(f'task=="Move"&data_type=="base"')[key]
    move_imu = df.query(f'task=="Move"&data_type=="imu"')[key]
    move_track = df.query(f'task=="Move"&data_type=="track"')[key]
    rotate_base = df.query(f'task=="Rotate"&data_type=="base"')[key]
    rotate_imu = df.query(f'task=="Rotate"&data_type=="imu"')[key]
    rotate_track = df.query(f'task=="Rotate"&data_type=="track"')[key]
    print(f'\n### ANOVA on F1-score:')
    print(f'MoveIMU vs MoveTrack:')
    print(stats.ttest_rel(move_imu, move_track))
    print(stats.ttest_rel(move_track, move_imu))
    print(f'RotateIMU vs RotateTrack:')
    print(stats.ttest_rel(rotate_imu, rotate_track))
    print(stats.ttest_rel(rotate_track, rotate_imu))
    
    
    
def analyze_aug_parameters():
    file_paths = glob(f'../data/output/*/*RotateTrackComb_model5*.txt')
    items = []
    for file_path in file_paths:
        info = parse_output_file(file_path)
        items.append((info['file_name'], info['val_score']))
    items = sorted(items, key=lambda item:item[1], reverse=True)
    for item in items:
        print(f'{item[0]}: {item[1]}')
        
        
def visualize_matrix():
    ''' Plot six matrix across users: Move(NoAug, IMU best, track best),
        Rotate(NoAug, IMU best, track best).
    '''
    matrix_dict = { 'MoveNoAug': [], 'MoveIMU': [], 'MoveTrack': [],
        'RotateNoAug': [], 'RotateIMU': [], 'RotateTrack': [] }
    for user_list_id in range(10):
        for task in ('Move', 'Rotate'):
            # NoAug
            file_path = glob(f'../data/output/*/*{task}IMU_userlist{user_list_id}_N*.txt')[0]
            info = parse_output_file(file_path)
            matrix_dict[f'{task}NoAug'].append(info['matrix'])
            # IMU and Track
            for method in ('IMU', 'Track'):
                file_paths = glob(f'../data/output/*/*{task}{method}_userlist{user_list_id}*.txt')
                best_val, best_matrix = 0, None
                for file_path in file_paths:
                    if '_N_' in file_path: continue
                    info = parse_output_file(file_path)
                    val_score = info['val_score']
                    if val_score > best_val:
                        best_val = val_score
                        best_matrix = info['matrix']
                matrix_dict[f'{task}{method}'].append(best_matrix)
    for key, matrix_list in matrix_dict.items():
        matrix_dict[key] = np.mean(matrix_list, axis=0)
    
    # plot the confusion matrix
    matrix_list = [matrix_dict[key] for key in
        ('MoveNoAug', 'MoveIMU', 'MoveTrack', 'RotateNoAug', 'RotateIMU', 'RotateTrack')]
    matrix_titles = ('移动：无数据增强', '移动：增强 IMU 数据', '移动：增强运动轨迹数据',
        '旋转：无数据增强', '旋转：增强 IMU 数据', '旋转：增强运动轨迹数据', )
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for i, ax in enumerate(axes.flat):
        if i < 3: label_names = ('负例', '10 厘米', '20 厘米', '30 厘米', '40 厘米')
        else: label_names = ('负例', '45 度', '90 度', '135 度', '180 度')
        sns.heatmap(matrix_list[i], annot=True, cmap='Blues', fmt='.2f',
            vmin=0, vmax=100, ax=ax, cbar=False,)
        if i >= 3: ax.set_xlabel('模型预测标签', fontsize=14)
        if i == 0 or i == 3: ax.set_ylabel('真实标签', fontsize=14)
        ax.set_title(f'{matrix_titles[i]}', fontsize=14)
        ax.set_xticklabels(label_names, rotation=0, ha='center', fontsize=12)
        ax.set_yticklabels(label_names, rotation=90, fontsize=12)
    cbar_ax = fig.add_axes([0.92, 0.11, 0.01, 0.3])
    cb = fig.colorbar(ax.collections[0], cax=cbar_ax)
    cb.set_label('预测比例 (%)', fontsize=14)
    plt.show()
    

if __name__ == '__main__':
    fu.check_cwd()
    analyze_test_f1_score()
    