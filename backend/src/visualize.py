import os
import cv2
import time
import tqdm
import json
import pickle
import numpy as np
import pandas as pd
from torch import nn
from glob import glob
from scipy import stats
from scipy import interpolate as interp
from scipy.spatial.transform import Rotation
from mpl_toolkits import mplot3d
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from matplotlib.pyplot import MultipleLocator
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.spatial.transform import Rotation

import config as cf
import file_utils as fu
from data_process.record import Record
from data_process import augment as aug
from data_process.cutter import PeakCutter
from data_process.filter import Butterworth
from data_process.dataset import Dataset, DataLoader
from data_process.dtw import dtw_match
from train.model import *


plt.rcParams["font.sans-serif"]=["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"]=False


COLORS = {
    'r': "#F14D4D",
    'g': '#46B454',
    'b': '#56B7EC',
    'o': '#EE8B29',
    'p': '#8051C9',
    'black': '#000000',
}


def output_track_video(record:Record, save_path:str, start_frame:int, end_frame:int) -> None:
    ''' Output the tracking video from tracking data.
    args:
        record: Record, the record data.
        save_path: str, the video save path.
        start_frame: int, the start frame index.
        end_frame: int, the end frame index.
    '''
    track_data = record.track_data
    center_pos = track_data['center_pos']
    marker_pos = track_data['marker_pos']
    H, W, PAD = 480, 640, 20
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(save_path, fourcc, 50, (W, H))
    fig = plt.figure()
    canvas = FigureCanvasAgg(fig)
    # x, y, z axis limits
    x_min, x_max = np.min(marker_pos[:,:,0])-PAD, np.max(marker_pos[:,:,0])+PAD
    y_min, y_max = np.min(marker_pos[:,:,1])-PAD, np.max(marker_pos[:,:,1])+PAD
    z_min, z_max = np.min(marker_pos[:,:,2])-PAD, np.max(marker_pos[:,:,2])+PAD
    # rotation axes
    axes = 100 * aug.calc_local_axes(marker_pos)
    # rotate x, y, z to -x, z, y
    rot_matrix = np.array([[-1,0,0],[0,0,1],[0,1,0]], dtype=np.float32)
    center_pos = np.matmul(rot_matrix, center_pos.T).T
    marker_pos = np.matmul(rot_matrix[None,:,:], marker_pos.transpose(0,2,1)).transpose(0,2,1)
    axes = np.matmul(rot_matrix[None,:,:], axes.transpose(0,2,1)).transpose(0,2,1)
    # x, y, z axis limits
    x_min, x_max = np.min(marker_pos[:,:,0])-PAD, np.max(marker_pos[:,:,0])+PAD
    y_min, y_max = np.min(marker_pos[:,:,1])-PAD, np.max(marker_pos[:,:,1])+PAD
    z_min, z_max = np.min(marker_pos[:,:,2])-PAD, np.max(marker_pos[:,:,2])+PAD
    for i in tqdm.trange(start_frame, end_frame, 4):
        x, y, z = center_pos[i,:]
        ax = plt.axes(projection='3d')
        ax.scatter(x, y, z, color='black', s=40)
        ax.scatter(marker_pos[:,i,0], marker_pos[:,i,1], marker_pos[:,i,2], color='orange', s=20)
        ax.quiver(x, y, z, axes[0,i,0], axes[0,i,1], axes[0,i,2], color='red')
        ax.quiver(x, y, z, axes[1,i,0], axes[1,i,1], axes[1,i,2], color='green')
        ax.quiver(x, y, z, axes[2,i,0], axes[2,i,1], axes[2,i,2], color='blue')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_major_locator(MultipleLocator(100))
        ax.zaxis.set_major_locator(MultipleLocator(100))
        ax.set_box_aspect((x_max-x_min,y_max-y_min,z_max-z_min))
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.set_title(f'Smartphone Motion Visualization', fontsize=16)
        buffer, (w, h) = canvas.print_to_buffer()
        img = np.frombuffer(buffer, np.uint8).reshape((h, w, 4))
        img = cv2.resize(img, (W, H))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        video_writer.write(img)
        fig.clear()
    video_writer.release()


def visualize_track_to_imu(record:Record) -> None:
    idx = 10
    start, end = 100, 500
    # prepare imu data
    imu_data = record.cutted_imu_data
    acc = imu_data['acc'][idx,start:end,:]
    gyro = imu_data['gyro'][idx,start:end,:]
    # prepare track data
    track_data = record.cutted_track_data
    center_pos = track_data['center_pos'][idx,start:end,:]
    marker_pos = track_data['marker_pos'][idx,:,start:end,:]
    axes = aug.calc_local_axes(marker_pos)
    track_acc = aug.track_to_acc(center_pos, axes, cf.FS_PREPROCESS)
    track_gyro = aug.track_to_gyro(axes, cf.FS_PREPROCESS)
    # plot acc data
    colors = ('r', 'g', 'b')
    t = np.linspace(0, 2, num=(end-start), endpoint=False)
    plt.figure(figsize=(8,8))
    plt.subplot(2, 1, 1)
    for i in range(3):
        plt.plot(t, acc[:,i], color=COLORS[colors[i]], linestyle='-.')
    for i in range(3):
        plt.plot(t, track_acc[:,i], color=COLORS[colors[i]])
    plt.title('运动轨迹转化为 IMU 数据', fontsize=20)
    plt.ylabel('加速度 (m/s^2)', fontsize=14)
    plt.grid('both', linestyle='--')
    plt.legend(['X', 'Y', 'Z', 'X\'', 'Y\'', 'Z\''], loc='lower right')
    # plot gyro data
    plt.subplot(2, 1, 2)
    for i in range(3):
        plt.plot(t, gyro[:,i], color=COLORS[colors[i]], linestyle='-.')
    for i in range(3):
        plt.plot(t, track_gyro[:,i], color=COLORS[colors[i]])
    plt.xlabel('时间 (s)', fontsize=14)
    plt.ylabel('角速度 (rad/s)', fontsize=14)
    plt.grid('both', linestyle='--')
    plt.legend(['X', 'Y', 'Z', 'X\'', 'Y\'', 'Z\''], loc='lower right')
    # plt.figure(figsize=(10,10))
    plt.show()
    return


def visualize_imu_to_track(record:Record):
    idx = 10
    start, end = 100, 500
    # prepare imu data
    imu_data = record.cutted_imu_data
    acc = imu_data['acc'][idx,start:end,:]
    gyro = imu_data['gyro'][idx,start:end,:]
    # prepare track data
    track_data = record.cutted_track_data
    center_pos = track_data['center_pos'][idx,start:end,:]
    marker_pos = track_data['marker_pos'][idx,:,start:end,:]
    # border condition    
    axes = aug.calc_local_axes(marker_pos)
    bound_pos = np.concatenate([center_pos[:1,:],center_pos[-1:,:]], axis=0)
    bound_velocity = cf.FS_PREPROCESS * np.concatenate([center_pos[1:2,:]-center_pos[0:1,:],
        center_pos[-1:,:]-center_pos[-2:-1,:]], axis=0)
    bound_axes = np.concatenate([axes[:,:2,:],axes[:,-2:,:]], axis=1)
    generated_pos, generated_axes = aug.imu_to_track(acc, gyro, bound_pos,
        bound_velocity, bound_axes, cf.FS_PREPROCESS)
    # rotate x, y, z, to -x, z, y
    rot_matrix = np.array([[-1,0,0],[0,0,1],[0,1,0]], dtype=np.float32)
    center_pos = np.matmul(rot_matrix, center_pos.T).T
    generated_pos = np.matmul(rot_matrix, generated_pos.T).T
    # plot 3d track
    plt.figure(figsize=(8,8))
    ax = plt.subplot(2, 1, 1, projection='3d')
    ax.plot(center_pos[:,0], center_pos[:,1], center_pos[:,2], color=COLORS['o'])
    ax.plot(generated_pos[:,0], generated_pos[:,1], generated_pos[:,2], color=COLORS['p'])
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.zaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_box_aspect((3, 1, 1))
    ax.set_xlim(-0.03, 0.27)
    ax.set_ylim(0.075, 0.125)
    ax.set_zlim(-0.025, 0.025)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(['原始运动轨迹', 'IMU 转化轨迹'], bbox_to_anchor=(1.55, 0.1))
    plt.title(f'IMU 数据转化为运动轨迹', fontsize=20)
    # plot signals
    plt.subplot(2, 1, 2)
    colors = ('r', 'g', 'b')
    t = np.linspace(0, 2, num=(end-start), endpoint=False)
    for i in range(3):
        plt.plot(t, center_pos[:,i], color=COLORS[colors[i]], linestyle='-.')
    for i in range(3):
        plt.plot(t, generated_pos[:,i], color=COLORS[colors[i]])
    plt.grid(axis='both', linestyle='--')
    plt.legend(['X', 'Y', 'Z', 'X\'', 'Y\'', 'Z\''], loc='lower right')
    plt.ylabel('空间位移 (m)', fontsize=14)
    plt.xlabel('时间 (s)', fontsize=14)
    plt.show()  
    
    
def visualize_augmented_imu(record:Record):
    idx = 10
    start, end = 100, 500
    # prepare imu data
    imu_data = record.cutted_imu_data
    acc = imu_data['acc'][idx,start:end,:]
    gyro = imu_data['gyro'][idx,start:end,:]
    params = 0.9
    augmented_acc = aug.zoom(acc, axis=0, params=params)
    augmented_gyro = aug.zoom(gyro, axis=0, params=params)
    # params = 1.1
    # augmented_acc = aug.scale(augmented_acc, params=params)
    # augmented_gyro = aug.scale(augmented_gyro, params=params)
    # prepare track data
    track_data = record.cutted_track_data
    center_pos = track_data['center_pos'][idx,start:end,:]
    marker_pos = track_data['marker_pos'][idx,:,start:end,:]
    axes = aug.calc_local_axes(marker_pos)
    bound_pos = np.concatenate([center_pos[:1,:],center_pos[-1:,:]], axis=0)
    bound_velocity = cf.FS_PREPROCESS * np.concatenate([center_pos[1:2,:]-center_pos[0:1,:],
        center_pos[-1:,:]-center_pos[-2:-1,:]], axis=0)
    bound_axes = np.concatenate([axes[:,:2,:],axes[:,-2:,:]], axis=1)
    augmented_pos, _ = aug.imu_to_track(augmented_acc, augmented_gyro, bound_pos,
        bound_velocity, bound_axes, cf.FS_PREPROCESS)
    # rotate x, y, z, to -x, z, y
    rot_matrix = np.array([[-1,0,0],[0,0,1],[0,1,0]], dtype=np.float32)
    center_pos = np.matmul(rot_matrix, center_pos.T).T
    augmented_pos = np.matmul(rot_matrix, augmented_pos.T).T
    # plot
    t = np.linspace(0, 2, num=(end-start), endpoint=False)
    colors = ('r', 'g', 'b')
    plt.figure(figsize=(16,8))
    plt.subplot(2, 2, 1)
    for i in range(3):
        plt.plot(t, acc[:,i], color=COLORS[colors[i]], linestyle='-.')
    for i in range(3):
        plt.plot(t, augmented_acc[:,i], color=COLORS[colors[i]])
    plt.ylabel('加速度 (m/s^2)', fontsize=14)
    plt.legend(['X', 'Y', 'Z', 'X\'', 'Y\'', 'Z\''], loc='lower right')
    plt.grid('both', linestyle='--')
    plt.subplot(2, 2, 3)
    for i in range(3):
        plt.plot(t, gyro[:,i], color=COLORS[colors[i]], linestyle='-.')
    for i in range(3):
        plt.plot(t, augmented_gyro[:,i], color=COLORS[colors[i]])
    plt.ylabel('角速度 (rad/s)', fontsize=14)
    plt.xlabel('时间 (s)', fontsize=14)
    plt.legend(['X', 'Y', 'Z', 'X\'', 'Y\'', 'Z\''], loc='lower right')
    plt.grid('both', linestyle='--')
    ax = plt.subplot(2, 2, 2, projection='3d')
    ax.plot(center_pos[:,0], center_pos[:,1], center_pos[:,2], color=COLORS['o'])
    ax.plot(augmented_pos[:,0], augmented_pos[:,1], augmented_pos[:,2], color=COLORS['p'])
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.zaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_box_aspect((3.5, 1, 1))
    ax.set_xlim(-0.03, 0.32)
    ax.set_ylim(0.075, 0.125)
    ax.set_zlim(-0.025, 0.025)
    ax.legend(['原始运动轨迹', '增强后转化的轨迹'], bbox_to_anchor=(1.50, 0.1))     
    plt.subplot(2, 2, 4)
    for i in range(3):
        plt.plot(t, center_pos[:,i], color=COLORS[colors[i]], linestyle='-.')
    for i in range(3):
        plt.plot(t, augmented_pos[:,i], color=COLORS[colors[i]])
    plt.ylabel('空间位移 (m)', fontsize=14)
    plt.xlabel('时间 (s)', fontsize=14)
    plt.legend(['X', 'Y', 'Z', 'X\'', 'Y\'', 'Z\''], loc='lower right')
    plt.grid('both', linestyle='--')
    plt.show()
    
    
def visualize_augmented_track(record:Record):
    idx = 10
    start, end = 100, 500
    # prepare imu data
    imu_data = record.cutted_imu_data
    acc = imu_data['acc'][idx,start:end,:]
    gyro = imu_data['gyro'][idx,start:end,:]
    # prepare track data
    track_data = record.cutted_track_data
    center_pos = track_data['center_pos'][idx,start:end,:]
    marker_pos = track_data['marker_pos'][idx,:,start:end,:]
    axes = aug.calc_local_axes(marker_pos)
    params = 0.9
    augmented_pos = aug.zoom(center_pos, axis=0, params=params)
    augmented_axes = aug.zoom(axes, axis=1, params=params)
    augmented_acc = aug.track_to_acc(augmented_pos, augmented_axes, cf.FS_PREPROCESS)
    augmented_gyro = aug.track_to_gyro(augmented_axes, cf.FS_PREPROCESS)
    # rotate x, y, z, to -x, z, y
    rot_matrix = np.array([[-1,0,0],[0,0,1],[0,1,0]], dtype=np.float32)
    center_pos = np.matmul(rot_matrix, center_pos.T).T
    augmented_pos = np.matmul(rot_matrix, augmented_pos.T).T
    # plot
    t = np.linspace(0, 2, num=(end-start), endpoint=False)
    colors = ('r', 'g', 'b')
    plt.figure(figsize=(16,8))
    plt.subplot(2, 2, 1)
    for i in range(3):
        plt.plot(t, acc[:,i], color=COLORS[colors[i]], linestyle='-.')
    for i in range(3):
        plt.plot(t, augmented_acc[:,i], color=COLORS[colors[i]])
    plt.ylabel('加速度 (m/s^2)', fontsize=14)
    plt.legend(['X', 'Y', 'Z', 'X\'', 'Y\'', 'Z\''], loc='lower right')
    plt.grid('both', linestyle='--')
    plt.subplot(2, 2, 3)
    for i in range(3):
        plt.plot(t, gyro[:,i], color=COLORS[colors[i]], linestyle='-.')
    for i in range(3):
        plt.plot(t, augmented_gyro[:,i], color=COLORS[colors[i]])
    plt.ylabel('角速度 (rad/s)', fontsize=14)
    plt.xlabel('时间 (s)', fontsize=14)
    plt.legend(['X', 'Y', 'Z', 'X\'', 'Y\'', 'Z\''], loc='lower right')
    plt.grid('both', linestyle='--')
    ax = plt.subplot(2, 2, 2, projection='3d')
    ax.plot(center_pos[:,0], center_pos[:,1], center_pos[:,2], color=COLORS['o'])
    ax.plot(augmented_pos[:,0], augmented_pos[:,1], augmented_pos[:,2], color=COLORS['p'])
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.zaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_box_aspect((3, 1, 1))
    ax.set_xlim(-0.03, 0.27)
    ax.set_ylim(0.075, 0.125)
    ax.set_zlim(-0.025, 0.025)
    ax.legend(['原始运动轨迹', '增强后的运动轨迹'], bbox_to_anchor=(1.50, 0.1))     
    plt.subplot(2, 2, 4)
    for i in range(3):
        plt.plot(t, center_pos[:,i], color=COLORS[colors[i]], linestyle='-.')
    for i in range(3):
        plt.plot(t, augmented_pos[:,i], color=COLORS[colors[i]])
    plt.ylabel('空间位移 (m)', fontsize=14)
    plt.xlabel('时间 (s)', fontsize=14)
    plt.legend(['X', 'Y', 'Z', 'X\'', 'Y\'', 'Z\''], loc='lower right')
    plt.grid('both', linestyle='--')
    plt.show()


def calc_rmse_error():
    ''' Calculate the RMSE error of track2imu and imu2track on study-2 dataset.
    '''
    task_list_id = f'TLm5wv3uex'
    record_paths = glob(f'../data/record/{task_list_id}/TK*/ST*/RD*')
    acc_data, gyro_data = [], []
    track_acc_data, track_gyro_data = [], []
    center_pos_data = []
    imu_center_pos_data = []
    for record_path in tqdm.tqdm(record_paths):
        record = Record(record_path, n_sample=20)
        imu_data = record.cutted_imu_data
        acc, gyro = imu_data['acc'][:,50:450,:], imu_data['gyro'][:,50:450,:]
        acc_data.append(acc)
        gyro_data.append(gyro)
        track_data = record.cutted_track_data
        center_pos = track_data['center_pos'][:,50:450,:]
        marker_pos = track_data['marker_pos'][:,:,50:450,:]
        center_pos_data.append(center_pos)
        for i in range(center_pos.shape[0]):
            # track to imu
            axes = aug.calc_local_axes(marker_pos[i,:,:,:])
            track_gyro_data.append(aug.track_to_gyro(axes, cf.FS_PREPROCESS)[None,:,:])
            track_acc_data.append(aug.track_to_acc(center_pos[i,:,:], axes, cf.FS_PREPROCESS)[None,:,:])
            # imu to track
            bound_pos = np.concatenate([center_pos[i,:1,:],center_pos[i,-1:,:]], axis=0)
            bound_velocity = cf.FS_PREPROCESS * np.concatenate([center_pos[i,1:2,:]-center_pos[i,0:1,:],
                center_pos[i,-1:,:]-center_pos[i,-2:-1,:]], axis=0)
            bound_axes = np.concatenate([axes[:,:2,:],axes[:,-2:,:]], axis=1)
            imu_center_pos, _ = aug.imu_to_track(acc[i,:,:], gyro[i,:,:], bound_pos,
            bound_velocity, bound_axes, cf.FS_PREPROCESS)
            imu_center_pos_data.append(imu_center_pos[None,:,:])
    acc_data = np.concatenate(acc_data, axis=0)
    gyro_data = np.concatenate(gyro_data, axis=0)
    track_acc_data = np.concatenate(track_acc_data, axis=0)
    track_gyro_data = np.concatenate(track_gyro_data, axis=0)
    print(f'track to acc error: {aug.rmse_error(acc_data, track_acc_data):.6f}')
    print(f'track to gyro error: {aug.rmse_error(gyro_data, track_gyro_data):.6f}')
    center_pos_data = np.concatenate(center_pos_data, axis=0)
    imu_center_pos_data = np.concatenate(imu_center_pos_data, axis=0)
    print(f'imu to track error: {aug.rmse_error(center_pos_data, imu_center_pos_data):.6f}')
    
    
def visualize_move_and_rotate():
    ''' Visualize imu signal of move 10-40, rotate 45-180, medium.
    '''
    task_list_id = 'TLm5wv3uex'
    task_list = fu.load_task_list_with_users(task_list_id)
    record_paths = []
    task_names = ('Move10', 'Move20', 'Move30', 'Move40', 'Rotate45', 'Rotate90', 'Rotate135', 'Rotate180')
    task_names_zh = ('移动 10 厘米', '移动 20 厘米', '移动 30 厘米', '移动 40 厘米', 
        '旋转 45 度', '旋转 90 度', '旋转 135 度', '旋转 180 度')
    subtask_name = 'Medium'
    user_name = 'lzj'
    for task_name in task_names:
        for task in task_list['tasks']:
            if task['name'] == task_name: break
        assert task['name'] == task_name
        for subtask in task['subtasks']:
            if subtask['name'] == subtask_name: break
        assert subtask['name'] == subtask_name
        record_id = subtask['record_dict'][user_name]
        record_path = fu.get_record_path(task_list_id, task['id'], subtask['id'], record_id)
        record_paths.append(record_path)
    # plot imu signals
    idx = 10
    start, end = 50, 450
    t = np.linspace(0, 2, num=(end-start), endpoint=False)
    plt.figure(figsize=(20,10))
    for i, record_path in enumerate(record_paths):
        record = Record(record_path, n_sample=20)
        imu_data = record.cutted_imu_data
        acc = imu_data['acc'][idx,start:end,:]
        gyro = imu_data['gyro'][idx,start:end,:]
        colors = ('r', 'g', 'b')
        if i < 4: idx = i + 1
        else: idx = i + 5
        plt.subplot(4, 4, idx)
        for j in range(3):
            plt.plot(t, acc[:,j], color=COLORS[colors[j]])
        if i < 4: plt.ylim(-13, 19)
        else: plt.ylim(-16, 16)
        plt.grid('both', linestyle='--')
        plt.title(task_names_zh[i], fontsize=14)
        plt.xticks(ticks=[0,0.5,1.0,1.5,2.0], labels=['']*5)
        if i == 0 or i == 4:
            plt.yticks(ticks=[-10,0,10])
            plt.ylabel('加速度 (m/s^2)', fontsize=14)
        else: plt.yticks(ticks=[-10,0,10], labels=['']*3)
        plt.subplot(4, 4, idx+4)
        for j in range(3):
            plt.plot(t, gyro[:,j], color=COLORS[colors[j]])
        plt.ylim(-16, 16)
        plt.grid('both', linestyle='--')
        if i < 4:
            plt.xticks(ticks=[0,0.5,1.0,1.5,2.0], labels=['']*5)
        else: plt.xticks(ticks=[0,0.5,1.0,1.5,2.0])
        if i == 0 or i == 4:
            plt.yticks(ticks=[-10,0,10])
            plt.ylabel('角速度 (rad/s)', fontsize=14)
        else: plt.yticks(ticks=[-10,0,10], labels=['']*3)
        if i >= 4:
            plt.xlabel('时间 (t)', fontsize=14)
        if i == 7:
            plt.legend(['X', 'Y', 'Z'], bbox_to_anchor=(1.28, 0.46))
            
    plt.show()
    
    
def visualize_markers(record:Record) -> None:
    ''' Visualize the marker positions.
    '''
    track_data = record.track_data
    center_pos = track_data['center_pos'] * 1e3
    marker_pos = track_data['marker_pos'] * 1e3
    axes = aug.calc_local_axes(marker_pos)
    center_pos -= marker_pos[0,:,:]
    center_pos = np.sum(center_pos[None,:,:] * axes, axis=2).T
    center_pos = np.mean(center_pos, axis=0)
    marker_pos -= marker_pos[0:1,:,:]
    for i in range(marker_pos.shape[0]):
        marker_pos[i,:,:] = np.sum((marker_pos[i:i+1,:,:]) * axes, axis=2).T
    marker_pos = np.mean(marker_pos, axis=1)
    np.set_printoptions(formatter={'float': ' {:0.1f} '.format})
    print(f'### center_pos (mm):')
    print(center_pos[:2])
    print(f'### marker_pos (mm):')
    print(marker_pos[:,:2])
    plt.scatter([center_pos[0]], [center_pos[1]])
    plt.scatter(marker_pos[:,0], marker_pos[:,1])
    plt.gca().set_aspect(1)
    plt.title(f'Marker Positions', fontsize=16)
    plt.xlabel(f'x (mm)', fontsize=14)
    plt.ylabel(f'y (mm)', fontsize=14)
    plt.show()
    

def visualize_cutter(record:Record):
    cutted_imu_data = record.cutted_imu_data
    print(f'acc: {cutted_imu_data["acc"].shape}')
    print(f'gyro: {cutted_imu_data["gyro"].shape}')
    cutted_track_data = record.cutted_track_data
    print(f'center_pos: {cutted_track_data["center_pos"].shape}')
    print(f'marker_pos: {cutted_track_data["marker_pos"].shape}')
    
    
def visualize_filter(record:Record):
    gyro = record.imu_data['gyro']
    norm = np.sqrt(np.sum(np.square(gyro), axis=1))
    butter = Butterworth(cf.FS_PREPROCESS, 0.003*cf.FS_PREPROCESS, 'lowpass', order=4)
    filtered_norm = butter.filt(norm)
    for i in range(3):
        plt.plot(gyro[:,i])
    plt.plot(filtered_norm, color='red')
    for i in range(21):
        plt.plot([i*600, i*600], [-10, 10], color='black')
    plt.show()
    
    
def visualize_error():
    task_list_id = 'TLnmdi15b8'
    task_ids = ['TK7t3ql6jb', 'TK9fe2fbln', 'TK5rsia9fw', 'TKtvkgst8r', 'TKie8k1h6r']
    root_list = fu.load_root_list_info()
    task_lists = root_list['tasklists']
    for task_list in task_lists:
        if task_list['id'] == task_list_id: break
    assert task_list['id'] == task_list_id
    W = 600
    acc_stds = []
    gyro_stds = []
    for task_id in task_ids:
        for task in task_list['tasks']:
            if task['id'] == task_id: break
        assert task['id'] == task_id
        for subtask in task['subtasks']:
            subtask_id = subtask['id']
            n_sample = subtask['times']
            record_paths = glob(f'{fu.DATA_RECORD_ROOT}/{task_list_id}/{task_id}/{subtask_id}/RD*')
            for record_path in tqdm.tqdm(record_paths):
                record = Record(record_path, n_sample)
                imu_data = record.imu_data
                acc = imu_data['acc']
                gyro = imu_data['gyro']
                track_data = record.track_data
                center_pos = track_data['center_pos']
                marker_pos = track_data['marker_pos']
                axes = aug.calc_local_axes(marker_pos)
                generated_acc = aug.track_to_acc(center_pos, axes, cf.FS_PREPROCESS)
                generated_gyro = aug.track_to_gyro(axes, cf.FS_PREPROCESS)
                for i in range(int(np.round(gyro.shape[0]//W))):
                    # acc_std = np.std(acc[i*W:(i+1)*W,:])
                    # acc_error = aug.mse_error(acc[i*W:(i+1)*W,:], generated_acc[i*W:(i+1)*W,:])
                    acc_std = np.std(acc[i*W:(i+1)*W,:])
                    acc_stds.append(acc_std)
                    # gyro_error = aug.mse_error(gyro[i*W:(i+1)*W,:], generated_gyro[i*W:(i+1)*W,:])
                    gyro_std = np.std(gyro[i*W:(i+1)*W,:])
                    gyro_stds.append(gyro_std)
    acc_stds = np.sort(acc_stds)
    gyro_stds = np.sort(gyro_stds)
    plt.subplot(2, 1, 1)
    plt.plot(acc_stds)
    plt.subplot(2, 1, 2)
    plt.plot(gyro_stds)
    plt.show()
    
    
def visualize_clean_mask():
    clean_mask = np.zeros(0, dtype=np.bool8)
    task_list_id = 'TLnmdi15b8'
    task_ids = ['TK7t3ql6jb', 'TK9fe2fbln', 'TK5rsia9fw', 'TKtvkgst8r', 'TKie8k1h6r']
    root_list = fu.load_root_list_info()
    task_lists = root_list['tasklists']
    for task_list in task_lists:
        if task_list['id'] == task_list_id: break
    assert task_list['id'] == task_list_id
    for task_id in task_ids:
        for task in task_list['tasks']:
            if task['id'] == task_id: break
        assert task['id'] == task_id
        for subtask in task['subtasks']:
            subtask_id = subtask['id']
            n_sample = subtask['times']
            record_paths = glob(f'{fu.DATA_RECORD_ROOT}/{task_list_id}/{task_id}/{subtask_id}/RD*')
            for record_path in tqdm.tqdm(record_paths):
                record = Record(record_path, n_sample)
                clean_mask = np.concatenate([clean_mask, record.clean_mask])
    W = int(np.sqrt(len(clean_mask))) + 1
    img = np.zeros((W, W, 3), dtype=np.uint8)
    for k in range(len(clean_mask)):
        i, j = k // W, k % W
        if clean_mask[k]:
            img[i,j,:] = 0, 255, 0
        else: img[i,j,:] = 0, 0, 255
    img = cv2.resize(img, (W*4, W*4))
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    
def visualize_cleaned_negative_data():
    paths = glob(f'../data/negative/day*/*.pkl')
    np.random.shuffle(paths)
    paths = paths[:10]
    for path in paths:
        file_name = path.split('/')[-1].split('.')[0]
        data = pickle.load(open(path, 'rb'))
        plt.subplot(2, 1, 1)
        for i in range(3): plt.plot(data[:,i])
        plt.subplot(2, 1, 2)
        for i in range(3): plt.plot(data[:,i+3])
        plt.savefig(f'../data/media/{file_name}.jpg')
        plt.clf()
        

def visualize_tsne():
    data = []
    labels = []
    task_list_id = 'TLm5wv3uex'
    days = [1, 2, 3, 4, 6]
    negative_paths = []
    for day in days:
        negative_paths.extend(glob(f'../data/negative/day{day}/*.pkl'))
    
    # load task list
    task_list = fu.load_task_list_with_users(task_list_id)
    assert task_list is not None
            
    # load positive data
    print('### Load positive data')
    # task_names = ['Move10', 'Move20', 'Move30', 'Move40']
    # task_names_zh = ['移动 10 厘米', '移动 20 厘米', '移动 30 厘米', '移动 40 厘米']
    task_names = ['Rotate45', 'Rotate90', 'Rotate135', 'Rotate180']
    task_names_zh = ['旋转 45 度', '旋转 90 度', '旋转 135 度', '旋转 180 度']
    
    for task_name, label in zip(task_names, (1, 2, 3, 4)):
        for task in task_list['tasks']:
            if task['name'] == task_name: break
        assert task['name'] == task_name
        for subtask in task['subtasks']:
            for record_id in list(subtask['record_dict'].values()):
                record_path = fu.get_record_path(task_list_id, task['id'], subtask['id'], record_id)
                record = Record(record_path, n_sample=20)
                imu_data = record.cutted_imu_data
                acc = imu_data['acc']
                gyro = imu_data['gyro']
                imu = np.concatenate([acc, gyro], axis=2)
                imu = aug.down_sample_by_step(imu, axis=1, step=2)
                data.append(imu)
                labels.append(np.zeros(imu.shape[0], dtype=np.int32) + label)
    
    data = np.concatenate(data, axis=0)
    data = np.reshape(data, (data.shape[0], -1))
    labels = np.concatenate(labels, axis=0)
    
    # plot t-SNE graph
    plt.figure(figsize=(8, 6))
    color_map = {0:'black', 1:'r', 2:'o', 3:'g', 4:'b'}
    tic = time.perf_counter()
    tsne = TSNE()
    embedded = tsne.fit_transform(data)
    toc = time.perf_counter()
    print(f'### t-SNE time cost: {toc-tic:.3f} s')
    for label in (1, 2, 3, 4):
        plt.scatter(embedded[(labels==label),0], embedded[(labels==label),1], c=COLORS[color_map[label]], s=1)
    # plt.scatter(embedded[:,0], embedded[:,1], c=[COLORS[color] for color in colors], s=1)
    plt.legend(task_names_zh, loc='lower right')
    plt.grid('both', linestyle='--')
    plt.xlabel('t-SNE x', fontsize=14)
    plt.ylabel('t-SNE y', fontsize=14)
    plt.show()
    
    
def visualize_dtw_augment(record:Record):
    weight = 0.5
    idx1, idx2 = 15, 11
    track_data = record.cutted_track_data
    center_pos = track_data['center_pos']
    x1 = center_pos[idx1,:,:]
    x2 = center_pos[idx2,:,:]
    x3, pairs = aug.dtw_augment(x1, x2, axis=0, weight=weight, window=None)
    marker_pos = track_data['marker_pos']
    y1 = marker_pos[idx1,:,:,:]
    y2 = marker_pos[idx2,:,:,:]
    f1 = interp.interp1d(np.arange(500), y1, kind='quadratic', axis=1)
    f2 = interp.interp1d(np.arange(500), y2, kind='quadratic', axis=1)
    y3 = f1(pairs[:,0])*weight + f2(pairs[:,1])*(1-weight)
    axes1 = aug.calc_local_axes(y1)
    axes2 = aug.calc_local_axes(y2)
    axes3 = aug.calc_local_axes(y3)
    gyro1 = aug.track_to_gyro(axes1, fs=cf.FS_PREPROCESS)
    gyro2 = aug.track_to_gyro(axes2, fs=cf.FS_PREPROCESS)
    gyro3 = aug.track_to_gyro(axes3, fs=cf.FS_PREPROCESS)
    acc1 = aug.track_to_acc(x1, axes1, fs=cf.FS_PREPROCESS)
    acc2 = aug.track_to_acc(x2, axes2, fs=cf.FS_PREPROCESS)
    acc3 = aug.track_to_acc(x3, axes3, fs=cf.FS_PREPROCESS)
    plt.subplot(3, 2, 1)
    for i in range(3): plt.plot(x1[:,i])
    plt.subplot(3, 2, 3)
    for i in range(3): plt.plot(x3[:,i])
    plt.subplot(3, 2, 5)
    for i in range(3): plt.plot(x2[:,i])
    plt.subplot(3, 2, 2)
    for i in range(3): plt.plot(gyro1[:,i])
    plt.subplot(3, 2, 4)
    for i in range(3): plt.plot(gyro3[:,i])
    plt.subplot(3, 2, 6)
    for i in range(3): plt.plot(gyro2[:,i])
    plt.show()
    

def visualize_dtw_offset():
    bad_data = json.load(open(f'../data/bad_data.json', 'r'))
    np.random.shuffle(bad_data)
    for item in bad_data:
        record_path = item['record_path']
        n_sample = 40 if 'TK7t3ql6jb' in record_path else 20
        record = Record(record_path, n_sample)
        weight = 0.5
        idx1, idx2 = item['idx1'], item['idx2']
        track_data = record.cutted_track_data
        center_pos = track_data['center_pos']
        x1 = center_pos[idx1,:,:]
        x2 = center_pos[idx2,:,:]
        dx1 = np.diff(x1, axis=0, append=x1[-1:,:])
        dx2 = np.diff(x2, axis=0, append=x2[-1:,:])
        warping_path = aug.dtw_match(dx1, dx2, axis=0, window=200)
        x3 = aug.dtw_augment(x1, x2, warping_path, axis=0, weight=weight)
        marker_pos = track_data['marker_pos']
        y1 = marker_pos[idx1,:,:,:]
        y2 = marker_pos[idx2,:,:,:]
        y3 = aug.dtw_augment(y1, y2, warping_path, axis=1, weight=weight)
        axes1 = aug.calc_local_axes(y1)
        axes2 = aug.calc_local_axes(y2)
        axes3 = aug.calc_local_axes(y3)
        gyro1 = aug.track_to_gyro(axes1, fs=cf.FS_PREPROCESS)
        gyro2 = aug.track_to_gyro(axes2, fs=cf.FS_PREPROCESS)
        gyro3 = aug.track_to_gyro(axes3, fs=cf.FS_PREPROCESS)
        print(f'### {record_path}, {idx1}, {idx2}')
        plt.subplot(3, 2, 1)
        for i in range(3): plt.plot(x1[:,i])
        plt.subplot(3, 2, 3)
        for i in range(3): plt.plot(x3[:,i])
        plt.subplot(3, 2, 5)
        for i in range(3): plt.plot(x2[:,i])
        plt.subplot(3, 2, 2)
        for i in range(3): plt.plot(gyro1[:,i])
        plt.subplot(3, 2, 4)
        for i in range(3): plt.plot(gyro3[:,i])
        plt.subplot(3, 2, 6)
        for i in range(3): plt.plot(gyro2[:,i])
        plt.show()    
        
        
def visualize_scale_gyro(record:Record):
    idx = 5
    params = aug.magnitude_warp_params(n_knots=16, std=0.2)
    start, end = 100, 400
    cutted_imu_data = record.cutted_imu_data
    acc = cutted_imu_data['acc'][idx,start:end,:]
    gyro = cutted_imu_data['gyro'][idx,start:end,:]
    scaled_acc = aug.magnitude_warp(acc, axis=0, params=params)
    scaled_gyro = aug.magnitude_warp(gyro, axis=0, params=params)
    
    cutted_track_data = record.cutted_track_data
    center_pos = cutted_track_data['center_pos'][idx,start:end,:]
    marker_pos = cutted_track_data['marker_pos'][idx,:,start:end,:]
    axes = aug.calc_local_axes(marker_pos)
    track_acc = aug.track_to_acc(center_pos, axes, fs=cf.FS_PREPROCESS)
    track_gyro = aug.track_to_gyro(axes, fs=cf.FS_PREPROCESS)
    scaled_center_pos = aug.magnitude_warp(center_pos, axis=0, params=params)
    axes = aug.calc_local_axes(marker_pos)
    q = axes.transpose(1, 2, 0)     # rotation matrix
    delta_q = np.matmul(np.linalg.inv(q[0,:,:])[None,:,:], q)
    rotvec = Rotation.from_matrix(delta_q).as_rotvec()
    scaled_rotvec = aug.magnitude_warp(rotvec, axis=0, params=params)
    scaled_axes = np.matmul(q[0:1,:,:], Rotation.from_rotvec(scaled_rotvec).as_matrix()).transpose(2,0,1)
    scaled_track_acc = aug.track_to_acc(scaled_center_pos, scaled_axes, fs=cf.FS_PREPROCESS)
    scaled_track_gyro = aug.track_to_gyro(scaled_axes, fs=cf.FS_PREPROCESS)
     
    plt.subplot(2, 2, 1)
    for i in range(3): plt.plot(acc[:,i], color='blue')
    for i in range(3): plt.plot(scaled_acc[:,i], color='red')
    plt.ylabel(f'Acc', fontsize=14)
    plt.subplot(2, 2, 3)
    for i in range(3): plt.plot(gyro[:,i], color='blue')
    for i in range(3): plt.plot(scaled_gyro[:,i], color='red')
    plt.ylabel(f'Gyro', fontsize=14)
    plt.subplot(2, 2, 2)
    for i in range(3): plt.plot(track_acc[:,i], color='blue')
    for i in range(3): plt.plot(scaled_track_acc[:,i], color='red')
    plt.ylabel(f'Track Acc', fontsize=14)
    plt.subplot(2, 2, 4)
    for i in range(3): plt.plot(track_gyro[:,i], color='blue')
    for i in range(3): plt.plot(scaled_track_gyro[:,i], color='red')
    plt.ylabel(f'Track Gyro', fontsize=14)
    plt.show()
    
    
def visualize_3d_track_by_subtask():
    ''' Visulize the 3d track orientations of different subtasks.
    '''
    task_list_id = f'TLnmdi15b8'
    task_list = fu.load_task_list_with_users(task_list_id)
    task_id = f'TK9fe2fbln'
    user_name = f'zxyx'
    subtask_ids = [f'ST4rcuu4mk', f'ST1d4ykyui']
    rot_matrix = np.array([[-1,0,0],[0,0,1],[0,1,0]], dtype=np.float32)
    rot_x_90 = Rotation.from_rotvec(np.array([0.5*np.pi,0,0])).as_matrix()
    colors = ['blue', 'red']
    ax = plt.axes(projection='3d')
    l1, l2 = None, None
    
    for task in task_list['tasks']:
        if task['id'] == task_id: break
    assert task['id'] == task_id
    for subtask_id, color in zip(subtask_ids, colors):
        for subtask in task['subtasks']:
            if subtask['id'] == subtask_id: break
        assert subtask['id'] == subtask_id
        record_id = subtask['record_dict'][user_name]
        record_path = fu.get_record_path(task_list_id, task_id, subtask_id, record_id)
        record = Record(record_path, n_sample=subtask['times'])
        center_pos = record.cutted_track_data['center_pos'][:,150:350,:]
        # plot track
        for segment in center_pos:
            segment = np.matmul(rot_matrix, segment.T).T
            # if subtask_id == 'ST4rcuu4mk':
            #     segment = np.matmul(rot_x_90, segment.T).T
            segment -= np.mean(segment, axis=0)[None,:]
            l, = ax.plot(segment[:,0], segment[:,1], segment[:,2], color=color)
            if subtask_id == 'ST4rcuu4mk':
                if l1 is None: l1 = l
            elif l2 is None: l2 = l
        # set equal box aspect
        x_scale = np.abs(np.diff(ax.get_xlim()))[0]
        y_scale = np.abs(np.diff(ax.get_ylim()))[0]
        z_scale = np.abs(np.diff(ax.get_zlim()))[0]
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.zaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_box_aspect((x_scale,y_scale,z_scale))
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.legend(handles=[l1, l2], labels=['StandUp', 'LieDown'])
    plt.title(f'StandUp and LieDown of {user_name}')
    plt.show()
    
    
def visualize_rotated_imu_signal(record:Record):
    ''' Rotate action track and convert to imu. Visualize the imu signal difference.
    '''
    track_data = record.cutted_track_data
    center_pos = track_data['center_pos'][5,:,:]
    marker_pos = track_data['marker_pos'][5,:,:,:]
    rot = Rotation.from_rotvec(np.array([-0.5*np.pi,0,0])).as_matrix()
    rot_center_pos = np.matmul(rot, center_pos.T).T
    rot_marker_pos = np.empty_like(marker_pos)
    for i in range(6): rot_marker_pos[i,:,:] = np.matmul(rot, marker_pos[i,:,:].T).T
    axes = aug.calc_local_axes(marker_pos)
    acc = aug.track_to_acc(center_pos, axes, fs=cf.FS_PREPROCESS)
    gyro = aug.track_to_gyro(axes, fs=cf.FS_PREPROCESS)
    rot_axes = aug.calc_local_axes(rot_marker_pos)
    rot_acc = aug.track_to_acc(rot_center_pos, rot_axes, fs=cf.FS_PREPROCESS)
    rot_gyro = aug.track_to_gyro(rot_axes, fs=cf.FS_PREPROCESS)
    plt.subplot(2, 1, 1)
    for i in range(3): plt.plot(acc[:,i], color='b')
    for i in range(3): plt.plot(rot_acc[:,i], color='r')
    plt.ylabel('Acc', fontsize=14)
    plt.subplot(2, 1, 2)
    for i in range(3): plt.plot(gyro[:,i], color='b')
    for i in range(3): plt.plot(rot_gyro[:,i], color='r')
    plt.ylabel('Gyro', fontsize=14)
    plt.show()
    
    
def visualize_move():
    ''' Visualize Move in different distance and speed.
    '''
    user_name, idx = 'lzj2', 10
    task_list = fu.load_task_list_with_users('TL3wni1oq3')
    for i, task_name in enumerate(('Move10cm', 'Move20cm', 'Move30cm', 'Move40cm', 'Move50cm')):
        for task in task_list['tasks']:
            if task['name'] == task_name: break
        assert task['name'] == task_name
        for j, subtask_name in enumerate(('Fast', 'Medium', 'Slow')):
            for subtask in task['subtasks']:
                if subtask['name'] == subtask_name: break
            assert subtask['name'] == subtask_name
            record_id = subtask['record_dict'][user_name]
            record_path = fu.get_record_path(task_list['id'], task['id'], subtask['id'], record_id)
            record = Record(record_path, subtask['times'])
            imu = record.cutted_imu_data
            acc = imu['acc'][idx,:,:]
            plt.subplot(5, 3, i*3+j+1)
            for k in range(3): plt.plot(acc[:,k])
            plt.ylim(-25, 25)
            plt.title(f'{task_name}.{subtask_name}')
    plt.show()
    
    
def visualize_move_distance():
    ''' Visualize the actual move distance of the action samples.
    '''
    task_list = fu.load_task_list_with_users('TLm5wv3uex')
    task_names = ['Move10', 'Move20', 'Move30', 'Move40']
    task_names_zh = ['移动 10 厘米', '移动 20 厘米', '移动 30 厘米', '移动 40 厘米']
    distance = {task_name: [] for task_name in task_names}
    for task_name in task_names:
        for task in task_list['tasks']:
            if task['name'] == task_name: break
        assert task['name'] == task_name
        record_paths = glob(f'../data/record/{task_list["id"]}/{task["id"]}/ST*/RD*')
        for record_path in record_paths:
            record = Record(record_path, n_sample=20)
            track_data = record.cutted_track_data
            center_pos = track_data['center_pos']
            for i in range(center_pos.shape[0]):
                dis = np.max(center_pos[i,:,0]) - np.min(center_pos[i,:,0])
                distance[task_name].append(dis * 100)
    for task_name in task_names:
        dis_arr = distance[task_name]
        print(f'{task_name}: M = {np.mean(dis_arr):.2f}, SD = {np.std(dis_arr,ddof=1):.2f}')
    plt.figure(figsize=(8, 6))
    plt.violinplot([distance[task_name] for task_name in task_names], positions=range(4))
    plt.xticks(ticks=range(4), labels=task_names_zh, fontsize=14)
    plt.grid(axis='both', linestyle='--')
    plt.ylabel('移动距离 (cm)', fontsize=14)
    plt.ylim(0, 60)
    plt.show()
    

def visualize_rotate_angle():
    ''' Visualize the actual rotate angle of the action samples.
    '''
    task_list = fu.load_task_list_with_users('TLm5wv3uex')
    task_names = ['Rotate45', 'Rotate90', 'Rotate135', 'Rotate180']
    task_names_zh = ['旋转 45 度', '旋转 90 度', '旋转 135 度', '旋转 180 度']
    angles = {task_name: [] for task_name in task_names}
    for task_name in task_names:
        for task in task_list['tasks']:
            if task['name'] == task_name: break
        assert task['name'] == task_name
        record_paths = glob(f'../data/record/{task_list["id"]}/{task["id"]}/ST*/RD*')
        for record_path in record_paths:
            record = Record(record_path, n_sample=20)
            track_data = record.cutted_track_data
            marker_pos = track_data['marker_pos']
            for i in range(marker_pos.shape[0]):
                axes = aug.calc_local_axes(marker_pos[i,...])
                q = axes.transpose(1,2,0)
                q = np.matmul(q, q[0,:,:].transpose())
                rot_vec = Rotation.from_matrix(q).as_rotvec()
                mask = rot_vec[:,2] > 1
                norm = np.sqrt(np.sum(np.square(rot_vec), axis=1))
                norm[mask] = 2 * np.pi - norm[mask]
                angle = np.max(norm) * 180 / np.pi
                angles[task_name].append(angle)
                
    for task_name in task_names:
        angle_arr = angles[task_name]
        print(f'{task_name}: M = {np.mean(angle_arr):.2f}, SD = {np.std(angle_arr,ddof=1):.2f}')
    
    plt.figure(figsize=(8, 6))
    plt.violinplot([angles[task_name] for task_name in task_names], positions=range(4))
    plt.xticks(ticks=range(4), labels=task_names_zh, fontsize=14)
    plt.grid(axis='both', linestyle='--')
    plt.ylabel('旋转角度 (度)', fontsize=14)
    plt.ylim(0, 240)
    plt.show()
   
   
def model_structure():
    model = Model4()
    params = model.parameters()
    total = 0
    for item in params:
        print(item.shape, np.prod(item.shape))
        total += np.prod(item.shape)
    print(f'### total: {total / 1000} k')
 
        
if __name__ == '__main__':
    np.random.seed(0)
    fu.check_cwd()
    # task_list_id = 'TLm5wv3uex'
    # task_id = 'TK7mek47cs'
    # subtask_id = 'STu1wy07jb'
    # record_id = 'RDn4jpscoo'
    # record_path = fu.get_record_path(task_list_id, task_id, subtask_id, record_id)
    # tic = time.perf_counter()
    # record = Record(record_path, n_sample=20)
    # toc = time.perf_counter()
    # print(f'time: {(toc-tic)*1000:.3f} ms')
    
    visualize_tsne()
    
