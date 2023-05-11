import os
import cv2
import time
import tqdm
import json
import pickle
import numpy as np
import pandas as pd
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
    # prepare imu data
    imu_data = record.cutted_imu_data
    acc = imu_data['acc'][idx,:,:]
    gyro = imu_data['gyro'][idx,:,:]
    # prepare track data
    track_data = record.cutted_track_data
    center_pos = track_data['center_pos'][idx,:,:]
    marker_pos = track_data['marker_pos'][idx,:,:,:]
    axes = aug.calc_local_axes(marker_pos)
    track_acc = aug.track_to_acc(center_pos, axes, cf.FS_PREPROCESS)
    track_gyro = aug.track_to_gyro(axes, cf.FS_PREPROCESS)
    # plot acc data
    plt.subplot(2, 1, 1)
    for i in range(3): plt.plot(acc[:,i])
    for i in range(3): plt.plot(track_acc[:,i])
    plt.ylabel('Accelerometer', fontsize=14)
    plt.legend(['X', 'Y', 'Z', 'X\'', 'Y\'', 'Z\''], loc='lower right')
    # plot gyro data
    plt.subplot(2, 1, 2)
    for i in range(3): plt.plot(gyro[:,i])
    for i in range(3): plt.plot(track_gyro[:,i])
    plt.ylabel('Gyroscope', fontsize=14)
    plt.legend(['X', 'Y', 'Z', 'X\'', 'Y\'', 'Z\''], loc='lower right')
    plt.show()
    return
    
    
def visualize_markers(record:Record) -> None:
    ''' Visualize the marker positions.
    '''
    track_data = record.track_data
    center_pos = track_data['center_pos']
    marker_pos = track_data['marker_pos']
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
    

def visualize_augmented_3d_track(record:Record) -> None:
    # prepare imu data
    imu_data = record.imu_data
    start_imu, end_imu = 3000, 3500
    acc = imu_data['acc'][start_imu:end_imu,:]
    acc = aug.down_sample_by_step(acc, axis=0, step=5)
    # augmented_acc = aug.jitter(acc, std=1)
    # augmented_acc = aug.scale(acc, 0.05)
    augmented_acc = aug.time_warp(acc, axis=0, n_knots=4, std=0.05)
    # augmented_acc = aug.magnitude_warp(acc, axis=0, n_knots=4, std=0.1)
    gyro = imu_data['gyro'][start_imu:end_imu,:]
    gyro = aug.down_sample_by_step(gyro, axis=0, step=5)
    # prepare track data
    track_data = record.track_data
    start_track, end_track = 3134, 3334
    center_pos = track_data['center_pos'][start_track:end_track,:]
    marker_pos = track_data['marker_pos'][:,start_track:end_track,:]
    center_pos = aug.down_sample_by_step(center_pos, axis=0, step=2)
    marker_pos = aug.down_sample_by_step(marker_pos, axis=1, step=2)
    axes = aug.calc_local_axes(marker_pos)
    # convert imu data to track data
    bound_pos = np.row_stack([center_pos[0,:], center_pos[-1,:]])
    bound_velocity = np.row_stack([center_pos[1,:]-center_pos[0,:],
        center_pos[-1,:]-center_pos[-2,:]])
    bound_axes = np.concatenate([axes[:,0:2,:], axes[:,-2:,:]], axis=1)
    converted_pos, converted_axes = aug.imu_to_track(augmented_acc, gyro,
        bound_pos, bound_velocity, bound_axes, 100.0)
    # rotate x, y, z to -x, z, y
    rot_matrix = np.array([[-1,0,0],[0,0,1],[0,1,0]], dtype=np.float32)
    center_pos = np.matmul(rot_matrix, center_pos.T).T
    converted_pos = np.matmul(rot_matrix, converted_pos.T).T
    # plot track
    ax = plt.axes(projection='3d')
    ax.plot(center_pos[:,0], center_pos[:,1], center_pos[:,2], color='blue')
    ax.plot(converted_pos[:,0], converted_pos[:,1], converted_pos[:,2], color='orange')
    # set equal box aspect
    x_scale = np.abs(np.diff(ax.get_xlim()))[0]
    y_scale = np.abs(np.diff(ax.get_ylim()))[0]
    z_scale = np.abs(np.diff(ax.get_zlim()))[0]
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.zaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_box_aspect((x_scale,y_scale,z_scale))
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Smartphone Track Visualization', fontsize=16)
    plt.show()
    
    
def visualize_imu_to_track(record:Record):
    imu_data = record.imu_data
    track_data = record.track_data
    start_imu, end_imu = 3000, 3500
    start_track, end_track = 3134, 3334
    acc, gyro = imu_data['acc'][start_imu:end_imu,:], imu_data['gyro'][start_imu:end_imu,:]
    center_pos = track_data['center_pos'][start_track:end_track,:]
    marker_pos = track_data['marker_pos'][:,start_track:end_track,:]
    axes = aug.calc_local_axes(marker_pos)
    # down_sample
    acc = aug.down_sample_by_step(acc, axis=0, step=5)
    gyro = aug.down_sample_by_step(gyro, axis=0, step=5)
    center_pos = aug.down_sample_by_step(center_pos, axis=0, step=2)
    axes = aug.down_sample_by_step(axes, axis=1, step=2)
    bound_pos = np.concatenate([center_pos[:1,:],center_pos[-1:,:]], axis=0)
    bound_velocity = 100*np.concatenate([center_pos[1:2,:]-center_pos[0:1,:],
        center_pos[-1:,:]-center_pos[-2:-1,:]], axis=0)
    bound_axes = np.concatenate([axes[:,:2,:],axes[:,-2:,:]], axis=1)
    generated_pos, generated_axes = aug.imu_to_track(acc, gyro, bound_pos,
        bound_velocity, bound_axes, 100.0)
    mse = aug.mse_error(center_pos, generated_pos)
    print(f'MSE Error: {mse:.6f}.')
    # rotate x, y, z, to -x, z, y
    rot_matrix = np.array([[-1,0,0],[0,0,1],[0,1,0]], dtype=np.float32)
    center_pos = np.matmul(rot_matrix, center_pos.T).T
    generated_pos = np.matmul(rot_matrix, generated_pos.T).T
    # plot 3d track
    ax = plt.subplot(1, 2, 1, projection='3d')
    ax.plot(center_pos[:,0], center_pos[:,1], center_pos[:,2], color='blue')
    ax.plot(generated_pos[:,0], generated_pos[:,1], generated_pos[:,2], color='orange')
    x_scale = np.abs(np.diff(ax.get_xlim()))[0]
    y_scale = np.abs(np.diff(ax.get_ylim()))[0]
    z_scale = np.abs(np.diff(ax.get_zlim()))[0]
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.zaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_box_aspect((x_scale,y_scale,z_scale))
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(['Original', 'Generated'], loc='lower right')
    plt.title(f'Position Track (m)', fontsize=16)
    # plot signals
    plt.subplot(1, 2, 2)
    for i in range(3):
        plt.plot(center_pos[:,i])
    for i in range(3):
        plt.plot(generated_pos[:,i])
    plt.grid(axis='both', linestyle='--')
    plt.legend(['X', 'Y', 'Z', 'X\'', 'Y\'', 'Z\''], loc='lower right')
    plt.title(f'Position Signals (m)', fontsize=16)
    plt.show()
    

def augment_track(record:Record):
    # prepare imu data
    imu_data = record.imu_data
    start_imu, end_imu = 2500, 5000
    acc = imu_data['acc'][start_imu:end_imu,:]
    gyro = imu_data['gyro'][start_imu:end_imu,:]
    # prepare track data
    track_data = record.track_data
    start_track, end_track = 2934, 3934
    center_pos = track_data['center_pos'][start_track:end_track,:]
    marker_pos = track_data['marker_pos'][:,start_track:end_track,:]
    axes = aug.calc_local_axes(marker_pos)
    # augment data
    # augmented_pos = aug.jitter(center_pos, std=1e-3)
    # augmented_pos = aug.scale(center_pos, std=0.05)
    augmented_pos = aug.magnitude_warp(center_pos, axis=0, n_knots=8, std=0.1)
    # augmented_pos = aug.time_warp(center_pos, axis=0, n_knots=4, std=0.1)
    augmented_acc = aug.track_to_acc(augmented_pos, axes, 200.0)
    augmented_gyro = aug.track_to_gyro(axes, 200.0)
    # downsample data
    center_pos = aug.down_sample_by_step(center_pos, axis=0, step=2)
    augmented_pos = aug.down_sample_by_step(augmented_pos, axis=0, step=2)
    acc = aug.down_sample_by_step(acc, axis=0, step=5)
    augmented_acc = aug.down_sample_by_step(augmented_acc, axis=0, step=2)
    gyro = aug.down_sample_by_step(gyro, axis=0, step=5)
    augmented_gyro = aug.down_sample_by_step(augmented_gyro, axis=0, step=2)
    # rotate x, y, z, to -x, z, y
    rot_matrix = np.array([[-1,0,0],[0,0,1],[0,1,0]], dtype=np.float32)
    center_pos = np.matmul(rot_matrix, center_pos.T).T
    augmented_pos = np.matmul(rot_matrix, augmented_pos.T).T
    # plot 3d track
    grids = gs.GridSpec(2, 3)
    ax = plt.subplot(grids[:,0], projection='3d')
    ax.plot(center_pos[:,0], center_pos[:,1], center_pos[:,2], color='blue')
    ax.plot(augmented_pos[:,0], augmented_pos[:,1], augmented_pos[:,2], color='orange')
    x_scale = np.abs(np.diff(ax.get_xlim()))[0]
    y_scale = np.abs(np.diff(ax.get_ylim()))[0]
    z_scale = np.abs(np.diff(ax.get_zlim()))[0]
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.zaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_box_aspect((x_scale,y_scale,z_scale))
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(['Original', 'Generated'], loc='lower right')
    ax.set_title(f'3D Track', fontsize=14)
    # plot track data
    ax = plt.subplot(grids[:,1])
    for i in range(3): ax.plot(center_pos[:,i])
    for i in range(3): ax.plot(augmented_pos[:,i])
    ax.legend(['X','Y','Z','X\'','Y\'','Z\''], loc='lower right')
    ax.set_title(f'Global Position', fontsize=14)
    # plot acc and gyro data
    ax = plt.subplot(grids[0,2])
    for i in range(3): ax.plot(acc[:,i])
    for i in range(3): ax.plot(augmented_acc[:,i])
    ax.set_title(f'Acc and Gyro', fontsize=14)
    ax = plt.subplot(grids[1,2])
    for i in range(3): ax.plot(gyro[:,i])
    for i in range(3): ax.plot(augmented_gyro[:,i])
    plt.show()
    
    
def augment_imu(record:Record):
    # prepare imu data
    imu_data = record.imu_data
    start_imu, end_imu = 3000, 3500
    acc = imu_data['acc'][start_imu:end_imu,:]
    gyro = imu_data['gyro'][start_imu:end_imu,:]
    # prepare track data
    track_data = record.track_data
    start_track, end_track = 3134, 3334
    center_pos = track_data['center_pos'][start_track:end_track,:]
    marker_pos = track_data['marker_pos'][:,start_track:end_track,:]
    center_pos = aug.resample(center_pos, axis=0, ratio=2.5)
    marker_pos = aug.resample(marker_pos, axis=1, ratio=2.5)
    axes = aug.calc_local_axes(marker_pos)
    # augment data
    # augmented_acc = aug.jitter(acc, std=1.0)
    # augmented_acc = aug.scale(acc, std=0.1)
    # augmented_acc = aug.magnitude_warp(acc, axis=0, n_knots=8, std=0.1)
    augmented_acc = aug.time_warp(acc, axis=0, n_knots=4, std=0.1)
    # augmented_gyro = gyro.copy()
    augmented_gyro = aug.time_warp(gyro, axis=0, n_knots=4, std=0.1)
    bound_pos = np.concatenate([center_pos[:1,:],center_pos[-1:,:]], axis=0)
    bound_velocity = 500 * np.concatenate([center_pos[1:2,:]-center_pos[0:1,:],
        center_pos[-1:,:]-center_pos[-2:-1,:]], axis=0)
    bound_axes = np.concatenate([axes[:,:2,:],axes[:,-2:,:]], axis=1)
    generated_pos, generated_axes = aug.imu_to_track(acc, gyro,
        bound_pos, bound_velocity, bound_axes, 500.0)
    augmented_pos, augmented_axes = aug.imu_to_track(augmented_acc, augmented_gyro,
        bound_pos, bound_velocity, bound_axes, 500.0)
    # rotate x, y, z, to -x, z, y
    rot_matrix = np.array([[-1,0,0],[0,0,1],[0,1,0]], dtype=np.float32)
    center_pos = np.matmul(rot_matrix, center_pos.T).T
    generated_pos = np.matmul(rot_matrix, generated_pos.T).T
    augmented_pos = np.matmul(rot_matrix, augmented_pos.T).T
    # plot 3d track
    grids = gs.GridSpec(2, 3)
    ax = plt.subplot(grids[:,0], projection='3d')
    ax.plot(center_pos[:,0], center_pos[:,1], center_pos[:,2], color='blue')
    ax.plot(generated_pos[:,0], generated_pos[:,1], generated_pos[:,2], color='green')
    ax.plot(augmented_pos[:,0], augmented_pos[:,1], augmented_pos[:,2], color='orange')
    x_scale = np.abs(np.diff(ax.get_xlim()))[0]
    y_scale = np.abs(np.diff(ax.get_ylim()))[0]
    z_scale = np.abs(np.diff(ax.get_zlim()))[0]
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.zaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_box_aspect((x_scale,y_scale,z_scale))
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(['Original', 'Generated', 'Augmented'], loc='lower right')
    ax.set_title(f'3D Track', fontsize=14)
    # plot track data
    ax = plt.subplot(grids[:,1])
    for i in range(3): ax.plot(center_pos[:,i])
    for i in range(3): ax.plot(augmented_pos[:,i])
    ax.legend(['X','Y','Z','X\'','Y\'','Z\''], loc='lower right')
    ax.set_title(f'Global Position', fontsize=14)
    # plot acc and gyro data
    ax = plt.subplot(grids[0,2])
    for i in range(3): ax.plot(acc[:,i])
    for i in range(3): ax.plot(augmented_acc[:,i])
    ax.set_title(f'Acc and Gyro', fontsize=14)
    ax = plt.subplot(grids[1,2])
    for i in range(3): ax.plot(gyro[:,i])
    for i in range(3): ax.plot(augmented_gyro[:,i])
    plt.show()
    

def visualize_cutter(record:Record):
    cutted_imu_data = record.cutted_imu_data
    print(f'acc: {cutted_imu_data["acc"].shape}')
    print(f'gyro: {cutted_imu_data["gyro"].shape}')
    cutted_track_data = record.cutted_track_data
    print(f'center_pos: {cutted_track_data["center_pos"].shape}')
    print(f'center_rot: {cutted_track_data["center_rot"].shape}')
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
    labels = np.zeros(0, dtype=np.int32)
    task_list_id = 'TLnmdi15b8'
    user_set = list(cf.USERS)
    np.random.shuffle(user_set)
    user_set = set(user_set)
    days = list(range(1,76))
    np.random.shuffle(days)
    day_set = set(days[:25])
    negative_paths = []
    for day in day_set:
        negative_paths.extend(glob(f'../data/negative/day{day}/*.pkl'))
    
    # load task list
    task_list = fu.load_task_list_with_users(task_list_id)
    assert task_list is not None
            
    # load Shake, DoubleShake, Flip, DoubleFlip by users
    print(f'### Load Shake, DoubleShake, Flip and DoubleFlip.')
    record_info = []
    for task_id, label in (('TK9fe2fbln', 3), ('TK5rsia9fw', 4), ('TKtvkgst8r', 5), ('TKie8k1h6r', 6)):
        for task in task_list['tasks']:
            if task['id'] == task_id: break
        assert task['id'] == task_id
        for subtask in task['subtasks']:
            subtask_id = subtask['id']
            record_dict = subtask['record_dict']
            for user_name, record_id in record_dict.items():
                if user_name not in user_set: continue
                record_info.append((task_id, subtask_id, record_id, user_name, subtask['times'], label))
    for task_id, subtask_id, record_id, user_name, times, label in tqdm.tqdm(record_info):
        record_path = fu.get_record_path(task_list_id, task_id, subtask_id, record_id)
        try:
            record = Record(record_path, times)
        except:
            print(f'### Error: {record_path}')
            continue
        clean_mask = record.clean_mask
        acc_data = record.cutted_imu_data['acc']
        gyro_data = record.cutted_imu_data['gyro']
        imu_data = np.concatenate([acc_data, gyro_data], axis=2)[clean_mask,...]
        imu_data = aug.down_sample_by_step(imu_data, axis=1, step=2)
        imu_data = np.reshape(imu_data, (imu_data.shape[0], -1))
        data.append(imu_data)
        labels = np.concatenate([labels, np.zeros(np.sum(clean_mask), dtype=np.int32) + label])
        
    # load Raise and Drop
    print(f'### Load Raise and Drop.')
    record_info = []
    task_id = 'TK7t3ql6jb'
    for task in task_list['tasks']:
        if task['id'] == task_id: break
    assert task['id'] == task_id
    for subtask in task['subtasks']:
        subtask_id = subtask['id']
        record_dict = subtask['record_dict']
        for user_name, record_id in record_dict.items():
            if user_name not in user_set: continue
            record_info.append((subtask_id, record_id, user_name, subtask['times']))
    for subtask_id, record_id, user_name, times in tqdm.tqdm(record_info):
        record_path = fu.get_record_path(task_list_id, task_id, subtask_id, record_id)
        try:
            record = Record(record_path, times)
        except:
            print(f'### Error: {record_path}')
            continue
        clean_mask = record.clean_mask
        acc_data = record.cutted_imu_data['acc']
        gyro_data = record.cutted_imu_data['gyro']
        imu_data = np.concatenate([acc_data, gyro_data], axis=2)[clean_mask,...]
        imu_data = aug.down_sample_by_step(imu_data, axis=1, step=2)
        imu_data = np.reshape(imu_data, (imu_data.shape[0], -1))
        data.append(imu_data)
        new_labels = np.zeros(times, dtype=np.int32) + 1
        new_labels[1::2] = 2
        labels = np.concatenate([labels, new_labels[clean_mask]])
    
    # load negative data
    print(f'### Load Negative.')
    batch = 100
    W = int(cf.FS_PREPROCESS * cf.CUT_DURATION)
    for i in tqdm.trange(int(np.ceil(len(negative_paths)/batch))):
        negative_data = []
        for path in negative_paths[i*batch:(i+1)*batch]:
            item = pickle.load(open(path, 'rb'))
            negative_data.append(item[None,:W,:])
        negative_data = np.concatenate(negative_data, axis=0)
        negative_data = aug.down_sample_by_step(negative_data, axis=1, step=2)
        negative_data = np.reshape(negative_data, (negative_data.shape[0], -1))
        data.append(negative_data)
        labels = np.concatenate([labels, np.zeros(negative_data.shape[0], dtype=np.int32)])
    data = np.concatenate(data, axis=0)
    
    # plot t-SNE graph
    color_map = {0:'black', 1:'red', 2:'orange', 3:'green', 4:'blue', 5:'purple', 6:'pink'}
    colors = [color_map[i] for i in labels]
    tic = time.perf_counter()
    tsne = TSNE()
    embedded = tsne.fit_transform(data)
    toc = time.perf_counter()
    print(f'### t-SNE time cost: {toc-tic:.3f} s')
    plt.scatter(embedded[:,0], embedded[:,1], c=colors, s=1)
    plt.show()
    
    
def visualize_data_distribution():
    task_list_id = f'TLnmdi15b8'
    task_id, n_sample = f'TKtvkgst8r', 20
    paths = glob(f'{fu.DATA_RECORD_ROOT}/{task_list_id}/{task_id}/ST*/RD*')
    for path in tqdm.tqdm(paths):
        record = Record(path, n_sample)
        gyro = record.cutted_imu_data['gyro']
        for i in range(gyro.shape[0]):
            for j in range(3):
                plt.plot(gyro[i,:,j])
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
    task_list = fu.load_task_list_with_users('TL3wni1oq3')
    task_names = ['Move10cm', 'Move20cm', 'Move30cm', 'Move40cm', 'Move50cm']
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
                distance[task_name].append(dis)
    for task_name in task_names:
        dis_arr = distance[task_name]
        print(f'{task_name}: M = {np.mean(dis_arr):.3f}, SD = {np.std(dis_arr,ddof=1):.3f}')
    plt.violinplot([distance[task_name] for task_name in task_names], positions=range(5))
    plt.xticks(ticks=range(5), labels=task_names, fontsize=14)
    plt.grid(axis='both', linestyle='--')
    plt.ylabel('Distance (m)', fontsize=14)
    plt.title('Moving distance distributions', fontsize=16)
    plt.show()
    
        
if __name__ == '__main__':
    np.random.seed(0)
    fu.check_cwd()
    task_list_id = 'TLm5wv3uex'
    task_id = 'TKhydju8hc'
    subtask_id = 'STb7yi4gsq'
    record_id = 'RDebi44mtx'
    record_path = fu.get_record_path(task_list_id, task_id, subtask_id, record_id)
    tic = time.perf_counter()
    record = Record(record_path, n_sample=20)
    toc = time.perf_counter()
    print(f'time: {(toc-tic)*1000:.3f} ms')
    
    visualize_track_to_imu(record)
    
        
