import os
import cv2
import time
import tqdm
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
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
    # prepare imu data
    imu_data = record.imu_data
    start_imu, end_imu = 0, 5000
    acc = imu_data['acc']
    acc = np.column_stack([acc[axis] for axis in ('x','y','z')])[start_imu:end_imu,:]
    # acc = aug.down_sample_by_step(acc, axis=0, step=5)
    gyro = imu_data['gyro']
    gyro = np.column_stack([gyro[axis] for axis in ('x','y','z')])[start_imu:end_imu,:]
    # gyro = aug.down_sample_by_step(gyro, axis=0, step=5)
    # prepare track data
    track_data = record.track_data
    start_track, end_track = 0, 5000
    center_pos = track_data['center_pos'][start_track:end_track,:]
    marker_pos = track_data['marker_pos'][:,start_track:end_track,:]
    axes = aug.calc_local_axes(marker_pos)
    generated_acc = aug.track_to_acc(center_pos, axes, cf.FS_PREPROCESS)
    # generated_acc = aug.down_sample_by_step(generated_acc, axis=0, step=2)
    generated_gyro = aug.track_to_gyro(axes, cf.FS_PREPROCESS)
    # generated_gyro = aug.down_sample_by_step(generated_gyro, axis=0, step=2)
    # MSE error
    mse_acc = aug.mse_error(acc, generated_acc)
    mse_gyro = aug.mse_error(gyro, generated_gyro)
    print(f'mse_acc: {mse_acc:.6f}')
    print(f'mse_gyro: {mse_gyro:.6f}')
    # plot acc data
    plt.subplot(2, 1, 1)
    for i in range(3): plt.plot(acc[:,i])
    for i in range(3): plt.plot(generated_acc[:,i])
    plt.ylabel('Accelerometer', fontsize=14)
    plt.legend(['X', 'Y', 'Z', 'X\'', 'Y\'', 'Z\''], loc='lower right')
    # plot gyro data
    plt.subplot(2, 1, 2)
    for i in range(3): plt.plot(gyro[:,i])
    for i in range(3): plt.plot(generated_gyro[:,i])
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
    

def visualize_track(record:Record) -> None:
    '''
    '''
    # prepare imu data
    imu_data = record.imu_data
    start_imu, end_imu = 3000, 3500
    acc = imu_data['acc']
    acc = np.column_stack([acc[axis] for axis in ('x', 'y', 'z')])[start_imu:end_imu,:]
    acc = aug.down_sample_by_step(acc, axis=0, step=5)
    augmented_acc = aug.jitter(acc, std=1)
    # augmented_acc = aug.scale(acc, 0.05)
    augmented_acc = aug.time_warp(acc, axis=0, n_knots=4, std=0.05)
    # augmented_acc = aug.magnitude_warp(acc, axis=0, n_knots=4, std=0.1, preserve_bound=True)
    gyro = imu_data['gyro']
    gyro = np.column_stack([gyro[axis] for axis in ('x','y','z')])[start_imu:end_imu,:]
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
    acc, gyro = imu_data['acc'], imu_data['gyro']
    acc = np.column_stack([acc[axis] for axis in ('x','y','z')])[start_imu:end_imu,:]
    gyro = np.column_stack([gyro[axis] for axis in ('x','y','z')])[start_imu:end_imu,:]
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
    acc = imu_data['acc']
    acc = np.column_stack([acc[axis] for axis in ('x','y','z')])[start_imu:end_imu,:]
    gyro = imu_data['gyro']
    gyro = np.column_stack([gyro[axis] for axis in ('x','y','z')])[start_imu:end_imu,:]
    # prepare track data
    track_data = record.track_data
    start_track, end_track = 2934, 3934
    center_pos = track_data['center_pos'][start_track:end_track,:]
    marker_pos = track_data['marker_pos'][:,start_track:end_track,:]
    axes = aug.calc_local_axes(marker_pos)
    # augment data
    # augmented_pos = aug.jitter(center_pos, std=1e-3)
    # augmented_pos = aug.scale(center_pos, std=0.05)
    augmented_pos = aug.magnitude_warp(center_pos, axis=0, n_knots=8, std=0.1, preserve_bound=True)
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
    acc = imu_data['acc']
    acc = np.column_stack([acc[axis] for axis in ('x','y','z')])[start_imu:end_imu,:]
    gyro = imu_data['gyro']
    gyro = np.column_stack([gyro[axis] for axis in ('x','y','z')])[start_imu:end_imu,:]
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
    # augmented_acc = aug.magnitude_warp(acc, axis=0, n_knots=8, std=0.1, preserve_bound=True)
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
    data = record.cutted_imu_data['acc'][1,...]
    data = aug.down_sample_by_step(data, axis=0, step=2)
    butterworth = Butterworth(fs=100, cut=32, mode='highpass')
    filtered = butterworth.filt(data, axis=0)
    plt.subplot(2, 1, 1)
    for i in range(3):
        plt.plot(data[:,i])
    plt.subplot(2, 1, 2)
    for i in range(3):
        plt.plot(filtered[:,i])
    plt.show()
    

if __name__ == '__main__':
    fu.check_cwd()
    task_list_id = 'TL13r912je'
    task_id = 'TKfvdarv6k'
    # Shake
    # subtask_id = 'ST6klid59e'
    # record_id = 'RDmb2zdzis'
    # DoubleShake
    subtask_id = 'STxw6enkhj'
    record_id = 'RD6fu3gmp6'
    record_path = fu.get_record_path(task_list_id, task_id, subtask_id, record_id)
    tic = time.perf_counter()
    record = Record(record_path)
    toc = time.perf_counter()
    print(f'time: {(toc-tic)*1000:.3f} ms')
     
    # output_track_video(record, 'track.mp4', 1934, 22934)
    # visualize_track_to_imu(record)
    # visualize_markers(record)
    # visualize_track(record)
    # visualize_imu_to_track(record)
    # augment_track(record)
    # augment_imu(record)
    # visualize_cutter(record)
    visualize_filter(record)
