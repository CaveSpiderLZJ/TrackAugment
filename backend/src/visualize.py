import os
import cv2
import tqdm
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.spatial.transform import Rotation

import file_utils as fu
from data_process.record import Record
from data_process import augment as aug


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


def visualize_record(record:Record) -> None:
    # plot imu data
    plt.subplot(2, 1, 1)
    start1, end1 = 500, 1000
    imu_data = record.imu_data
    sensor = imu_data['acc']
    sensor = np.column_stack([sensor[axis] for axis in ('x', 'y', 'z')])
    sensor = aug.down_sample(sensor, axis=0, step=5)
    for i in range(3):
        plt.plot(sensor[start1:end1, i])
    # plot track data
    plt.subplot(2, 1, 2)
    start2, end2 = 2934, 3934
    track_data = record.track_data
    center_pos = track_data['center_pos']
    for i in range(3):
        plt.plot(center_pos[start2:end2,i])
    plt.legend(['X', 'Y', 'Z'], loc='lower right')
    # plot generated data
    plt.subplot(2, 1, 1)
    axes = aug.calc_local_axes(track_data['marker_pos'])
    generated = aug.track_to_acc(1e-3*center_pos[start2:end2,:], axes[:,start2:end2,:], 200.0)
    # generated = aug.track_to_gyro(axes[:,start2:end2,:], 200.0)
    generated = aug.down_sample(generated, axis=0, step=2)
    offset = aug.align_time_series(sensor[start1:end1,:], generated, axis=0, padding=20)
    print(f'MSE error: {aug.mse_error(sensor[start1-offset:end1-offset,:], generated):.3f}')
    for i in range(3):
        plt.plot(generated[:,i])
    plt.legend(['X', 'Y', 'Z', 'X1', 'Y1', 'Z1'], loc='lower right')
    plt.show()
    
    
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
    plt.title(f'Marker Positions')
    plt.xlabel(f'x (mm)')
    plt.ylabel(f'y (mm)')
    plt.show()
    

def visualize_track(record:Record) -> None:
    '''
    '''
    track_data = record.track_data
    center_pos = track_data['center_pos']
    start, end = 2934, 22934
    pos = center_pos[start:end,:]
    # rotate x, y, z to -x, z, y
    rot_matrix = np.array([[-1,0,0],[0,0,1],[0,1,0]], dtype=np.float32)
    pos = np.matmul(rot_matrix, pos.T).T
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(pos[:,0], pos[:,1], pos[:,2])
    # set equal box aspect
    x_scale = np.abs(np.diff(ax.get_xlim()))[0]
    y_scale = np.abs(np.diff(ax.get_ylim()))[0]
    z_scale = np.abs(np.diff(ax.get_zlim()))[0]
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.zaxis.set_major_locator(MultipleLocator(50))
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
    acc = aug.down_sample(acc, axis=0, step=5)
    gyro = aug.down_sample(gyro, axis=0, step=5)
    center_pos = aug.down_sample(center_pos, axis=0, step=2)
    axes = aug.down_sample(axes, axis=1, step=2)
    bound_pos = 1e-3*np.concatenate([center_pos[:1,:],center_pos[-1:,:]], axis=0)
    bound_velocity = 0.1*np.concatenate([center_pos[1:2,:]-center_pos[0:1,:],
        center_pos[-1:,:]-center_pos[-2:-1,:]], axis=0)
    bound_axes = np.concatenate([axes[:,:2,:],axes[:,-2:,:]], axis=1)
    generated_pos, generated_axes = aug.imu_to_track(acc, gyro, bound_pos,
        bound_velocity, bound_axes, 100.0)
    generated_pos *= 1e3
    mse = aug.mse_error(center_pos, generated_pos)
    print(f'MSE Error: {mse:.6f}.')
    for i in range(3):
        plt.plot(center_pos[:,i])
    for i in range(3):
        plt.plot(generated_pos[:,i])
    plt.show()
    

def visualize_augmentation(record:Record):
    track_data = record.track_data
    start, end = 2934, 3934
    center_pos = track_data['center_pos'][start:end,:]
    # augmented = aug.jitter(center_pos, std=3.0)
    # matrix = np.array([[1,0,0],[0,0,1],[0,1,0]], dtype=np.float32)
    # augmented = aug.rotate(center_pos, matrix) - 5
    # augmented = aug.scale(center_pos, 0.05)
    # augmented = aug.window_slice(center_pos, axis=0, start=0, window_length=800) - 5
    # augmented = aug.magnitude_warp(center_pos, axis=0, n_knots=8, std=0.05, preserve_bound=False)
    augmented = aug.time_warp(center_pos, axis=0, n_knots=4, std=0.02)
    for i in range(3):
        plt.plot(center_pos[:,i])
    for i in range(3):
        plt.plot(augmented[:,i])
    plt.legend(['X','Y','Z','X1','Y1','Z1'], loc='lower right')
    plt.show()
    
    
if __name__ == '__main__':
    task_list_id = 'TL13r912je'
    task_id = 'TKfvdarv6k'
    subtask_id = 'ST6klid59e'
    record_id = 'RDmb2zdzis'
    record_path = fu.get_record_path(task_list_id, task_id, subtask_id, record_id)
    record = Record(record_path)
    # output_track_video(record, 'track.mp4', 1934, 22934)
    # visualize_record(record)
    # visualize_markers(record)
    # visualize_track(record)
    # visualize_imu_to_track(record)
    visualize_augmentation(record)
