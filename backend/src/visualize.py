import os
import cv2
import tqdm
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.spatial.transform import Rotation

import file_utils as fu
from data_process.record import Record
from data_process import augment as aug


def output_track_video(save_path:str, phone_pos:np.ndarray, phone_rot:np.ndarray,
    marker_pos:np.ndarray, start_frame:int, end_frame:int) -> None:
    ''' Output the tracking video from tracking data.
    args:
        save_path: str, the video save path.
        phone_pos: np.ndarray[(T,3), np.float32].
        phone_rot: np.ndarray[(T,4), np.float32],
            each row is a quanternion in (X, Y, Z, W).
        marker_pos: np.ndarray[(N_MARKER=6,T,3), np.float32].
        start_frame: int, the start frame index.
        end_frame: int, the end frame index.
    '''
    H, W, PAD = 480, 640, 20
    T = phone_pos.shape[0]
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
    # axes = np.identity(n=3, dtype=np.float32)
    # rotation = Rotation.from_quat(phone_rot)
    # axes = S * np.concatenate([rotation.apply(axes[:,i])[np.newaxis,:,:] for i in range(3)], axis=0)
    for i in tqdm.trange(start_frame, end_frame, 4):
        x, y, z = phone_pos[i,:]
        ax = plt.axes(projection='3d')
        ax.scatter(x, y, z, color='black', s=40)
        ax.scatter(marker_pos[:,i,0], marker_pos[:,i,1], marker_pos[:,i,2], color='orange', s=20)
        ax.quiver(x, y, z, axes[0,i,0], axes[0,i,1], axes[0,i,2], color='red')
        ax.quiver(x, y, z, axes[1,i,0], axes[1,i,1], axes[1,i,2], color='green')
        ax.quiver(x, y, z, axes[2,i,0], axes[2,i,1], axes[2,i,2], color='blue')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        buffer, (w, h) = canvas.print_to_buffer()
        img = np.frombuffer(buffer, np.uint8).reshape((h, w, 4))
        img = cv2.resize(img, (W, H))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        video_writer.write(img)
        fig.clear()
    video_writer.release()


def visualize_record(record:Record) -> None:
    # plot imu data
    plt.subplot(3, 1, 1)
    start1, end1 = 2500, 3000
    motion_data = record.motion_data
    sensor = motion_data['gyro']
    sensor = np.column_stack([sensor[axis] for axis in ('x', 'y', 'z')])
    sensor = aug.down_sample(sensor, axis=0, step=5)
    for i in range(3):
        plt.plot(sensor[start1:end1, i])
    plt.legend(['X', 'Y', 'Z'])
    # plot track data
    plt.subplot(3, 1, 2)
    start2, end2 = 6934, 7934
    track_data = record.track_data
    phone_pos = track_data['phone_pos']
    for i in range(3):
        plt.plot(phone_pos[start2:end2,i])
    plt.legend(['X', 'Y', 'Z'])
    # convert track data to imu data
    plt.subplot(3, 1, 3)
    axes = aug.calc_local_axes(track_data['marker_pos'])
    # generated = aug.track_to_acc(1e-3*phone_pos[start2:end2], axes[:,start2:end2,:], 200.0)
    generated = aug.track_to_gyro(axes[:,start2:end2,:], 200.0)
    generated = aug.down_sample(generated, axis=0, step=2)
    offset = aug.align_time_series(sensor[start1:end1,:], generated, axis=0, padding=20)
    print(f'MSE error: {aug.mse_error(sensor[start1-offset:end1-offset,:], generated):.6f}')
    for i in range(3):
        plt.plot(generated[:,i])
    plt.legend(['X', 'Y', 'Z'])
    plt.show()
    
    
def visualize_markers(record:Record) -> None:
    ''' Visualize the marker positions.
    '''
    track_data = record.track_data
    phone_pos = track_data['phone_pos']
    marker_pos = track_data['marker_pos']
    axes = aug.calc_local_axes(marker_pos)
    phone_pos -= marker_pos[0,:,:]
    phone_pos = np.sum(phone_pos[None,:,:] * axes, axis=2).transpose()
    phone_pos = np.mean(phone_pos, axis=0)
    marker_pos -= marker_pos[0:1,:,:]
    for i in range(marker_pos.shape[0]):
        marker_pos[i,:,:] = np.sum((marker_pos[i:i+1,:,:]) * axes, axis=2).transpose()
    marker_pos = np.mean(marker_pos, axis=1)
    np.set_printoptions(formatter={'float': ' {:0.1f} '.format})
    print(f'### phone_pos (mm):')
    print(phone_pos[:2])
    print(f'### marker_pos (mm):')
    print(marker_pos[:,:2])
    plt.scatter([phone_pos[0]], [phone_pos[1]])
    plt.scatter(marker_pos[:,0], marker_pos[:,1])
    plt.gca().set_aspect(1)
    plt.title(f'Marker Positions')
    plt.xlabel(f'x (mm)')
    plt.ylabel(f'y (mm)')
    plt.show()
    
    
if __name__ == '__main__':
    task_list_id = 'TL13r912je'
    task_id = 'TKfvdarv6k'
    subtask_id = 'ST6klid59e'
    record_id = 'RDmb2zdzis'
    record_path = fu.get_record_path(task_list_id, task_id, subtask_id, record_id)
    record = Record(record_path)
    # visualize_record(record)
    visualize_markers(record)
