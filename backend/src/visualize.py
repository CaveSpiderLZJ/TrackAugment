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
    S = 100
    x_axis = marker_pos[1,:,:] - marker_pos[0,:,:]
    x_axis /= np.sqrt(np.sum(np.square(x_axis), axis=1)).clip(1e-8)[:,np.newaxis]
    y_axis = marker_pos[4,:,:] - marker_pos[1,:,:]
    y_axis -= x_axis * np.sum(y_axis * x_axis, axis=1)[:,np.newaxis]
    y_axis /= np.sqrt(np.sum(np.square(y_axis), axis=1)).clip(1e-8)[:,np.newaxis]
    z_axis = np.cross(x_axis, y_axis)
    axes = S * np.concatenate([x_axis[None,:,:], y_axis[None,:,:], z_axis[None,:,:]], axis=0)
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
    # plot motion data
    plt.subplot(2, 1, 1)
    start, end = 12500, 15000
    # start, end = 0, -1
    motion_data = record.motion_data
    values = motion_data['acc']
    for axis in ('x', 'y', 'z'):
        plt.plot(values[axis][start:end])
    plt.legend(['X', 'Y', 'Z'])
    # plot track data
    plt.subplot(2, 1, 2)
    start, end = 6935, 7935
    # start, end = 0, -1
    track_data = record.track_data
    phone_pos = track_data['phone_pos']
    for i in range(3):
        plt.plot(phone_pos[start:end,i])
    plt.legend(['X', 'Y', 'Z'])
    plt.show()
    
    
if __name__ == '__main__':
    task_list_id = 'TL13r912je'
    task_id = 'TKfvdarv6k'
    subtask_id = 'ST6klid59e'
    record_id = 'RDmb2zdzis'
    record_path = fu.get_record_path(task_list_id, task_id, subtask_id, record_id)
    record = Record(record_path)
    visualize_record(record)
    
    
