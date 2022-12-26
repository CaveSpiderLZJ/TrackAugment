import numpy as np
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt

from data_process.filter import Butterworth


# Warning: fs is hardcoded as the samping frequency of track data.
butter_acc = Butterworth(fs=200.0, cut=20.0, mode='lowpass', order=1)
butter_gyro = Butterworth(fs=200.0, cut=40.0, mode='lowpass', order=1)


def calc_local_axes(marker_pos:np.ndarray) -> np.ndarray:
    ''' Calculate local axes from marker position tracking data.
    args;
        marker_pos: np.ndarray[(N_MARKER=6,N,3), np.float32],
            the marker position in global frame.
    returns:
        np.ndarray[(3,N,3), np.float32], the local x, y, z axis unit direction vecotors
            in the global frame. axes[0,:,:] represents the x axis direction and so on.
    '''
    x_axis = marker_pos[1,:,:] - marker_pos[0,:,:]
    x_axis /= np.sqrt(np.sum(np.square(x_axis), axis=1)).clip(1e-8)[:,None]
    y_axis = marker_pos[4,:,:] - marker_pos[1,:,:]
    y_axis -= x_axis * np.sum(y_axis * x_axis, axis=1)[:,None]
    y_axis /= np.sqrt(np.sum(np.square(y_axis), axis=1)).clip(1e-8)[:,None]
    z_axis = np.cross(x_axis, y_axis)
    return np.concatenate([x_axis[None,:,:], y_axis[None,:,:], z_axis[None,:,:]], axis=0)


def track_to_acc(pos:np.ndarray, axes:np.ndarray, sr:float) -> np.ndarray:
    ''' Convert global tracking position to local accelerometer data.
    args:
        pos: np.ndarray[(N,3), np.float32], the x, y, z global position of
            the tracking center, unit = (m / s^2).
        axes: np.ndarray[(3,N,3), np.float32], the local x, y, z axis unit direction
            vecotors in the global frame. axes[0,:,:] represents the x axis direction and so on.
        sr: float, the sampling rate, unit = (1 / s).
    returns:
        np.ndarray[(N,3), np.float32], the accelerometer data in the local frame.
    '''
    acc = (np.concatenate([pos[:1,:],pos[:1,:],pos[:-2,:]]) - 8*np.concatenate([pos[:1,:],pos[:-1,:]])
        + 8*np.concatenate([pos[1:,:],pos[-1:,:]]) - np.concatenate([pos[2:,:],pos[-1:,:],pos[-1:,:]]))
    acc = (sr/12) * np.sum(acc[None,:,:] * axes, axis=2).transpose()
    acc = (np.concatenate([acc[:1,:],acc[:1,:],acc[:-2,:]]) - 8*np.concatenate([acc[:1,:],acc[:-1,:]])
        + 8*np.concatenate([acc[1:,:],acc[-1:,:]]) - np.concatenate([acc[2:,:],acc[-1:,:],acc[-1:,:]]))
    gravity = np.array([0.0, 9.805, 0.0], dtype=np.float32)
    gravity = np.sum(gravity[None,None,:] * axes, axis=2).transpose()
    return butter_acc.filt((sr/12) * acc + gravity, axis=0)


def track_to_gyro(axes:np.ndarray, sr:float) -> np.ndarray:
    ''' Convert global tracking position to local gyroscope data.
    args:
        axes: np.ndarray[(3,N,3), np.float32], the same as track_to_acc.
        sr: float, the sampling rate, unit = (1 / s).
    returns:
        np.ndarray[(N,3), np.float32], the gyroscope data in the local frame.
    '''
    q = axes.transpose(1,2,0)
    q_inv = np.linalg.inv(q)
    r = np.matmul(np.concatenate([q[1:,:,:], q[-1:,:,:]]),
        np.concatenate([q_inv[:1,:,:], q_inv[:-1,:,:]]))
    gyro = (0.5*sr) * Rotation.from_matrix(r).as_rotvec()
    gyro = np.sum(gyro[None,:,:] * axes, axis=2).transpose()
    return butter_gyro.filt(gyro, axis=0)


def down_sample(data:np.ndarray, axis:int, step:int) -> np.ndarray:
    ''' Down-sample any np.ndarray along a dynamically designated axis with a specfic integer step.
    args:
        data: np.ndarray, with any shape and dtype.
        axis: int, the index of the axis to be down-sampled (i.e. the time-series axis).
        step: int, the sampling step along the axis, step=1 means no sampling.
    returns:
        np.ndarray, of the same shape as data.
    '''
    index = [slice(None)] * data.ndim
    index[axis] = slice(None,None,step)
    return data[tuple(index)]


def align_time_series(data1:np.ndarray, data2:np.ndarray, axis:int, padding:int) -> int:
    ''' Align two similar time series arrays.
    args:
        data1: np.ndarray, with any shape and dtype.
        data2: np.ndarray, with the same shape and dtype as data1.
        axis: int, the index of the axis to be aligned (i.e. the time-series axis).
        padding: int, the padding length at the start and end of the time series.
            Equivalently, the search range along the axis.
    returns:
        int, the offset of data2 compared to data1. Positive values means data2 is
            on the right side of data1 along the axis.
    '''
    N = data1.shape[axis]
    index = [slice(None)] * data1.ndim
    index[axis] = slice(padding, -padding)
    data1 = data1[tuple(index)]
    length = N - 2 * padding
    min_error, offset = 1e30, 0
    for i in range(0, 2*padding+1):
        index[axis] = slice(i, i+length)
        data = data2[tuple(index)]
        error = np.sum(np.square(data1 - data))
        if error < min_error:
            min_error = error
            offset = i
    return offset - padding


def mse_error(data1:np.ndarray, data2:np.ndarray) -> float:
    ''' Calculate the MSE error of two arrays.
    args:
        data1: np.ndarray, with any shape and dtype.
        data2: np.ndarray, with the same shape and dtype as data1.
    return:
        float, the MSE error of the two arrays.
    '''
    return np.sum(np.square(data1-data2)) / np.prod(data1.shape)
