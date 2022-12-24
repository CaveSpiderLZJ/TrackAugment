import numpy as np
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt


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
    x_axis /= np.sqrt(np.sum(np.square(x_axis), axis=1)).clip(1e-8)[:,np.newaxis]
    y_axis = marker_pos[4,:,:] - marker_pos[1,:,:]
    y_axis -= x_axis * np.sum(y_axis * x_axis, axis=1)[:,np.newaxis]
    y_axis /= np.sqrt(np.sum(np.square(y_axis), axis=1)).clip(1e-8)[:,np.newaxis]
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
    dpos = np.concatenate([pos[1:,:],pos[-1:,:]]) - np.concatenate([pos[:1,:],pos[:-1,:]])
    dpos = (0.5*sr) * np.sum(dpos[np.newaxis,:,:] * axes, axis=2).transpose()
    acc = np.concatenate([dpos[1:,:],dpos[-1:,:]]) - np.concatenate([dpos[:1,:],dpos[:-1,:]])
    gravity = np.array([0.0, 9.805, 0.0], dtype=np.float32)
    gravity = np.sum(gravity[None,None,:] * axes, axis=2).transpose()
    return (0.5*sr) * acc + gravity


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
    return np.sum(gyro[np.newaxis,:,:] * axes, axis=2).transpose()
