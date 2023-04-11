import numpy as np
from scipy import stats
from dtaidistance import dtw
from scipy.spatial.transform import Rotation
from scipy import interpolate as interp
from matplotlib import pyplot as plt
from typing import Tuple, List, Dict

import config as cf
from data_process.dtw import dtw_match
from data_process.filter import Butterworth


# Warning: fs is hardcoded as the samping frequency of track data.
butter_acc = Butterworth(fs=cf.FS_PREPROCESS, cut=0.1*cf.FS_PREPROCESS, mode='lowpass', order=1)
butter_gyro = Butterworth(fs=cf.FS_PREPROCESS, cut=0.2*cf.FS_PREPROCESS, mode='lowpass', order=1)


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
    x_length = np.sqrt(np.sum(np.square(x_axis), axis=1))
    x_axis[x_length < 1e-8, :] = [1e-8, 0, 0]
    x_axis /= np.sqrt(np.sum(np.square(x_axis), axis=1))[:,None]
    y_axis = marker_pos[2,:,:] - marker_pos[0,:,:]
    y_axis -= x_axis * np.sum(y_axis * x_axis, axis=1)[:,None]
    y_length = np.sqrt(np.sum(np.square(y_axis), axis=1))
    y_axis[y_length < 1e-8, :] = [0, 1e-8, 0]
    y_axis /= np.sqrt(np.sum(np.square(y_axis), axis=1))[:,None]
    z_axis = np.cross(x_axis, y_axis)
    return np.concatenate([x_axis[None,:,:], y_axis[None,:,:], z_axis[None,:,:]], axis=0)


def track_to_acc(pos:np.ndarray, axes:np.ndarray, fs:float) -> np.ndarray:
    ''' Convert global tracking position to local accelerometer data.
    args:
        pos: np.ndarray[(N,3), np.float32], the x, y, z global position of
            the tracking center, unit = (m / s^2).
        axes: np.ndarray[(3,N,3), np.float32], the local x, y, z axis unit direction
            vecotors in the global frame. axes[0,:,:] represents the x axis direction and so on.
        fs: float, the sampling frequency, unit = (1 / s).
    returns:
        np.ndarray[(N,3), np.float32], the accelerometer data in the local frame.
    '''
    acc = (fs/12) * (np.concatenate([pos[:1,:],pos[:1,:],pos[:-2,:]]) - 8*np.concatenate([pos[:1,:],pos[:-1,:]])
        + 8*np.concatenate([pos[1:,:],pos[-1:,:]]) - np.concatenate([pos[2:,:],pos[-1:,:],pos[-1:,:]]))
    acc = (fs/12) * (np.concatenate([acc[:1,:],acc[:1,:],acc[:-2,:]]) - 8*np.concatenate([acc[:1,:],acc[:-1,:]])
        + 8*np.concatenate([acc[1:,:],acc[-1:,:]]) - np.concatenate([acc[2:,:],acc[-1:,:],acc[-1:,:]]))
    acc = np.sum(acc[None,:,:] * axes, axis=2).T
    gravity = np.array([0.0, 9.805, 0.0], dtype=np.float32)
    gravity = np.sum(gravity[None,None,:] * axes, axis=2).T
    return butter_acc.filt(acc + gravity, axis=0)


def track_to_gyro(axes:np.ndarray, fs:float) -> np.ndarray:
    ''' Convert global tracking position to local gyroscope data.
    args:
        axes: np.ndarray[(3,N,3), np.float32], the same as track_to_acc.
        fs: float, the sampling freqency, unit = (1 / s).
    returns:
        np.ndarray[(N,3), np.float32], the gyroscope data in the local frame.
    '''
    q = axes.transpose(1,2,0)
    q_inv = np.linalg.inv(q)
    r = np.matmul(np.concatenate([q[1:,:,:], q[-1:,:,:]]),
        np.concatenate([q_inv[:1,:,:], q_inv[:-1,:,:]]))
    gyro = (0.5*fs) * Rotation.from_matrix(r).as_rotvec()
    gyro = np.sum(gyro[None,:,:] * axes, axis=2).T
    return butter_gyro.filt(gyro, axis=0)


def imu_to_track(acc:np.ndarray, gyro:np.ndarray, bound_pos:np.ndarray,
        bound_velocity:np.ndarray, bound_axes:np.ndarray, fs:float) -> Tuple[np.ndarray, np.ndarray]:
    ''' Given the initial motion status, convert 6-axis imu data to track data.
    args:
        acc: np.ndarray[(N,3), np.float32], the acc data in the local frame.
        gyro: np.ndarray[(N,3), np.float32], the gyro data in the local frame.
        bound_pos: np.ndarray[(2,3), np.float32], the initial and ending
            position in the global frame.
        bound_velocity: np.ndarray[(2,3), np.float32], the initial and ending
            velocity in the global frame.
        bound_axes: np.ndarray[(3,4,3), np.float32], the initial and ending
            x, y, z axis unit direction vectors in the global frame.
        fs: float, the sampling frequency in Hz.
    returns:
        Tuple[np.ndarray[(N,3), np.float32], np.ndarray[(3,N,3), np.float32]],
            the converted global position and axes directions.
    '''
    N = acc.shape[0]
    weight1 = np.linspace(1.0, 0.0, num=N)
    weight2 = 1.0 - weight1
    # calculate axes
    axes1, axes2 = np.empty((3,N,3), dtype=np.float32), np.empty((3,N,3), dtype=np.float32)
    axes1[:,:2,:], axes2[:,-2:,:] = bound_axes[:,:2,:], bound_axes[:,-2:,:]
    for t in range(N-2):
        rot1 = (2/fs) * np.sum(gyro[t+1,:,None] * axes1[:,t+1,:], axis=0)
        rot2 = (2/fs) * np.sum(gyro[N-2-t,:,None] * axes2[:,N-2-t,:], axis=0)
        rot1, rot2 = Rotation.from_rotvec(rot1), Rotation.from_rotvec(rot2)
        axes1[:,t+2,:] = np.matmul(rot1.as_matrix(), axes1[:,t,:].T).T
        axes2[:,N-3-t,:] = np.matmul(np.linalg.inv(rot2.as_matrix()), axes2[:,N-1-t,:].T).T
    axes = axes1 * weight1[None,:,None] + axes2 * weight2[None,:,None]
    # calculate global acc
    gravity = np.array([0.0, 9.805, 0.0], dtype=np.float32)
    a = np.sum(acc.T[:,:,None] * axes, axis=0) - gravity[None,:]
    # integrate to velocity
    v1, v2 = np.empty((N,3), dtype=np.float32), np.empty((N,3), dtype=np.float32)
    v1[:1,:] = bound_velocity[:1,:]
    v1[1:,:] = bound_velocity[:1,:] + (1/(2*fs))*np.cumsum((a[:-1,:] + a[1:,:]), axis=0)
    v2[-1:,:] = bound_velocity[-1:,:]
    v2[:-1,:] = bound_velocity[-1:,:] - (1/(2*fs))*np.cumsum((a[-2::-1,:] + a[:0:-1,:]), axis=0)[::-1,:]
    v = v1 * weight1[:,None] + v2 * weight2[:,None]
    # integrate to position
    p1, p2 = np.empty((N,3), dtype=np.float32), np.empty((N,3), dtype=np.float32)
    p1[:1,:] = bound_pos[:1,:]
    p1[1:,:] = bound_pos[:1,:] + (1/(2*fs))*np.cumsum((v[:-1,:] + v[1:,:]), axis=0)
    p2[-1:,:] = bound_pos[-1:,:]
    p2[:-1,:] = bound_pos[-1:,:] - (1/(2*fs))*np.cumsum((v[-2::-1,:] + v[:0:-1,:]), axis=0)[::-1,:]
    pos = p1 * weight1[:,None] + p2 * weight2[:,None]
    return pos, axes


def resample(data:np.ndarray, axis:int, ratio:float, kind='quadratic') -> np.ndarray:
    ''' Resample data by spline interpolation.
    args:
        data: np.ndarray, with any shape and dtype.
        axis: int, the samping axis.
        ratio: float, the number of resampled data points over
            the number of original data points.
        kind: str, the interpolation time.
    returns:
        np.ndarray, the resampled data, dtype = the original dtype.
    '''
    N = data.shape[axis]
    M = int(np.round(N * ratio))
    t = np.arange(N, dtype=np.float64)
    x = np.linspace(0, N-1, num=M, endpoint=True)
    interp_func = interp.interp1d(t, data, kind=kind, axis=axis)
    return interp_func(x)


def down_sample_by_step(data:np.ndarray, axis:int, step:int) -> np.ndarray:
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


def jitter(data:np.ndarray, std:float) -> np.ndarray:
    ''' Add random Gaussian noise to the data.
    args:
        data: np.ndarray, with any shape and dtype.
        std: float, the standard deviation of the Gaussian noise.
    returns:
        np.ndarray, with the same shape as data.
    '''
    return data + np.random.randn(*data.shape) * std


def rotate(data:np.ndarray, matrix:np.ndarray) -> np.ndarray:
    ''' Rotate the signals by the roration matrix.
    args:
        data: np.ndarray[(...,N,M)], with any dtype.
        matrix: np.ndarray[(M,M)], with any dtype.
    '''
    ndim = data.ndim
    schema = np.arange(ndim)
    schema[-2:] = ndim-1, ndim-2
    return np.matmul(matrix, data.transpose(schema)).transpose(schema)


def zoom(data:np.ndarray, axis:int, low:float=0.9, high:float=1.0):
    ''' Zoom data in time axis, by a random factor s in [low, high].
    args:
        data: np.ndarray, of any shape, data to be augmented.
        axis: int, the index of the time series axis.
        low: float, the lower bound of the range of s, default = 0.9.
        high: float, the higher bound of the range of s, default = 1.0.
    '''
    N = data.shape[axis]
    s = np.random.rand() * (high-low) + low
    start = 0.5 * (1-s) * (N-1)
    end = start + s * (N-1)
    t = np.linspace(start, end, num=N)
    f = interp.interp1d(np.arange(N), data, kind='quadratic',
        axis=axis, fill_value=0, bounds_error=False)
    return f(t)


def scale(data:np.ndarray, std:float=0.2, low:float=0.0, high:float=2.0) -> np.ndarray:
    ''' Scale the data by a random factor s ~ N(1, std^2).
    args:
        data: np.ndarray, with any shape and dtype.
        std: float, the std deviation of the scaling factor.
        low and high: float, the lower and higher bounds of the random factor.
    '''
    s = np.clip(1.0 + np.random.randn() * std, a_min=low, a_max=high)
    return data * s


def window_slice(data:np.ndarray, axis:int, start:int, window_length:int) -> np.ndarray:
    ''' Slice a window in the data.
    args:
        data: np.ndarray, length in time axis must >= window_length.
        axis: int, the index of the time axis.
        start: int, the slice start index.
        window_length: int, the sliced window length.
    returns:
        np.ndarray, length in the time axis == window_length.
    '''
    slices = [slice(None)] * data.ndim
    slices[axis] = slice(start, start+window_length)
    return data[tuple(slices)]


def magnitude_warp(data:np.ndarray, axis:int, n_knots:int,
        std:float, preserve_bound:bool=False) -> np.ndarray:
    ''' Warp the data magnitude by a smooth curve along the time axis.
    args:
        data: np.ndarray, with any shape and dtype.
        axis: int, the index of time axis.
        n_knots: int, the number of random knots on the random curve.
        std: float, the standard deviations of knots.
        preserve_bound: bool, whether to preserve data magnitude at the start
            and end of the time series. Default is False.
    returns:
        np.ndarray, with the same shape as data.
    '''
    N = data.shape[axis]
    x_knots = np.arange(n_knots) * N / (n_knots-1)
    y_knots = 1.0 + np.random.randn(n_knots) * std
    if preserve_bound:
        y_knots[0], y_knots[-1] = 1.0, 1.0
    tck = interp.splrep(x_knots, y_knots, s=0, per=False)
    ys = interp.splev(np.arange(N), tck, der=0)
    slices = [None] * data.ndim
    slices[axis] = slice(None)
    return data * ys[tuple(slices)]


def time_warp(data:np.ndarray, axis:int, n_knots:int=4, std:float=0.05,
        low:float=0.0, high:float=2.0) -> np.ndarray:
    ''' Warp the timestamps by a smooth curve. Unlike magnitude warping,
        time warping always preserves the timestamps at the start and end.
    args:
        data: np.ndarray, with any shape and dtype.
        axis: int, the index of time axis.
        n_knots: int, the number of random knots on the random curve.
            To be unified with magnitude warping, n_knots includes the start
            and end timestamps, though they are not warped, which means,
            if n_knots == 2, nothing will happen.
        std: float, the standard deviations of knots.
        low and high: the lower and higher bounds of the random factors.
    '''
    N = data.shape[axis]
    x_knots = np.linspace(0, 1, num=n_knots, endpoint=True, dtype=np.float32)
    y_knots = x_knots * (1.0 + np.random.randn(n_knots) * std).clip(low, high)
    y_knots[0], y_knots[-1] = 0.0, 1.0
    tck = interp.splrep(x_knots, y_knots, s=0, per=False)
    xs = np.linspace(0, 1, num=N, endpoint=True, dtype=np.float32)
    t = (N-1) * interp.splev(xs, tck, der=0)
    t = t.clip(0, N-1)
    f = interp.interp1d(np.arange(N), data, axis=axis)
    return f(t)


def dtw_match(x:np.ndarray, y:np.ndarray, axis:int=0, window:int=None) -> np.ndarray:
    ''' Match two N-D array along an axis.
        Return the DTW distance and match pairs.
    args:
        x and y: np.ndarray, of any shape, but ndim must be equal,
            and only the length in <axis> axis could be different.
        axis: int, the index of axis to be calculated.
        window: int, the dtw window, default = None.
    returns:
        np.ndarray: the DTW warping paths.
    '''
    N = x.shape[axis]
    if axis != 0:
        indices = list(range(x.ndim))
        indices[0], indices[axis] = axis, 0
        x = x.transpose(*indices)
        y = y.transpose(*indices)
    warping_path = np.array(dtw.warping_path(stats.zscore(x, axis=None),
        stats.zscore(y, axis=None), use_c=True, use_ndim=True, window=window), dtype=np.int32)
    warping_path = down_sample_by_step(warping_path, axis=0, step=32)
    warping_path = resample(warping_path, axis=0, ratio=N/warping_path.shape[0]).clip(0, N-1)
    return warping_path


def dtw_augment(data1:np.ndarray, data2:np.ndarray,
        warping_path:np.ndarray, axis:int, weight:float=0.5) -> np.ndarray:
    ''' Use DTW match to smoothly merge data1 and data2, generating new data sequence.
    args:
        data1 and data2: np.ndarray, with the same shape.
        warping_path: np.ndarray, returned by dtw_match.
        axis: int, indicates the index of the time series.
        weight: float, in (0, 1), the weight of data1.
    returns:
        np.ndarray, the augmented data.
    '''
    N = data1.shape[axis]
    f1 = interp.interp1d(np.arange(N), data1, kind='quadratic', axis=axis)
    f2 = interp.interp1d(np.arange(N), data2, kind='quadratic', axis=axis)
    x, y = f1(warping_path[:,0]), f2(warping_path[:,1])
    return x * weight + y * (1-weight)


def classic_augment(data:np.ndarray, axis:int) -> np.ndarray:
    ''' Combine Zoom, Scale and Time Warp.
    args:  
        data: np.ndarray, of any shape, data to be augmented.
        axis: int, the time series axis.
    '''
    strategies = np.random.randint(1,8)
    if strategies in (1, 3, 5, 7): data = zoom(data, axis=axis)
    if strategies in (2, 3, 6, 7): data = scale(data)
    if strategies in (4, 5, 6, 7): data = time_warp(data, axis=axis)
    return data


if __name__ == '__main__':
    data1 = np.sin(np.linspace(0, 4*np.pi, num=50))
    data2 = classic_augment(data1, axis=0)
    plt.plot(data1)
    plt.plot(data2)
    plt.show()
    
