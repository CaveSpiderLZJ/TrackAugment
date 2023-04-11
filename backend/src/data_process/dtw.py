import numpy as np
from scipy import stats
from dtaidistance import dtw
from matplotlib import pyplot as plt
from typing import Tuple


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
    if axis != 0:
        indices = list(range(x.ndim))
        indices[0], indices[axis] = axis, 0
        x = x.transpose(*indices)
        y = y.transpose(*indices)
    return np.array(dtw.warping_path(stats.zscore(x, axis=None),
        stats.zscore(y, axis=None), use_c=True, use_ndim=True, window=window), dtype=np.int32)
    

if __name__ == '__main__':
    np.random.seed(0)
    STD = 0.1
    N = 50
    cos_peak = np.zeros(N)
    cos_peak[10:30] = np.cos(np.linspace(-np.pi, np.pi, num=20)) + 1
    x = STD * np.random.randn(N, 3, 3)
    x += cos_peak[:,None,None]
    cos_peak = np.zeros(N)
    cos_peak[20:40] = np.cos(np.linspace(-np.pi, np.pi, num=20)) + 1
    y = STD * np.random.randn(N, 3, 3)
    y += cos_peak[:,None,None]
    match_pairs = dtw_match(x, y, axis=0)
    print(match_pairs)
    
    
    