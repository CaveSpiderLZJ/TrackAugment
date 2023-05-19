import torch
import numpy as np
from typing import Dict

# global
WORKING_DIR = 'TrackAugment/backend/src'

# data preprocessing
FS_PREPROCESS: int = 200        # the frequency after data resampling (unified for preprocessing)
FS_TRACK: int = 200             # the frequency of the original track data
FS_IMU: Dict[str, int] = {      # the frequencies of the original imu sensors
    'acc': 500,
    'acc_un': 250,
    'gyro': 500,
    'gyro_un': 500,
    'mag': 100,
    'mag_un': 100,
    'linear_acc': 100,
    'gravity': 100,
    'rotation': 100,
}
MARKER_DIS = np.array([[.06775,.09995,.12611,.13021,.15169], [.09111,.11171,.11587,.12611,.14103],
    [.05080,.06775,.08562,.12124,.14103], [.03489,.05080,.07206,.09995,.11587],
    [.03489,.04208,.08562,.11171,.13021], [.04208,.07206,.09111,.12124,.15169]])
USERS = ('lzj', 'lzj2', 'hz', 'xq', 'zyh', 'hyw', 'zxyx', 'crj', 'jjx', 'zz',
    'sxy', 'zyx', 'wzy', 'ms', 'gsq', 'wsy', 'ccr', 'xmy', 'gxy', 'syf',
    'hfx', 'yjy', 'hyh', 'shb', 'zcw', 'xwx', 'gx', 'cx', 'bwy', 'xw',
    'lez', 'll', 'ykh', 'lyl', 'xjy', 'wzn', 'ysc', 'wxy', 'pyh', 'fhy',
    'ljw', 'wyf', 'cyj', 'yzj', 'lap', 'lxt', 'cyf', 'qk', 'mfj', 'lby', 'wzy2')

# data augmentation
CUT_DURATION: float = 2.5       # action duration after cutting (s)
TRAIN_DURATION: float = 2.0     # action duration for training (s)
RAND_SAMPLE_EN: bool = True     # whether to randomly sample action data from cut duration to train duration
RAND_SAMPLE_DURATION: float = 0.2   # the random sample duration range (s), its inverse would be the real-time sample frequency.

# train
RAND_SEED = 0
FS_TRAIN = 100
MODEL_ROOT = '../data/model'
LOG_ROOT = '../data/log'
PLAN_ROOT = '../data/plan'
OUTPUT_ROOT = '../data/output'
PLAN_NAME = '24_RotateTrack_userlist4-9'
N_CLASSES = 5
# CLASS_NAMES = ('Negative', 'Raise', 'Drop', 'Shake', 'DoubleShake', 'Flip', 'DoubleFlip')
# CLASS_NAMES = ('Negative', 'Move10', 'Move20', 'Move30', 'Move40')
CLASS_NAMES = ('Negative', 'Rotate45', 'Rotate90', 'Rotate135', 'Rotate180')
N_TRAIN_USERS = 1   # positive, users
N_TEST_USERS = 1
N_TRAIN_DAYS = 1     # negative, days
N_TEST_DAYS = 1
N_EPOCHS = 200
LEARNING_RATE = 1e-4
SUPER_BATCH = 10    # how many batches a super batch contains
BATCH_SIZE = 64
WARMUP_STEPS = 10
LOG_STEPS = 1
EVAL_STEPS = 5
GC_STEPS = 4
AUGMENT_STEPS = 2
DEVICE = torch.device('cuda')
