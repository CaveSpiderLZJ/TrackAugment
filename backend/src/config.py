import torch
import numpy as np
from typing import Dict

# global
WORKING_DIR = 'TrackAugment/backend/src'

# data preprocessing
N_SAMPLE: int = 21              # action sample times in each record
WINDOW_DURATION: float = 2.0    # action duration after cutting for training (s)
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

# train
RAND_SEED = 0
FS_TRAIN = 100
MODEL_ROOT = '../data/model'
LOG_ROOT = '../data/log'
MODEL_NAME = 'model1'
N_CLASSES = 7
CLASS_NAMES = ('Negative', 'Raise', 'Drop', 'Shake', 'DoubleShake', 'Flip', 'DoubleFlip')
N_EPOCHS = 300
LEARNING_RATE = 1e-4
BATCH_SIZE = 10
WARMUP_STEPS = 10
LOG_STEPS = 1
EVAL_STEPS = 5
DEVICE = torch.device('cpu')
