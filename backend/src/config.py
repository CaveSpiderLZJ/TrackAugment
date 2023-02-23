from typing import Dict

# global
WORKING_DIR = 'TrackAugment/backend/src'

# data preprocessing
N_SAMPLE: int = 21              # action sample times in each record
WINDOW_DURATION: float = 3.0    # action duration after cutting for training (s)
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

# train
RAND_SEED = 0
FS_TRAIN = 100
MODEL_ROOT = '../data/model'
LOG_ROOT = '../data/log'
MODEL_NAME = 'debug'
N_CLASSES = 4
CLASS_NAMES = ('Shake', 'DoubleShake', 'Flip', 'DoubleFlip')
N_EPOCHS = 500
LEARNING_RATE = 1e-5
BATCH_SIZE = 10
WARMUP_STEPS = 10
LOG_STEPS = 1
EVAL_STEPS = 10
DEVICE = 'cpu'
