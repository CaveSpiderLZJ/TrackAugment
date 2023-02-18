from typing import Dict

# data preprocessing
FS_TRACK: int = 200
FS_IMU: Dict[str, int] = {
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
