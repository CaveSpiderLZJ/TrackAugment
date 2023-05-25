import json
import numpy as np

import file_utils as fu


if __name__ == '__main__':
    fu.check_cwd()
    file_path = '../data/plan/36_RotateTrackComb_model5_userlist0-9.json'
    strategies = {
        'scale': {'prob': 0.5, 'std': 0.0001},
        'zoom': {'prob': 0.5, 'low': 0.99},
        'time warp': {'prob': 0.5, 'N': 6, 'std': 0.4},
        'mag warp': {'prob': 0.5, 'N': 8, 'std': 0.005}
    }
    codes = ('N', 'ST', 'ZTM', 'SZT', 'T', 'STM', 'TM', 'SZTM', 'SM')
    class_name = 'rotate'
    method = 'classic_on_track'
    plan_list = dict()
    idx_start = 1837
    for i, user_list_id in enumerate(range(10)):
        for j, code in enumerate(codes):
            idx = idx_start + i*9 + j
            plan_name = f'{idx}_RotateTrackComb_model5_userlist{user_list_id}_{code}_epoch200_lr1e-4'
            plan = {'class_name': class_name, 'method': method, 'user_list_id': user_list_id}
            plan['strategies'] = {}
            for c in code:
                if c == 'S': plan['strategies']['scale'] = strategies['scale']
                elif c == 'Z': plan['strategies']['zoom'] = strategies['zoom']
                elif c == 'T': plan['strategies']['time warp'] = strategies['time warp']
                elif c == 'M': plan['strategies']['mag warp'] = strategies['mag warp']
            plan_list[plan_name] = plan
    json.dump(plan_list, open(file_path, 'w'), indent=4)

            