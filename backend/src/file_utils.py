import os
import json
import shutil
import hashlib

import config as cf


DEFAULT_TASK_LIST_ID = "TL13r912je"
DEFAULT_ROOT = os.path.join("..", "assets", "default")
DATA_ROOT = os.path.join("..", "data")
DATA_RECORD_ROOT = os.path.join(DATA_ROOT, "record")
DATA_TRAIN_ROOT = os.path.join(DATA_ROOT, "train")
DATA_FILE_ROOT = os.path.join(DATA_ROOT, "file")
DATA_DEX_ROOT = os.path.join(DATA_ROOT, "dex")
DATA_TEMP_ROOT  = os.path.join(DATA_ROOT, "temp")

md5 = dict()

def set_data_root(root):
    global DATA_ROOT, DATA_RECORD_ROOT, DATA_TRAIN_ROOT
    global DATA_FILE_ROOT, DATA_DEX_ROOT, DATA_TEMP_ROOT
    DATA_ROOT = root
    DATA_RECORD_ROOT = os.path.join(DATA_ROOT, "record")
    DATA_TRAIN_ROOT = os.path.join(DATA_ROOT, "train")
    DATA_FILE_ROOT = os.path.join(DATA_ROOT, "file")
    DATA_DEX_ROOT = os.path.join(DATA_ROOT, "dex")
    DATA_TEMP_ROOT  = os.path.join(DATA_ROOT, "temp")

# a series functions to get some file path
def get_temp_path():
    return DATA_TEMP_ROOT

def get_dex_name_path(name):
    return os.path.join(DATA_DEX_ROOT, name)

def get_dex_user_path(user_id, name):
    return os.path.join(get_dex_name_path(name), user_id)

def get_dex_path(user_id, name, timestamp):
    return os.path.join(get_dex_user_path(user_id, name), timestamp)

def get_task_list_path(task_list_id):
    return os.path.join(DATA_RECORD_ROOT, task_list_id)

def get_task_path(task_list_id, task_id):
    return os.path.join(get_task_list_path(task_list_id), task_id)

def get_subtask_path(task_list_id, task_id, subtask_id):
    return os.path.join(get_task_path(task_list_id, task_id), subtask_id)

def get_record_list_path(task_list_id, task_id, subtask_id, dataset_version='0.2'):
    if dataset_version == '0.1':
        return os.path.join(get_subtask_path(task_list_id, task_id, subtask_id), 'recordlist.txt')
    return os.path.join(get_subtask_path(task_list_id, task_id, subtask_id), 'recordlist.json')

def get_record_path(task_list_id, task_id, subtask_id, record_id):
    return os.path.join(get_subtask_path(task_list_id, task_id, subtask_id), record_id)

def get_task_list_info_path(task_list_id, timestamp = None):
    if timestamp is None or str(timestamp) == "0":
        return os.path.join(get_task_list_path(task_list_id), task_list_id + ".json")
    return os.path.join(get_task_list_path(task_list_id), task_list_id + "_" + str(timestamp) + ".json")

def get_task_info_path(task_list_id, taskid):
    return os.path.join(get_task_path(task_list_id, taskid), taskid + ".json")

def get_subtask_info_path(task_list_id, taskid, subtask_id):
    return os.path.join(get_subtask_path(task_list_id, taskid, subtask_id), subtask_id + ".json")

def get_train_path(train_id):
    return os.path.join(DATA_TRAIN_ROOT, train_id)

def get_train_info_path(train_id):
    return os.path.join(get_train_path(train_id), train_id + '.json')

def delete_dir(path):
    try:
        shutil.rmtree(path)
    except:
        pass

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_json(obj, path):
    with open(path, 'w') as fout:
        json.dump(obj, fout, indent=4)

def load_json(path):
    with open(path, 'r') as fin:
        return json.load(fin)

def load_task_list_info(task_list_id, timestamp = None):
    task_list_info_path = get_task_list_info_path(task_list_id, timestamp)
    if not os.path.exists(task_list_info_path):
        print(f'task list info path: {task_list_info_path}')
        task_list_info = {
            'date': '2022.07.03',
            'description': 'Task list for pilot study.',
            'id': task_list_id,
            'tasks': []
        }
        save_json(task_list_info, task_list_info_path)
        return task_list_info
    with open(task_list_info_path, 'r') as f:
        data = json.load(f)
        return data

def load_record_list(task_list_id, task_id, subtask_id, dataset_version):
    record_list_path = get_record_list_path(task_list_id, task_id, subtask_id, dataset_version)
    if not os.path.exists(record_list_path):
        return []

    record_list = []
    if dataset_version == '0.1':
        with open(record_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                record_id = line.strip()
                if record_id.startswith('RD') and record_id not in record_list:
                    record_list.append(record_id)
    else:
        with open(record_list_path, 'r') as fin:
            records = json.load(fin)
            for record in records:
                record_list.append(record['record_id'])
        
    return record_list

def append_record_list(task_list_id, task_id, subtask_id, user_name, record_id, dataset_version='0.2'):
    record_list_path = get_record_list_path(task_list_id, task_id, subtask_id, dataset_version)
    if not os.path.exists(record_list_path):
        save_json([], record_list_path)
    record_list:list = json.load(open(record_list_path, 'r'))
    record_list.append({'user_name': user_name, 'record_id': record_id})
    save_json(record_list, record_list_path)  

def delete_record_list(task_list_id, task_id, subtask_id, record_id, dataset_version='0.2'):
    record_list_path = get_record_list_path(task_list_id, task_id, subtask_id, dataset_version)
    if not os.path.exists(record_list_path): return
    record_list:list = json.load(open(record_list_path, 'r'))
    new_record_list = []
    for record in record_list:
        if record['record_id'] != record_id:
            new_record_list.append(record)
    save_json(new_record_list, record_list_path)

def allowed_file(filename):
    return os.path.splitext(filename)[-1] in ['.json', '.mp4', '.bin', '.csv', '.param', '.dex', '.jar']

def save_record_file(file, file_path):
    f = open(file_path, 'wb')
    file.save(f)
    f.close()
    '''
    try:
        record_path = "/".join(file_path.split('/')[:-1])
        file_suffix = "_".join((file_path.split('/')[-1]).split('_')[1:])
        if file_suffix.endswith('json'):
            sensor_path = os.path.join(record_path, "Sensor_" + file_suffix)
            timestamp_path = os.path.join(record_path, "Timestamp_" + file_suffix)
            if os.path.exists(sensor_path) and os.path.exists(timestamp_path):
                Record(sensor_path, timestamp_path, do_cut=False)

    except:
        pass
    '''

def save_file(file, file_path):
    file.save(file_path)

def calc_file_md5(file_name):
    ''' Calc md5 use the file data, and return the md5 digest.
    '''
    m = hashlib.md5()
    with open(file_name, 'rb') as f:
        while True:
            data = f.read(4096)
            if not data:
                break
            m.update(data)
    return m.hexdigest()

def update_md5():
    ''' Update the md5 mapping of all files in DATA_FILE_ROOT.
    '''
    global md5
    if not os.path.exists(DATA_FILE_ROOT):
        print(DATA_FILE_ROOT + 'does not exist.')
        return
    for filename in os.listdir(DATA_FILE_ROOT):
        md5[filename] = calc_file_md5(os.path.join(DATA_FILE_ROOT, filename))

def get_md5(filename):
    ''' Return the file md5 digest.
        If file does not exist, return ""
    '''
    global md5
    if filename in md5:
        return md5[filename]
    return ""

def create_default_files():
    mkdir(DATA_RECORD_ROOT)
    mkdir(DATA_TRAIN_ROOT)
    mkdir(DATA_FILE_ROOT)
    mkdir(DATA_DEX_ROOT)
    mkdir(DATA_TEMP_ROOT)
    shutil.copyfile(os.path.join(DEFAULT_ROOT, "config.json"), os.path.join(DATA_FILE_ROOT, "config.json"))
    default_task_list_src = os.path.join(DEFAULT_ROOT, DEFAULT_TASK_LIST_ID)
    default_task_list_dst = os.path.join(DATA_RECORD_ROOT, DEFAULT_TASK_LIST_ID)
    if os.path.exists(default_task_list_src) and not os.path.exists(default_task_list_dst):
        shutil.copytree(default_task_list_src, default_task_list_dst)
    
    
def check_cwd():
    ''' Check current working directory.
    '''
    cwd = os.getcwd()
    if not cwd.endswith(cf.WORKING_DIR):
        raise Exception(f'Incorrect working directory. Must be: {cf.WORKING_DIR}')
        

if __name__ == '__main__':
    pass
