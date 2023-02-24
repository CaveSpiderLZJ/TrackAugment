import os
import json
from flask import Flask, request, send_file
from __main__ import app


import file_utils as fu


'''
Name: get_root_list
Method: Get
Form:
    - timestamp

Respone:
    - root_list
'''
@app.route('/root_list', methods=['GET'])
def get_root_list():
    ''' Get the root list root_list.json file under "DATA_RECORD_ROOT/"
    '''
    timestamp = request.args.get('timestamp')
    print(f'Get root list, timestamp: {timestamp}')
    return fu.load_root_list_info()


'''
Name: update_root_list
Method: Post
Content-Type: multipart/form-data
Form:
    - root_list
    - timestamp
'''
@app.route('/root_list', methods=['POST'])
def update_root_list():
    timestamp = int(request.form.get("timestamp"))
    print(f'Update root list, timestamp: {timestamp}')
    root_list = json.loads(request.form.get('root_list'))
    root_list_info_path = fu.get_root_list_info_path()
    fu.save_json(root_list, root_list_info_path)
    for task_list in root_list:
        task_list_id = task_list['id']
        task_list_path = fu.get_task_list_path(task_list_id)
        task_list_info_path = fu.get_task_list_info_path(task_list_id)
        fu.mkdir(task_list_path)
        fu.save_json(task_list, task_list_info_path)
        for task in task_list['tasks']:
            task_id = task['id']
            task_path = fu.get_task_path(task_list_id, task_id)
            task_info_path = fu.get_task_info_path(task_list_id, task_id)
            fu.mkdir(task_path)
            fu.save_json(task, task_info_path)
            for subtask in task['subtasks']:
                subtask_id = subtask['id']
                subtask_path = fu.get_subtask_path(task_list_id, task_id, subtask_id)
                subtask_info_path = fu.get_subtask_info_path(task_list_id, task_id, subtask_id)
                fu.mkdir(subtask_path)
                fu.save_json(subtask, subtask_info_path)
    return ''

