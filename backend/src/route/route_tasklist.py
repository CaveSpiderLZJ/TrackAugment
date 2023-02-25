from flask import Flask, request, send_file
from __main__ import app

import file_utils
import os
import json

'''
Name: get_all_taskList
Method: Get
Respone:
    - List(tasktListId)
'''
@app.route("/all_taskList", methods=["GET"])
def get_all_taskList():
    ''' Get all tasklist names in DATA_RECORD_ROOT starting with "TL"
    '''
    response = []
    for dir in os.listdir(file_utils.DATA_RECORD_ROOT):
        if dir.startswith("TL"):
            response.append(dir)
    return {"result": response}

'''
Name: get_taskList_history
Method: Get
Form:
    - taskListId

Respone:
    - List(tasktlist history)
'''
@app.route("/taskList_history", methods=["GET"])
def get_taskList_history():
    ''' Not called from the frontend currently.
    '''
    taskListId = request.args.get("taskListId")

    taskList_path = file_utils.get_task_list_path(taskListId)
    response = []
    for file_name in os.listdir(taskList_path):
        if file_name.startswith("TL") and len(file_name.split('_')) == 2:
            response.append(file_name)
    return {"result": response}

'''
Name: get_taskList
Method: Get
Form:
    - taskListId
    - timestamp (Optional)

Respone:
    - taskList
'''
@app.route("/taskList", methods=["GET"])
def get_taskList():
    ''' Get the task list TLxxx.json file under "DATA_RECORD_ROOT/TLxxx/"
    '''
    taskListId = request.args.get("taskListId")
    timestamp = request.args.get("timestamp") 
    print(f'taskListId: {taskListId}, timestamp: {timestamp}')
    return file_utils.load_task_list_info(taskListId)


'''
Name: update_taskList
Method: Post
Content-Type: multipart/form-data
Form:
    - taskList
    - timestamp
'''
@app.route("/taskList", methods=["POST"])
def update_taskList():
    taskList = json.loads(request.form.get("taskList"))
    timestamp = int(request.form.get("timestamp"))
    taskListId = taskList['id']
    task_list_path = file_utils.get_task_list_path(taskListId)
    file_utils.mkdir(task_list_path)

    taskList_info_path = file_utils.get_task_list_info_path(taskListId)
    taskList_info_timestamp_path = file_utils.get_task_list_info_path(taskListId, timestamp)
    file_utils.save_json(taskList, taskList_info_path)
    file_utils.save_json(taskList, taskList_info_timestamp_path)

    print('taskList id:', taskListId)
    for task in taskList['tasks']:
        taskId = task['id']
        print('task id:', taskId)
        task_path = file_utils.get_task_path(taskListId, taskId)
        task_info_path = file_utils.get_task_info_path(taskListId, taskId)
        file_utils.mkdir(task_path)
        file_utils.save_json(task, task_info_path)
        for subtask in task['subtasks']:
            subtaskId = subtask['id']
            subtask_path = file_utils.get_subtask_path(taskListId, taskId, subtaskId)
            subtask_info_path = file_utils.get_subtask_info_path(taskListId, taskId, subtaskId)
            file_utils.mkdir(subtask_path)
            file_utils.save_json(subtask, subtask_info_path)
    return ""

