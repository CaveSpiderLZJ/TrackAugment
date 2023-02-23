import os
import json
import argparse
from flask import Flask, request, send_file
from flask_cors import CORS, cross_origin
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
# CORS: A Flask extension for handling Cross Origin Resource Sharing
# (CORS), making cross-origin AJAX possible.
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

import config as cf
import file_utils as fu
from route.route_record import *
from route.route_tasklist import *
from route.route_train import *
from route.route_file import *
from route.route_dex import *

# multi-thread saver
saver = ThreadPoolExecutor(max_workers=1)
saver_future_list = []

'''
Data structure:

/data
    /record
        /{task_list_id}
            - {task_list_id}.json
            - {task_list_id}_{timestamp0}.json
            - {task_list_id}_{timestamp1}.json
            - ...
            /{task_id}
                - {task_id}.json
                /{subtask_id}
                    - {subtask_id}.json
                    /{record_id}
                        - sensor_{record_id}.json
                        - timestamp_{record_id}.json
                        - audio_{record_id}.mp4
                        - video_{record_id}.mp4
                        - sample_{record_id}.csv
'''

if __name__ == '__main__':
    fu.check_cwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default = '../data', help='Root directory of raw data.')
    args = parser.parse_args()
    fu.set_data_root(args.data_root)
    fu.create_default_files()
    update_md5()
    app.run(port=6125, host="0.0.0.0")
