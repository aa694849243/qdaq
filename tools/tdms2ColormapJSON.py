#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 16:53:53 2021

@author: SA&Wall@Synovate

remark: read order colormap data from TDMS file and print to SigMA

version: 1.8.3

usage: python3 tdms2ColormapJSON.py -i D:\Wall_Work\3_Project\301_SigMA\Python_script\orderColormapRead -f simu-2vibs_210330035343.tdms

=============

"""

import os
import re
import sys
import getopt
import json
import datetime
import logging
import traceback
from nptdms import TdmsFile
import numpy as np


# 读取基本信息
try:
    log_folder = '/shengteng-platform/shengteng-platform-storage/pythonScript/Log'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
except Exception:
    print("script setting error")
    sys.exit()


# 新建日志文件并规定格式
logging.basicConfig(
    level=logging.DEBUG,  # 日志级别，只有日志级别大于等于设置级别的日志才会输出
    format='%(asctime)s %(filename)s %(funcName)s[line:%(lineno)d] %(levelname)s %(message)s',  # 日志输出格式
    datefmt='[%Y-%m-%d %H:%M:%S]',  # 日期表示格式
    filename=os.path.join(log_folder, 'orderColormapRead.log'),  # 输出定向的日志文件路径
    filemode='a'  # 日志写模式，是否尾部添加还是覆盖
)

# dB转换信息
sensitivity = 9.81
ref_accel = 1e-6
ref_value = 20 * np.log10(ref_accel * sensitivity)

config = {"input_filepath": "", "input_filename": ""}

# Step1: get user's input (input file path, file name)
try:
    opts, args = getopt.getopt(sys.argv[1:], '-i:-f:-h-v', ['input_filepath=', 'input_filename=', 'help', 'version'])
    for option, value in opts:
        if option in ["-h", "--help"]:
            print("usage:%s -i input_filepath -f input_filename")
            sys.exit()
        elif option in ['-i']:
            config["input_filepath"] = value
        elif option in ['-f']:
            config['input_filename'] = value
        elif option in ['-v', '--version']:
            print("order colormap(tdms) read version: v1.8.3")
            sys.exit()
except Exception:
    print('input command error')
    traceback.print_exc()
    logging.warning("input command error, failed msg:" + traceback.format_exc())
    sys.exit()


# 读取TDMS文件指定group的内容
def Speed_Vibration_X(tdms_file, Colormap_group_name):
    colormap = list()
    x_axis = list()
    temp = 0

    def tryint(s):
        try:
            return int(s)
        except ValueError:
            return s

    def str2int(v_str):
        return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

    data_group = tdms_file[Colormap_group_name]
    channels_name_list = [channel.name for channel in data_group.channels()]
    # 排序以防止数据存储循序错乱
    channels_name_list.sort(key=str2int)
    # print(channels_name_list)

    step = data_group[channels_name_list[1]].properties["wf_increment"]

    speed = list((data_group[channels_name_list[0]])[:])

    for channel_name in channels_name_list[1:]:
        # colormap.append(list(data_group[channel_name]))
        colormap.append(list(20 * np.log10(data_group[channel_name]) - ref_value))

    for i in range(0, len(colormap[0])):
        temp += step
        x_axis.append(temp)

    return speed, colormap, x_axis
    

if __name__ == "__main__":
    try:
        # this infomation for inter test
        # import time
        # t1 = time.time()
        # config = {"input_filepath": r"D:\Wall_Work\3_Project\301_SigMA\Python_script\test_data\orderColormap",
        #           "input_filename": "jinkang-1_210508065257.tdms"
        #           }

        # Step 2: merge the file path and file name, and generate the file name and file path for saving JSON
        sourcefile = os.path.join(config["input_filepath"], config["input_filename"])
        # Step3: read the TDMS file and write it in JSON format
        with TdmsFile.open(sourcefile) as tdms_file:
            result = list()
            for group in tdms_file.groups():
                if "ATEOLOSMAP_" in group.name:
                    Speed, Colormap, Order = Speed_Vibration_X(tdms_file, group.name)
                    single_sensor_data_length = len(Speed)
                    sensorNum = len(Colormap) // single_sensor_data_length
                    for j in range(sensorNum):
                        data = list()
                        data.append({'xName': "Order",
                                     'xUnit': "",
                                     'xValue': Order,
                                     'yName': "Speed",
                                     'yUnit': "RPM",
                                     'yValue': Speed,
                                     'zName': "Colormap",
                                     'zUnit': "dB",
                                     'zValue': Colormap[j*single_sensor_data_length: (j+1)*single_sensor_data_length]})

                        result.append({'input_filename': config["input_filename"],
                                       'sensorId': 'sensor0' + str(j + 1),
                                       'testName': group.name.replace('ATEOLOSMAP_', ''),
                                       'data': data})
        try:
            # 打印输出数据
            print(json.dumps(result))  # Return value to command console

            # for cdata in result:
            #     test_name = cdata['testName']
            #     sensor_id = cdata['sensorId']
            #     with open(os.path.join(config["input_filepath"], sensor_id + '-' + test_name + '_colormap.json'), 'w') as f:
            #         json.dump(cdata['data'][0], f)
            # with open(os.path.join(config['input_filepath'], 'tempColormap.json'), 'w') as f:
            #     json.dump(result, f)
        except Exception:
            print('order colormap data save error')
            traceback.print_exc()
            logging.warning("colormap tdms reading script errors, failed msg:" + traceback.format_exc())
            sys.exit()
        # print("Done")
        # print(time.time() - t1)
    except Exception:
        print('order colormap data script error')
        traceback.print_exc()
        logging.warning("colormap tdms reading script errors, failed msg:" + traceback.format_exc())
        sys.exit()
