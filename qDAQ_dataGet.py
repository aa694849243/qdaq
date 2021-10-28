#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2020/9/24 23:42
@Author : Wall@Synovate
@Email  : qxu@sonustc.com
@Function: real time speed recognition and NVH calculation
"""

import gc
from flask import Flask, request, Response
from flask_cors import CORS
from gevent import pywsgi
import os
import sys
import time
import logging
import traceback
import requests
from threading import Thread
from queue import Queue
import nidaqmx
import global_var as gv
from DAQTask import DAQTask, reset_ni_device
from parameters import Parameters
from common_info import config_folder, config_file, ni_device
from initial import confirm_target_folder, time_get
from utils import create_properities, file_namer, write_tdms


# set the global variable (to avoid stop with out start or start without stop)
start_command_flag = False
stop_command_flag = True
status_flag = 0


# define a server to receive the request of test bench
app = Flask(__name__)
CORS(app, supports_credentials=True)  # make server can be accessed from other PC
# ================================================================================================


# 定义台架接口（类red-ant)
@app.route('/AQS2RTRemoteControl/Command')
def qdaq():
    # define global data for real time update
    global start_command_flag, stop_command_flag, status_flag, gv_dict_rawdata
    global file_info, file_name, raw_data_save_properties, param, timer_queue, dtask_queue, timer_flag, dtask_flag
    # create queue to transfer data between threads

    try:
        # get command value from the request content
        cmd = request.args.get('Cmd')
    except Exception:
        # error request, 防止请求指令错误
        resp = Response(response="Error=0\nState=-1\nXInfo=cmd get error",
                        mimetype='text/plain')
        logging.error("command get failed, failed msg:" + traceback.format_exc())
        return resp

    # cmd = 1 means initial the test, cmd = 4 means terminate the test
    if cmd == '1':
        # to avoid double send start command
        if start_command_flag:
            resp = Response("Error=0\nState=2\nXInfo=start again")
            return resp
        else:
            start_command_flag = True
            stop_command_flag = False

        try:
            request_content = request.args.get('PropValues').split(';')
            if not request_content[0]:
                # type info missing
                resp = Response(response="Error=0\nState=-1\nXInfo=type missing",
                                mimetype='text/plain')
                return resp
            if not request_content[1]:
                # type info missing
                resp = Response(response="Error=0\nState=-1\nXInfo=serial number missing",
                                mimetype='text/plain')
                return resp
        except Exception:
            # error request, 防止请求指令错误
            resp = Response(response="Error=0\nState=-1\nXInfo=prop value get error",
                            mimetype='text/plain')
            logging.error(
                "type and serial number get failed, failed msg:" + traceback.format_exc())
            return resp

        try:
            # 记录启动信息到日志文件中（包括产品类型和序列号）
            logging.info("=" * 100)
            logging.info(
                "got the start command and basic info is: Type: " + request_content[
                    0] + "; SerialNo: " +
                request_content[1])
            time_click_start = time.time()
            logging.debug("time_click_start:{}".format(time_click_start))
            status_flag = -1
            # 参数配置读取
            config_file_name = os.path.join(config_folder,
                                            "_".join([request_content[0], config_file]))
            # 校验并读取制定产品类型的配置文件
            param = Parameters(config_file_name)
            logging.info("parameters found, file name: " + config_file_name)
        except FileNotFoundError:
            # 如果用户输入类型的配置参数文件不存在
            resp = Response(response="Error=0\nState=-1\nXInfo=param file not existed",
                            mimetype='text/plain')
            logging.error("parameters({}) file not existed, failed msg:".format(
                config_file_name) + traceback.format_exc())
            return resp
        except Exception:
            # 规避其他参数配置文件读取错误
            resp = Response(response="Error=0\nState=-1\nXInfo=param reading error",
                            mimetype='text/plain')
            logging.error("parameters({}) reading failed, failed msg:".format(
                config_file_name) + traceback.format_exc())
            return resp

        try:
            # 将产品类型和序列号更新到基础信息中
            param.basicInfo["serialNo"] = request_content[1]
            # create file info for the Data and Report
            # 获取开始测试的时间
            start_timestamp = time_get()
            # 确认需要的文件夹是否存在，若不存在则创建
            file_info = confirm_target_folder(start_timestamp, param.folderInfo,
                                              param.basicInfo)
            # 创建序列号_时间戳的文件名用于保存结果数据，原始数据等文件
            raw_data_save_properties = create_properities(start_timestamp,
                                                          1 / param.taskInfo['sampleRate'],
                                                          param.taskInfo['sampsPerChan'])
            file_name = file_namer(request_content[1], file_info[2])
            gv_dict_rawdata = gv.set_default_rawdata_with_channelNames(param.taskInfo["channelNames"])
            logging.info("initial values already confirmed!")
        except Exception:
            resp = Response(response="Error=0\nState=-1\nXInfo=initial error", mimetype='text/plain')
            traceback.print_exc()
            logging.error("data initial exec failed, failed msg:" + traceback.format_exc())
            return resp
        # 启动数据采集任务
        dtask_flag = True
        dtask_queue.put({'Cmd': '1', 'param': param.taskInfo, 'Type': request_content[0], 'serialNo': request_content[1]})
        # 启动定时任务
        timer_flag = True
        timer_queue.put(
            {'Cmd': '1', 'timeout': int(param.taskInfo["timeout"])})
        resp = Response(response="Error=0\nState=1\nXInfo=test start", mimetype='text/plain')
        return resp

    elif cmd == '2':
        resp = Response(response="Error=0\nState=1\nXInfo=", mimetype='text/plain')

    elif cmd == '3':
        resp = Response(response="Error=0\nState=2\nXInfo=", mimetype='text/plain')

    elif cmd == '4':
        if stop_command_flag:
            # 判断是否已经发送过结束命令
            resp = Response(response="Error=0\nState=0\nXInfo=already stop", mimetype='text/plain')
            return resp
        # first time to send the terminate command
        try:
            dtask_flag = False
            timer_flag = False
            logging.info("data reading stop!")
            if param.dataSaveFlag["rawData"]:
                raw_data_filename = os.path.join(file_info[0], file_name + '.tdms')
                for index, channel in enumerate(param.taskInfo['channelNames']):
                    raw_data_save_properties['NI_ChannelName'] = channel
                    raw_data_save_properties['NI_UnitDescription'] = param.taskInfo['units'][index]
                    if index == 0:
                        rawData_length = len(gv_dict_rawdata[channel])
                    write_tdms(raw_data_filename, 'AIData', channel,
                               gv_dict_rawdata[channel][:rawData_length],
                               raw_data_save_properties)
                logging.info("raw data already saved, filename: " + raw_data_filename)
            else:
                logging.info("raw data not saved!")
            del file_info
            del file_name
            del raw_data_save_properties
            del param
            gc.collect()
        except Exception:
            traceback.print_exc()
            logging.warning("raw data save failed, failed msg:" + traceback.format_exc())
        finally:
            # 重置参数并删除变量
            status_flag = 0
            start_command_flag = False
            stop_command_flag = True
            resp = Response(response="Error=0\nState=0\nXInfo=test stop", mimetype='text/plain')
            return resp
    else:
        # to response the error command
        resp = Response(response="Error=0\nState=1\nXInfo=cmd error", mimetype='text/plain')
        return resp


def rawdata_producer():
    # get raw data(from NI device)
    global gv_dict_rawdata, dtask_queue, dtask_flag
    while True:
        # 只有cmd=1的时候才能获取到队列里的数据，其他时候等待
        data = dtask_queue.get()
        task_info = data['param']
        if data['Cmd'] == '1':
            # create a task to read the raw data
            try:
                dtask = DAQTask(task_info)
                dtask.createtask()
                dtask.start_task()
            except nidaqmx.errors.DaqError:
                # if niDaq error, reset device and create task again
                logging.info('NI Daq error and reset device')
                dtask.stop_task()
                dtask.clear_task()
                reset_ni_device(ni_device)
                dtask = DAQTask(task_info)
                dtask.createtask()
                dtask.start_task()
            logging.info("data simu task started")
            while dtask_flag:
                try:
                    # read out the data from NI memory
                    cdata = dtask.read_data()
                    for i, channel_name in enumerate(task_info['channelNames']):
                        if len(task_info['channelNames']) > 1:
                            gv_dict_rawdata[channel_name].extend(cdata[i])
                        else:
                            gv_dict_rawdata[channel_name].extend(cdata)
                    del cdata
                except Exception:
                    logging.warning("NI DAQmx data reading failed, failed msg:" + traceback.format_exc())
                    break
            try:
                logging.info("data acquisition stop")
                # stop and clear the DAQ task
                dtask.stop_task()
                dtask.clear_task()
            except Exception:
                traceback.print_exc()
                logging.warning("NI DAQmx task stop and clear failed, failed msg:" + traceback.format_exc())


def terminate_command():
    # send stop command if data acquisition not stopped
    addr = 'http://localhost:8002/AQS2RTRemoteControl/Command?Cmd=4'
    headers = {'Content-Type': 'application/json;charset=UTF-8'}
    try:
        r = requests.get(addr, headers=headers)
        logging.info(r.text)
    except Exception:
        traceback.print_exc()
        logging.warning("internal stop command sending failed, failed msg:" + traceback.format_exc())


def auto_terminate():
    # 自动结束数据采集
    global timer_flag, timer_queue
    while True:
        # 只有cmd=1的时候才能获取到队列里的数据，其他时候等待
        data = timer_queue.get()
        if data['Cmd'] == '1':
            # 将计数器清零
            time_counter = 0
            interval = data['timeout']
            while timer_flag:
                if time_counter < interval:
                    time.sleep(1)
                    time_counter += 1
                else:
                    # 满足超时条件则调用cmd=4指令并退出
                    terminate_command()
                    break


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,  # 日志级别，只有日志级别大于等于设置级别的日志才会输出
        format='%(asctime)s %(filename)s %(funcName)s[line:%(lineno)d] %(levelname)s %(message)s',  # 日志输出格式
        datefmt='[%Y-%m-%d %H:%M:%S]',  # 日期表示格式
        filename=os.path.join('D:/qdaq/Log', 'dataGet.log'),  # 输出定向的日志文件路径
        filemode='a'  # 日志写模式，是否尾部添加还是覆盖
    )
    # real time mode
    try:
        # 开启线程执行定时任务（软件启动时开启，防止句柄增加）
        timer_queue = Queue(1)
        timertask = Thread(target=auto_terminate, args=())
        dtask_queue =Queue(1)
        data_get_task = Thread(target=rawdata_producer, args=())
        data_get_task.start()
        timertask.start()
    except Exception:
        logging.warning("NI Device reset failed, failed msg:" + traceback.format_exc())
        os.system("Pause")
        sys.exit()
    try:
        gc.enable = True
        gc.set_threshold(1, 1, 1)
        server = pywsgi.WSGIServer(('0.0.0.0', 8002), app)
        server.serve_forever()
    except Exception:
        logging.warning("exec failed, failed msg:" + traceback.format_exc())
        os.system("Pause")
        sys.exit()
