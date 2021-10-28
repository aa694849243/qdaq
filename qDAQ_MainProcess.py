#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2020/9/24 23:42
@Author : Wall@Synovate
@Email  : qxu@sonustc.com
@Function: real time speed recognition and NVH calculation
"""

import gc
import multiprocessing
import subprocess
from queue import Empty

from flask import Flask, request, Response, render_template
from flask_cors import CORS
from gevent import pywsgi
import json
import time
import os
import sys
import traceback
import requests
from multiprocessing import Process, Manager, shared_memory, Lock as Process_Lock, \
    Queue as Process_Queue
import global_var as gv
from threading import Thread
from queue import Queue
from parameters import Parameters
from common_info import config_folder, config_file, speed_signal, sensor_count, max_size, read_type, \
    flag_index_dict, qDAQ_logger, hardware_confirm_flag, cpu_serial, \
    ni_device, version, board,test_bench_ip
from initial import confirm_target_folder, create_empty_final_result, time_get
from utils import create_properities, file_namer
from nvh_calculate import nvh_process
from speed_process import speed_process
from data_pack import datapack_process
from hardware_tools import get_cpu_id, get_bios_id
from DAQTask import reset_ni_device
from mic_calibrate import cal_task, end_task, acquire as sen_acquire
import mic_calibrate

# set the global variable (to avoid stop with out start or start without stop)
# 是否正在进行测试的flag，start_command_flag=True，说明正在进行测试
# 为了避免多次发送开始结束指令
start_command_flag = False
stop_command_flag = True

# 前端发送simu请求的flag，若在前端simu任务内，simu_start_flag为True，
# eg：前端要求10份数据循环simu一次，共会simu10次，
# 在第一次simu完成后start_command_flag是False,但simu_start_flag仍为True,
# 第10次simu完成后simu_start_flag才变成False

simu_start_flag = False
simu_end_flag = True

# 前端是否发送了结束simu的请求
simu_end_request=False

# start时要向其它进程的put内容，stop时要从其它进程get内容，若参数配置等异常出现，还未向其它进程的put内容
# stop就会没有办法get到数据，会无法stop,若已向其它进程put内容，则将改flag置为True
queue_put_flag = False
status_flag = 0

# 该dict与gv_dict_status中内容相同，
# 由于shared_memory_dict不支持对嵌套dict的修改，也不支持对list的的append操作，
# 所以每次更新要将第一层dict的所有内容重新刷入。
# 每个进程中该temp始终与gv_dict_status保持一致
gv_dict_status_temp = gv.set_default_status()

# define a server to receive the request of test bench
app = Flask(__name__)
CORS(app, supports_credentials=True)  # make server can be accessed from other PC


# ================================================================================================

# 定义软件版本信息访问接口
@app.route('/version', methods=['GET'])
def get_version():
    resp = Response(
        response=json.dumps({"code": 200, "msg": "20210901_A", "data": "version: v3.4.1"}),
        mimetype='application/json')
    return resp


# 定义实时显示的路由(qDAQ实时显示界面），这里需要注意出程序必须个前端页面的templates和static文件夹放在一起才能起作用
@app.route('/realTimeShowing')
def test():
    return render_template('templates.html')


@app.route('/miccalibrate/start', methods=['GET', 'POST'])
def mic_cal():
    if mic_calibrate.startflag:
        resp = Response(response=json.dumps(
            {"code": 3000, "msg": "repeated start", "data": None}), mimetype='application/json')
        return resp
    mic_parameter = json.loads(request.data)
    channel = ['ai0', 'ai1', 'ai2', 'ai3'].index(mic_parameter.get('channel'))
    access = mic_parameter.get('access')
    ampl = mic_parameter.get('ampl')
    sample_freq = 102400
    if access == 'Mic':
        thread1 = Thread(
            target=cal_task,
            args=(
                int(channel),
                int(ampl),
                int(sample_freq),
            ),
            daemon=True,
        )
        thread1.start()
        resp = Response(
            response=json.dumps({"code": 200, "msg": "sucess", "data": None}),
            mimetype='application/json')
    else:
        resp = Response(response=json.dumps(
            {"code": 3000, "msg": "access error", "data": None}), mimetype='application/json')
    return resp


@app.route('/miccalibrate/end', methods=['GET', 'POST'])
def end_calibrate():
    res = end_task()
    resp = Response(response=json.dumps(
        {"code": 200, "msg": "sucess", "data": list(res)}), mimetype='application/json')
    return resp


@app.route('/miccalibrate/acquire', methods=['GET', 'POST'])
def acquire_sensitivity():
    val = sen_acquire()
    print(val)
    resp = Response(response=json.dumps(
        {"code": 200, "msg": "sucess", "data": val}), mimetype='application/json')
    print(mic_calibrate.q)
    return resp


# 定义网页实时显示端的状态请求路径
@app.route('/QDAQRemoteControl/Command', methods=['GET'])
def init():
    global gv_dict_status, gv_dict_status_temp
    # 初始化响应
    resp = Response(
        response=json.dumps({"code": 0, "msg": '检测未开始！', "data": None}),
        mimetype='application/json')
    try:
        cmd = request.args.get('Cmd')
        if cmd == "5":
            # speed curve and program status query
            if status_flag == 0:
                # no start command
                resp = Response(response=json.dumps(
                    {"code": 0, "msg": '检测未开始！', "data": None}),
                    mimetype='application/json')
            else:
                # Manager().dict()不能当作参数进行json.dumps，故需要先赋值出来
                # 由于不管怎么样都要先拷贝出来，将gv_dict_status扩展为所有内存之间需要共享的dict，
                # 尝试将dict中的所有数据放在status里面，但是变得很慢，在工控机上慢了大概30s
                # 不再只是需要向前端返回的数据
                gv_dict_status_temp['code'] = gv_dict_status['code']
                gv_dict_status_temp['result'] = gv_dict_status['result']
                gv_dict_status_temp['msg'] = gv_dict_status['msg']
                gv_dict_status_temp['data'] = gv_dict_status['data']

                resp = Response(response=json.dumps(gv_dict_status_temp), mimetype='application/json')
        else:
            # 防止Cmd错误
            resp = Response(response=json.dumps(
                {"code": 3000, "msg": "请求命令错误，请确认!", "data": None}), mimetype='application/json')
    except Exception:
        # 防止请求异常
        resp = Response(
            json.dumps({"code": 3000, "msg": "无法识别的请求!", "data": None}), mimetype='application/json')
    finally:
        return resp


@app.route('/frontEndControl/Start', methods=['POST'])
def frontEndControlStart():
    global simu_queue, simu_start_flag, simu_end_flag,simu_end_request
    if simu_start_flag:
        # 前端重复发送了simu指令
        resp = Response(response=json.dumps({"code": 3000, "msg": "simu指令重复发送", "data": None}),
                        mimetype='application/json')
        return resp
    try:
        type_from_front = request.form.get("type")
        serial_no_list = request.form.getlist("serialNoList")
        simu_count = int(request.form.get("simu_count"))
    except Exception:
        resp = Response(response=json.dumps({"code": 3000, "msg": "无法解析请求，请确认!", "data": None}),
                        mimetype='application/json')
        qDAQ_logger.error("parse request error:" + traceback.format_exc())
        return resp
    if len(serial_no_list) < 1 or simu_count < 1:
        resp = Response(response=json.dumps({"code": 3000, "msg": "序列号个数及simu次数均应大于0!", "data": None}),
                        mimetype='application/json')
        return resp
    simu_queue.put({
        "simu_count": simu_count,
        "type_from_front": type_from_front,
        "serial_no_list": serial_no_list
    })
    simu_start_flag = True
    simu_end_flag = False
    simu_end_request=False
    resp = Response(response=json.dumps({"code": 200, "msg": "已经成功将simu任务加进任务队列!", "data": None}),
                    mimetype='application/json')
    return resp

@app.route('/frontEndControl/Stop', methods=['POST'])
def frontEndControlStop():
    global simu_queue, simu_start_flag, simu_end_flag,simu_end_request
    if simu_end_flag:
        # 前端重复发送了simu指令
        resp = Response(json.dumps({"code": 3000, "msg": "重复发送了结束指令", "data": None}))
        return resp
    simu_end_request=True
    resp = Response(json.dumps({"code": 3000, "msg": "结束指令发送成功，该文件simu结束后会结束simu任务", "data": None}))
    return resp

@app.route("/testBench",methods=["GET"])
def testBench():
    global test_bench_ip
    with open(os.devnull, "wb") as limbo:
        result = subprocess.Popen(["ping", "-n", "1", "-w", "2", test_bench_ip],
                                  stdout=limbo, stderr=limbo).wait()
        if result:
            #
            return Response(response=json.dumps({"code": 200, "msg": 'inactive', "data": "inactive"}))
        else:
            return Response(response=json.dumps({"code": 200, "msg": 'active', "data": 'active'}))

@app.route("/resetCount",methods=["POST"])
def resetCount():
    global Q_dict
    if simu_start_flag or start_command_flag:
        return Response(response=json.dumps({"code": 3000, "msg": 'qdaq正在工作,请等待工作完成后重置计数', "data": "reset success"}))
    Q_dict['Q_datapack_in'].put("reset")
    Q_dict['Q_datapack_out'].get()
    return Response(response=json.dumps({"code": 200, "msg": 'reset success', "data": "reset success"}))


# 前端发送simu指令的queue
def start_stop_command():
    global simu_queue, simu_start_flag, simu_end_flag,simu_end_request,gv_dict_status
    while True:
        try:
            data = simu_queue.get()
        except Exception:
            print("error")
        simu_count = data["simu_count"]
        serial_no_list = data["serial_no_list"]
        type_from_front = data["type_from_front"]
        qDAQ_logger.info(simu_count)
        simu_all_times=simu_count*len(serial_no_list)
        simu_time=1


        for i in range(simu_count):
            for serial_no_from_front in serial_no_list:
                # 前端发送了结束本次simu任务的请求，结束本次simu
                if  simu_end_request:
                    break

                url_start = "http://" + "localhost" + ":" + "8002" + \
                            "/AQS2RTRemoteControl/Command?Cmd=1&Prop=1&PropNames=Type;SerialNo&PropValues=" \
                            + type_from_front + ";" + serial_no_from_front+"&mode=simu"
                headers = {'Content-Type': 'application/json;charset=UTF-8'}
                qDAQ_logger.info("发送开始指令")
                while start_command_flag:
                    time.sleep(1)
                r_start_command = requests.get(url_start, headers=headers)
                # 判断是否可以发送结束指令
                qDAQ_logger.info("等待发送结束指令")

                while not simu_end_request:
                    time.sleep(1)
                    # 其它进程中出现了错误 或者nvh进程正常结束了,说明可以发送结束请求了。
                    # 此处逻辑:前端发送了结束指令，这里会
                    qDAQ_logger.info("等待发送结束指令")
                    if gv_dict_flag[flag_index_dict["speed_error"]] or gv_dict_flag[flag_index_dict["nvh_error"]] or \
                            gv_dict_flag[flag_index_dict["datapack_error"]] or gv_dict_flag[
                        flag_index_dict["datapack_finish"]]:
                        url_stop = "http://" + "localhost" + ":" + "8002" + "/AQS2RTRemoteControl/Command?Cmd=4"
                        headers = {'Content-Type': 'application/json;charset=UTF-8'}
                        r_end_command = requests.get(url_stop, headers=headers)
                        break
                simu_time+=1
                # todo：前端显示simu到第几个了

        simu_start_flag = False
        simu_end_flag = True
        qDAQ_logger.info("type:{},serial_no_list:{} simu_count:{} simu任务已经完成".format(type_from_front
                                                                          ,serial_no_list,simu_count))


# 定义台架接口（类red-ant)
@app.route('/AQS2RTRemoteControl/Command')
def qdaq():
    # 初始化台架请求的响应
    global start_command_flag, stop_command_flag, queue_put_flag, status_flag, gv_dict_flag, param
    # 新建定时器，timertask是用于结束命令的控制（未得到结束命令则自动结束）
    global timer_flag, timer_queue, simu_start_flag, simu_end_flag

    try:
        # get command value from the request content
        cmd = request.args.get('Cmd')
    except Exception:
        # error request, 防止请求指令错误
        resp = Response(response="Error=0\nState=-1\nXInfo=cmd get error",
                        mimetype='text/plain')
        qDAQ_logger.error("command get failed, failed msg:" + traceback.format_exc())
        return resp

    # cmd = 1 means initial the test, cmd = 4 means terminate the test
    if cmd == '1':
        is_simu_mode = True if request.args.get("mode") and request.args.get("mode").lower()=="simu"  else False
        if not is_simu_mode and simu_start_flag:
            # 这种情况是在simu前端的任务时，台架发送了实采的请求
            resp = Response("Error=0\nState=2\nXInfo=simu start again")
            return resp

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
                resp = Response(response="Error=0\nState=-1\nXInfo=",
                                mimetype='text/plain')
                return resp
        except Exception:
            # error request, 防止请求指令错误
            resp = Response(response="Error=0\nState=-1\nXInfo=prop value get error",
                            mimetype='text/plain')
            qDAQ_logger.error(
                "type and serial number get failed, failed msg:" + traceback.format_exc())
            return resp
        try:
            # 获取开始测试的时间
            start_timestamp = time_get()

            # 记录启动信息到日志文件中（包括产品类型和序列号）
            qDAQ_logger.info("=" * 100)
            qDAQ_logger.info(
                "got the start command and basic info is: Type: " + request_content[
                    0] + "; SerialNo: " +
                request_content[1])
            time_click_start = time.time()
            qDAQ_logger.debug("time_click_start:{}".format(time_click_start))
            status_flag = -1
            # 初始化全局变量
            gv_dict_status['code'] = 0
            gv_dict_status["result"] = None
            gv_dict_status['msg'] = ""
            gv_dict_status["sectionResult"] = {"testName": list(), "limitComResult": list(), "mlResult": list()}
            gv_dict_status["allCount"]=None
            gv_dict_status["abnormalCount"]=None
            gv_dict_status["unqualifiedCount"]=None

            gv_dict_status['data'] = {"type": request_content[0], "serialNo": request_content[1],
                                      "testName": list(), "startX": list(), "startY": list(),
                                      "endX": list(), "endY": list(), "x": 0.0, "y": 0.0,
                                      "testResult": None, "reportPath": None,"testStartTime":start_timestamp.timestamp()}
            gv_dict_status['xml'] = list()
            # create flag dict to control the thread
            # 开启相应进程内的任务（转速计算识别，NVH分析和结果数据封装）
            gv_dict_flag[flag_index_dict['speedCalclation']] = 1
            gv_dict_flag[flag_index_dict['nvhCalclation']] = 1
            gv_dict_flag[flag_index_dict['dataPack']] = 1
        except Exception:
            resp = Response(response="Error=0\nState=-1\nXInfo=set default error",
                            mimetype='text/plain')
            qDAQ_logger.error("initial failed, failed msg:" + traceback.format_exc())
            return resp

        try:
            # 参数配置读取
            config_file_name = os.path.join(config_folder,
                                            "_".join([request_content[0], config_file]))
            # 校验并读取制定产品类型的配置文件
            param = Parameters(config_file_name)
            qDAQ_logger.info("parameters found, file name: " + config_file_name)
        except FileNotFoundError:
            # 如果用户输入类型的配置参数文件不存在，在实时显示界面提示给用户
            gv_dict_status["code"] = 3000
            gv_dict_status["msg"] = "该类型的参数信息不存在!"
            resp = Response(response="Error=0\nState=-1\nXInfo=param file not existed",
                            mimetype='text/plain')
            qDAQ_logger.error("parameters({}) file not existed, failed msg:".format(
                config_file_name) + traceback.format_exc())
            return resp
        except Exception:
            # 规避其他参数配置文件读取错误
            gv_dict_status["code"] = 3000
            gv_dict_status["msg"] = "参数信息读取错误!"
            resp = Response(response="Error=0\nState=-1\nXInfo=param reading error",
                            mimetype='text/plain')
            qDAQ_logger.error("parameters({}) reading failed, failed msg:".format(
                config_file_name) + traceback.format_exc())
            return resp

        try:
            # 将产品类型和序列号更新到基础信息中
            param.basicInfo["serialNo"] = request_content[1]
            # create file info for the Data and Report

            # 确认需要的文件夹是否存在，若不存在则创建
            file_info = confirm_target_folder(start_timestamp, param.folderInfo,
                                              param.basicInfo)
            # 创建序列号_时间戳的文件名用于保存结果数据，原始数据等文件
            param.basicInfo["fileName"] = file_namer(request_content[1], file_info[2])

            if read_type == 'hdf5':
                param.simuInfo["fileName"] = os.path.join(param.simuInfo["fileFolder"],
                                                          request_content[1] + ".h5")
            else:
                param.simuInfo["fileName"] = os.path.join(param.simuInfo["fileFolder"],
                                                          request_content[1] + ".tdms")

            # create empty result of final result
            # 基于基础信息来创建空的结果数据
            test_result = create_empty_final_result(start_timestamp, param.basicInfo,
                                                    param.speedRecogInfo,
                                                    param.limitCompareFlag['overLimit'])
            # set raw data save properties(raw data and colormap)
            colormap_save_properties = create_properities(start_timestamp, param.orderSpectrumCalcInfo[
                'orderResolution'], len(param.orderSpectrumCalcInfo['order']))
            qDAQ_logger.info("initial values already confirmed!")
        except Exception:
            gv_dict_status["code"] = 3000
            gv_dict_status["msg"] = "初始化失败，请确认参数信息!"
            resp = Response(response="Error=0\nState=-1\nXInfo=initial error", mimetype='text/plain')
            traceback.print_exc()
            qDAQ_logger.error("data initial exec failed, failed msg:" + traceback.format_exc())
            return resp

        try:
            # no error of initial
            # 传递参数参数到相应的进程
            # 1. 传递参数到原始数据和转速计算识别进程
            Q_dict['Q_speed_in'].put(
                {"param": param, "gv_dict_status": gv_dict_status, "file_info": file_info,
                 "start_timestamp": start_timestamp, "time_click_start": time_click_start,
                 "is_simu_mode":is_simu_mode})
            # 2. 传递参数到结果数据封装进程
            Q_dict['Q_datapack_in'].put(
                {"gv_dict_status": gv_dict_status, "param": param, "test_result": test_result,
                 "time_click_start": time_click_start,"is_simu_mode":is_simu_mode})
            # 传递数据到nvh分析进程（分别给指定传感器传输）
            for i in range(sensor_count):
                Q_dict["Q_nvh_in_" + str(i)].put(
                    {"gv_dict_status": gv_dict_status, "param": param, "test_result": test_result,
                     "sensor_index": i, "time_click_start": time_click_start, "file_info": file_info,
                     "colormap_save_properties": colormap_save_properties,"is_simu_mode":is_simu_mode})
            queue_put_flag = True

            # 启动定时任务
            timer_flag = True
            timer_queue.put({'Cmd': '1', 'timeout': int(param.taskInfo["timeout"])})

            # task start succeed
            status_flag = 1
            gv_dict_status["code"] = status_flag
            gv_dict_status["msg"] = "数据采集处理中..."
            # 记录开始信息
            qDAQ_logger.info("qDAQ tasks started")
            resp = Response(response="Error=0\nState=1\nXInfo=test start", mimetype='text/plain')
            del file_info
            del colormap_save_properties
            return resp
        except Exception:
            gv_dict_status["code"] = 3000
            gv_dict_status["msg"] = "qDAQ多任务启动失败!"
            resp = Response(response="Error=0\nState=-1\nXInfo=task start error", mimetype='text/plain')
            qDAQ_logger.error("qDAQ tasks start failed, failed msg:" + traceback.format_exc())
            return resp

    # 目前暂不处理start和stop命令（这两个命令主要用于手动标定测试段）
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
            # 停止相应进程内的任务
            time_click_stop = time.time()
            # TODO 将gv_dict_flag中的数据归0
            gv_dict_flag[flag_index_dict['speedCalclation']] = 0
            gv_dict_flag[flag_index_dict['nvhCalclation']] = 0
            gv_dict_flag[flag_index_dict['dataPack']] = 0
            gv_dict_flag[flag_index_dict['speed_finish']] = 0
            gv_dict_flag[flag_index_dict['datapack_finish']] = 0
            gv_dict_flag[flag_index_dict['speed_error']] = 0
            gv_dict_flag[flag_index_dict['nvh_error']] = 0
            gv_dict_flag[flag_index_dict['datapack_error']] = 0

            # 如果已向其它进程put数据，则需要等待其它进程中的方法执行结束
            if queue_put_flag:
                timer_flag = False
                qDAQ_logger.info("data reading stop!")
                Q_dict['Q_speed_out'].get()
                qDAQ_logger.info("speed_process stop")
                for i in range(sensor_count):
                    sensor_index = Q_dict["Q_nvh_out_" + str(i)].get()
                    qDAQ_logger.info("nvh_process{} stop".format(sensor_index))
                Q_dict['Q_datapack_out'].get()
                qDAQ_logger.info("datapack_process stop")
            # 将speed和nvh进程通信队列清空
            for Q_speed_nvh in Q_speed_nvh_list:
                while True:
                    try:
                        Q_speed_nvh.get(block=True, timeout=0.01)
                    except Empty:
                        break
            del param
            gc.collect()
            # 记录结束信息
            qDAQ_logger.info("qDAQ tasks stopped")
        except Exception:
            gv_dict_status["code"] = 3000
            gv_dict_status["msg"] = "qDAQ多任务结束失败!"
            qDAQ_logger.error("qDAQ tasks terminate failed, failed msg:" + traceback.format_exc())
        finally:
            # 重置参数并删除变量
            status_flag = 0
            start_command_flag = False
            stop_command_flag = True
            gv_dict_status['code'] = 0
            gv_dict_status["result"] = -2
            gv_dict_status["sectionResult"] = {"testName": list(), "limitComResult": list(), "mlResult": list()}
            gv_dict_status["allCount"]=None
            gv_dict_status["abnormalCount"]=None
            gv_dict_status["unqualifiedCount"]=None
            gv_dict_status['msg'] = ""
            gv_dict_status['xml'] = list()
            gv_dict_status['data'] = {"type": "", "serialNo": "", "testName": list(), "startX": list(),
                                      "startY": list(), "endX": list(), "endY": list(), "x": 0.0,
                                      "y": 0.0, "testResult": -2, "reportPath": None}
            gc.collect()
            qDAQ_logger.debug("time_for_stop:{}".format(time.time() - time_click_stop))
            resp = Response(response="Error=0\nState=0\nXInfo=test stop", mimetype='text/plain')
            return resp
    else:
        # to response the error command
        resp = Response(response="Error=0\nState=1\nXInfo=", mimetype='text/plain')
    return resp


# 定义结果返回接口(XML)
@app.route('/AQS2RTRemoteControl/Result')
def result():
    resp = Response(response="time out, no result available",
                    mimetype='text/plain')
    try:
        if request.args.get('Timeout'):
            result_request_timeout = float(request.args.get('Timeout')) / 1000
        else:
            result_request_timeout = 2
        gv_dict_status_temp['xml'] = gv_dict_status['xml']

        if gv_dict_status_temp["xml"]:
            # 每次访问如果存在xml则返回最早的一份
            resp = Response(response=gv_dict_status_temp["xml"].pop(0), status=200,
                            mimetype="application/xml")
            gv_dict_status['xml'] = gv_dict_status_temp['xml']
        else:
            # 若访问的时候不存在xml，则规定时间（来自设置文件）后再访问一次，仍不存在则直接放回超时
            time.sleep(result_request_timeout)
            gv_dict_status_temp['xml'] = gv_dict_status['xml']
            if gv_dict_status_temp["xml"]:
                resp = Response(response=gv_dict_status_temp["xml"].pop(0), status=200,
                                mimetype="application/xml")
                gv_dict_status['xml'] = gv_dict_status_temp['xml']
            else:
                resp = Response(response="time out3, no result available", mimetype='text/plain')
    except Exception:
        resp = Response(response="time out4, no result available", mimetype='text/plain')
    finally:
        return resp


def terminate_command():
    # send stop command if data acquisition not stopped(can not receive cmd=4 from test bench)
    addr = 'http://localhost:8002/AQS2RTRemoteControl/Command?Cmd=4'
    headers = {'Content-Type': 'application/json;charset=UTF-8'}
    try:
        print("internal stop")
        r = requests.get(addr, headers=headers)
        qDAQ_logger.info(r.text)
    except Exception:
        gv_dict_status["code"] = 3000
        gv_dict_status["msg"] = "结束命令发送失败!"
        qDAQ_logger.error(
            "internal stop command sending failed, failed msg:" + traceback.format_exc())


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
    # 主函数，程序执行的入口
    qDAQ_logger.debug("main start")
    multiprocessing.freeze_support()  # 保护主进程
    global gv_dict_status, timer_queue, simu_queue
    # global Q_dict
    try:
        # 信息校验
        # 硬件信息校验
        if hardware_confirm_flag:
            # 校验cpu信息（由于组装机可能无法获取bios信息，所以不再进行bios校验）
            if cpu_serial not in get_cpu_id():
                print("cpu info not matched, SN should be：{}".format(cpu_serial))
                raise Exception("cpu错误")
        # 转速信号校验，可以是ttl（脉冲信号），resolver（单路旋变），resolver2（双路旋变），都不是则报错
        if speed_signal != "ttl" and speed_signal != "resolver" and speed_signal != "resolver2":
            qDAQ_logger.error("speed_signal设置不合理")
            raise Exception(
                "speed_signal:{}设置不合理，应为\"ttl\"或\"resolver\"或\"resolver2\"".format(speed_signal))

        # 传感器个数校验，至少1个传感器
        if sensor_count <= 0:
            raise Exception("sensor_count:{} 设置不合理".format(sensor_count))
        # 校验整个测试的最大时长，如果开太大会额外占用内存（最大只支持102400采样率600s的数据大小）
        if max_size > 61440000 or max_size <= 0:
            raise Exception("max_seconds:{} 设置不合理,应不大于61440000(600*102400)".format(max_size))

    except Exception:
        qDAQ_logger.error("config info error, error msg:" + traceback.format_exc())
        os.system('Pause')
        sys.exit()

    if version == 1:
        # 普通qdaq
        try:
            # 假设采样率为102400，测试段时长600s
            size = max_size * 4
            # 创建共享内存
            # speed在ttl信号中存放speed通道，在旋变信号中存放sin通道，在speed进程中的speedChannel记为Speed/Sin
            shm_speed = shared_memory.SharedMemory(name="shm_speed", create=True, size=size)
            # 双路旋变，还要开启cos
            if speed_signal == "resolver2":
                shm_cos = shared_memory.SharedMemory(name="shm_cos", create=True, size=size)
            # 几个传感器，则开辟几个传感器的内存
            # 至少存在一个传感器，开辟vib0
            shm_vib_list = list()
            for i in range(sensor_count):
                shm_vib_list.append(
                    shared_memory.SharedMemory(name="shm_vib" + str(i), create=True, size=size))
            # trigger的位置,理论上每两个上升沿/下降沿所在的位置至少相差2，否则是无法识别trigger的 例：1 -1 1 -1

            shm_trigger = shared_memory.SharedMemory(name='shm_trigger', create=True,
                                                     size=int(size / 2 + 4))
            # 保存转速曲线，转速曲线的位置有trigger位置决定，最多每一个trigger处对应一个转速
            shm_rpml = shared_memory.SharedMemory(name='shm_rpml', create=True, size=int(size / 2 + 4))
            shm_rpm = shared_memory.SharedMemory(name='shm_rpm', create=True, size=int(size / 2 + 4))
            # 在连接内存的时候，不管开的时候有多大，最小连接单位为4096字节
            shm_flag = shared_memory.SharedMemory(name="shm_flag", create=True, size=4096)
            gv_dict_flag = shm_flag.buf
        except FileExistsError:
            # speed在ttl信号中存放speed通道，在旋变信号中存放sin通道，在speed进程中的speedChannel记为Speed/Sin
            shm_speed = shared_memory.SharedMemory(name="shm_speed")
            # 双路旋变，还要开启cos
            if speed_signal == "resolver2":
                shm_cos = shared_memory.SharedMemory(name="shm_cos")
            # 几个传感器，则开辟几个传感器的内存
            # 至少存在一个传感器，开辟vib0
            shm_vib_list = list()
            for i in range(sensor_count):
                shm_vib_list.append(shared_memory.SharedMemory(name="shm_vib" + str(i)))
            # 保存转速曲线
            shm_rpml = shared_memory.SharedMemory(name='shm_rpml')
            shm_rpm = shared_memory.SharedMemory(name='shm_rpm')
            # trigger的位置
            shm_trigger = shared_memory.SharedMemory(name='shm_trigger')
            # shared_memory_dict运行的时间长的话就会出错，该flag仅有True，false之分，可以放在共享内存中用0表示false，用1表示True
            shm_flag = shared_memory.SharedMemory(name="shm_flag")
            gv_dict_flag = shm_flag.buf
    elif version == 2:
        # BYD专版qdaq
        try:
            # 假设采样率为102400，测试段时长600s
            size = max_size * 4
            # 创建共享内存
            # speed在ttl信号中存放speed通道，在旋变信号中存放sin通道，在speed进程中的speedChannel记为Speed/Sin
            shm_ttl = shared_memory.SharedMemory(name="shm_ttl", create=True, size=size)
            # BYD
            # 直接开两路信号
            # 开启Sin通道
            shm_sin = shared_memory.SharedMemory(name="shm_sin", create=True, size=size)
            # 双路旋变，还要开启cos
            if speed_signal == "resolver2":
                shm_cos = shared_memory.SharedMemory(name="shm_cos", create=True, size=size)
            # 几个传感器，则开辟几个传感器的内存
            # 至少存在一个传感器，开辟vib0
            shm_vib_list = list()
            for i in range(sensor_count):
                shm_vib_list.append(
                    shared_memory.SharedMemory(name="shm_vib" + str(i), create=True, size=size))
            # trigger的位置,理论上每两个上升沿/下降沿所在的位置至少相差2，否则是无法识别trigger的 例：1 -1 1 -1
            shm_trigger = shared_memory.SharedMemory(name='shm_trigger', create=True,
                                                     size=int(size / 2 + 4))
            # 保存转速曲线，转速曲线的位置有trigger位置决定，最多每一个trigger处对应一个转速
            shm_rpml = shared_memory.SharedMemory(name='shm_rpml', create=True, size=int(size / 2 + 4))
            shm_rpm = shared_memory.SharedMemory(name='shm_rpm', create=True, size=int(size / 2 + 4))
            # 在连接内存的时候，不管开的时候有多大，最小连接单位为4096字节
            shm_flag = shared_memory.SharedMemory(name="shm_flag", create=True, size=4096)
            gv_dict_flag = shm_flag.buf
        except FileExistsError:
            # speed在ttl信号中存放speed通道，在旋变信号中存放sin通道，在speed进程中的speedChannel记为Speed/Sin
            shm_ttl = shared_memory.SharedMemory(name="shm_ttl")

            shm_sin = shared_memory.SharedMemory(name="shm_sin")
            # 双路旋变，还要开启cos
            if speed_signal == "resolver2":
                shm_cos = shared_memory.SharedMemory(name="shm_cos")
            # 几个传感器，则开辟几个传感器的内存
            # 至少存在一个传感器，开辟vib0
            shm_vib_list = list()
            for i in range(sensor_count):
                shm_vib_list.append(shared_memory.SharedMemory(name="shm_vib" + str(i)))
            # 保存转速曲线
            shm_rpml = shared_memory.SharedMemory(name='shm_rpml')
            shm_rpm = shared_memory.SharedMemory(name='shm_rpm')
            # trigger的位置
            shm_trigger = shared_memory.SharedMemory(name='shm_trigger')
            # shared_memory_dict运行的时间长的话就会出错，该flag仅有True，false之分，可以放在共享内存中用0表示false，用1表示True
            shm_flag = shared_memory.SharedMemory(name="shm_flag")
            gv_dict_flag = shm_flag.buf
    elif version == 3 or version == 4:
        # 恒速电机
        try:
            # 假设采样率为102400，测试段时长600s
            size = max_size * 4
            # 创建共享内存

            # 几个传感器，则开辟几个传感器的内存
            # 至少存在一个传感器，开辟vib0
            shm_vib_list = list()
            for i in range(sensor_count):
                shm_vib_list.append(
                    shared_memory.SharedMemory(name="shm_vib" + str(i), create=True, size=size))

            # 在连接内存的时候，不管开的时候有多大，最小连接单位为4096字节
            shm_flag = shared_memory.SharedMemory(name="shm_flag", create=True, size=4096)
            gv_dict_flag = shm_flag.buf
        except FileExistsError:

            # 几个传感器，则开辟几个传感器的内存
            # 至少存在一个传感器，开辟vib0
            shm_vib_list = list()
            for i in range(sensor_count):
                shm_vib_list.append(shared_memory.SharedMemory(name="shm_vib" + str(i)))
            # shared_memory_dict运行的时间长的话就会出错，该flag仅有True，false之分，可以放在共享内存中用0表示false，用1表示True
            shm_flag = shared_memory.SharedMemory(name="shm_flag")
            gv_dict_flag = shm_flag.buf
    # 创建status用于向前端页面返回
    gv_dict_status = Manager().dict()

    try:
        # 开启线程执行定时任务（软件启动时开启，防止句柄增加）
        timer_queue = Queue(1)
        timertask = Thread(target=auto_terminate, args=())
        timertask.start()

        # 设置自动回收垃圾
        gc.enable()
        gc.set_threshold(1, 1, 1)

        qDAQ_logger.info("server will start soon")

        lock_for_tdms = Process_Lock()
        # 设置主进程与其他进程通信的队列
        Q_dict = dict()
        Q_dict['Q_speed_in'] = Process_Queue(1)  # 主进程与speed进程
        Q_dict['Q_speed_out'] = Process_Queue(1)  # 主进程与speed进程
        Q_dict['Q_datapack_out'] = Process_Queue(1)  # 主进程与datapack进程
        Q_dict['Q_datapack_in'] = Process_Queue(1)  # 主进程与datapack进程
        Q_dict['Q_nvh_datapack'] = Process_Queue(sensor_count)  # nvh进程与datapack进程
        Q_speed_nvh_list = list()
        for i in range(sensor_count):
            Q_dict["Q_nvh_in_" + str(i)] = Process_Queue(1)  # 主进程与nvh进程
            Q_dict["Q_nvh_out_" + str(i)] = Process_Queue(1)  # 主进程与nvh进程
            # speed与nvh进程的queue不设置大小，可以一直放，只是会滞后计算
            Q_dict["Q_speed_nvh_" + str(i)] = Process_Queue()  # speed进程与nvh进程
            Q_speed_nvh_list.append(Q_dict["Q_speed_nvh_" + str(i)])

        # 进程定义
        process_dict = dict()
        # 原始数据读取和转速计算识别进程
        process_dict["speed"] = Process(target=speed_process, name="speed_process", args=(
            Q_dict['Q_speed_in'], Q_dict['Q_speed_out'], Q_speed_nvh_list))

        # nvh进程
        for i in range(sensor_count):
            process_dict["nvh_" + str(i)] = Process(target=nvh_process, name="nvh_process_" + str(i),
                                                    args=(
                                                        Q_dict["Q_nvh_in_" + str(i)],
                                                        Q_dict["Q_nvh_out_" + str(i)],
                                                        Q_dict["Q_speed_nvh_" + str(i)],
                                                        Q_dict["Q_nvh_datapack"],
                                                        i,
                                                        lock_for_tdms))
        # datapack进程
        process_dict["datapack"] = Process(target=datapack_process, name="datapack_process", args=(
            Q_dict['Q_datapack_in'], Q_dict['Q_datapack_out'], Q_dict['Q_nvh_datapack']))
        # 启动进程
        for proc in process_dict.values():
            proc.start()
    except Exception:
        qDAQ_logger.error("Process start error :" + traceback.format_exc())
        os.system('Pause')
        sys.exit()

    try:
        simu_queue = Queue(1)
        front_end_simu_thread = Thread(target=start_stop_command,args=())
        front_end_simu_thread.start()
    except Exception:
        qDAQ_logger.error("前端simu线程出现错误")
    try:

        # 启动服务器用于监听台架的请求
        server = pywsgi.WSGIServer(('0.0.0.0', 8002), app)
        server.serve_forever()
    except Exception:
        qDAQ_logger.error("Server start error :" + traceback.format_exc())
        os.system("Pause")
        sys.exit()
