import platform
import threading
import time

import matplotlib.pyplot as plt
import nidaqmx
import numpy as np
import datetime
import os
import gc
import traceback
import global_var as gv
import pyaudio
from common_info import flag_index_dict, speed_signal, max_size, sensor_count, ni_device, \
    save_type, read_type, qDAQ_logger, sleep_ratio, board, Umic_names, Umic_hostapis, version,sys_name,bit_depth
from speed_tools import speed_detect_start, speed_detect_end, resolver, trigger_detect_for_share, \
    rpm_calc_for_share, resolver_single_signal_for_share, ramp_quality, speed_calibration, \
    rpm_calc_for_share_for_byd, single_resolver_butter_filter, speed_calibration_with_orderlist
from torque_tools import speed_detect_start_with_torque, speed_detect_end_with_torque
from utils import create_properities, write_hdf5, read_raw_data, write_tdms, bit24_to_int, send_raw_data, \
    timestamp_to_time, bit_to_int
from initial import single_folder_confirm
from multiprocessing import shared_memory
from threading import Thread
from queue import Queue

if sys_name.lower()=="linux":
    from uldaq import (DaqDevice, AiInputMode, ScanStatus)
    import atexit
    from DTDAQTask import get_daq_devices, device_confirm, get_ai_device_info, ai_info_confirm, \
        get_channels_by_mode, buffer_creator, set_daq_task
elif sys_name.lower()=="windows":
    from DAQTask import DAQTask, reset_ni_device

running_mode=None

def ftp_file_upload():
    # 执行ftp上传功能
    global ftp_queue
    while True:
        # 获取数据信息
        upload_info = ftp_queue.get()
        print(upload_info)
        # 开始进行数据上传和解析
        upload_flag = send_raw_data(upload_info["urlInfo"], upload_info["folderInfo"],
                                    upload_info["dataInfo"], upload_info["formatTimeStamp"])
        if upload_flag:
            qDAQ_logger.info("data upload and analysis succeed")
        else:
            qDAQ_logger.info("data upload and analysis failed")


def speed_process(Q_speed_in, Q_speed_out, Q_speed_nvh_list):
    global gv_dict_flag, trigger_array, rpml_array, rpm_array, isResolver2, shm_vib, shm_cos, \
        shm_speed, shm_ttl, shm_sin
    global target_device, target_ai_device, target_ai_info, target_input_mode, target_mode_channels,ftp_queue
    global running_mode
    try:
        if version == 1:
            # 普通qdaq程序
            isResolver2 = (speed_signal == "resolver2")
            shm_speed = shared_memory.SharedMemory(name='shm_speed')
            # speed，nvh，datapack进程是否在运行的flag
            shm_flag = shared_memory.SharedMemory(name="shm_flag")
            gv_dict_flag = shm_flag.buf
            # 保存trigger的共享内存
            # 声明转速脉冲数据的共享内存并定义数据大小（原始数据的一半）
            shm_trigger = shared_memory.SharedMemory(name="shm_trigger")
            trigger_array = np.ndarray(shape=(max_size // 2 + 1,), dtype="i", buffer=shm_trigger.buf)
            # 保存转速曲线的共享内存
            # 声明转速曲线的共享内存并定义数据大小（原始数据的一半）
            shm_rpml = shared_memory.SharedMemory(name='shm_rpml')
            shm_rpm = shared_memory.SharedMemory(name='shm_rpm')
            rpml_array = np.ndarray(shape=(max_size // 2 + 1,), dtype="f", buffer=shm_rpml.buf)
            rpm_array = np.ndarray(shape=(max_size // 2 + 1,), dtype="f", buffer=shm_rpm.buf)

            shm_vib = dict()
            for i in range(sensor_count):
                shm_vib[i] = shared_memory.SharedMemory(name="shm_vib" + str(i))

            if isResolver2:
                shm_cos = shared_memory.SharedMemory(name="shm_cos")
        elif version == 2:
            # 普通qdaq程序
            isResolver2 = (speed_signal == "resolver2")
            shm_ttl = shared_memory.SharedMemory(name='shm_ttl')
            shm_sin = shared_memory.SharedMemory(name='shm_sin')
            # speed，nvh，datapack进程是否在运行的flag
            shm_flag = shared_memory.SharedMemory(name="shm_flag")
            gv_dict_flag = shm_flag.buf
            # 保存trigger的共享内存
            # 声明转速脉冲数据的共享内存并定义数据大小（原始数据的一半）
            shm_trigger = shared_memory.SharedMemory(name="shm_trigger")
            trigger_array = np.ndarray(shape=(max_size // 2 + 1,), dtype="i", buffer=shm_trigger.buf)
            # 保存转速曲线的共享内存
            # 声明转速曲线的共享内存并定义数据大小（原始数据的一半）
            shm_rpml = shared_memory.SharedMemory(name='shm_rpml')
            shm_rpm = shared_memory.SharedMemory(name='shm_rpm')
            rpml_array = np.ndarray(shape=(max_size // 2 + 1,), dtype="f", buffer=shm_rpml.buf)
            rpm_array = np.ndarray(shape=(max_size // 2 + 1,), dtype="f", buffer=shm_rpm.buf)

            shm_vib = dict()
            for i in range(sensor_count):
                shm_vib[i] = shared_memory.SharedMemory(name="shm_vib" + str(i))

            if isResolver2:
                shm_cos = shared_memory.SharedMemory(name="shm_cos")
        elif version == 3 or version == 4:
            # 恒速电机转速波动不重采样版
            # speed，nvh，datapack进程是否在运行的flag
            shm_flag = shared_memory.SharedMemory(name="shm_flag")
            gv_dict_flag = shm_flag.buf
            shm_vib = dict()
            for i in range(sensor_count):
                shm_vib[i] = shared_memory.SharedMemory(name="shm_vib" + str(i))
        qDAQ_logger.info("speed process memory success")
    except Exception:
        qDAQ_logger.error("speed process memory error, error msg:" + traceback.format_exc())
    try:
        # 启动线程进行原始数据上传（FTP）
        ftp_queue = Queue(100)
        ftptask = Thread(target=ftp_file_upload, args=())
        ftptask.start()
    except Exception:
        qDAQ_logger.error("speed process memory error, error msg:" + traceback.format_exc())

    try:
        # 确认是否为实采模式
        if  board == "DT":
            # 实采模式启动时需要先重置板卡
            devices = get_daq_devices()
            target_device = DaqDevice(devices[0])
            device_confirm(target_device)
            target_ai_device, target_ai_info = get_ai_device_info(target_device)
            ai_info_confirm(target_ai_info)

            target_input_mode = AiInputMode.SINGLE_ENDED
            # If SINGLE_ENDED input mode is not supported, set to DIFFERENTIAL.
            target_mode_channels = get_channels_by_mode(target_ai_info, target_input_mode)
            if target_mode_channels <= 0:
                target_input_mode = AiInputMode.DIFFERENTIAL
            target_mode_channels = get_channels_by_mode(target_ai_info, target_input_mode)
            qDAQ_logger.info('DT device initial')

            @atexit.register
            def dt_disconnect():
                global status
                global target_device
                global target_ai_device

                qDAQ_logger.info('At exit function')
                if target_device:
                    # Stop the acquisition if it is still running.
                    if status == ScanStatus.RUNNING:
                        target_ai_device.scan_stop()

                    if target_device.is_connected():
                        target_device.disconnect()
                    target_device.release()
                    qDAQ_logger.info('At exit DT9837B disconnected')
        elif board == "NI":
            reset_ni_device(ni_device)
            qDAQ_logger.info('NI device reset')
    except Exception:
        qDAQ_logger.error("DT device error, error msg:" + traceback.format_exc())

    while True:
        data = Q_speed_in.get()

        running_mode="simu" if data["is_simu_mode"] else "rt"
        try:
            if version == 1:
                rawdataConsumer_for_qdaq(Q_speed_nvh_list, data['param'], data['gv_dict_status'],
                                         data['file_info'],
                                         data['start_timestamp'], data['time_click_start'])
            elif version == 2:
                rawdataConsumer_for_byd(Q_speed_nvh_list, data['param'], data['gv_dict_status'],
                                        data['file_info'],
                                        data['start_timestamp'], data['time_click_start'])
            elif version == 4:
                rawdataConsumer_for_fluctuation_norsp(Q_speed_nvh_list, data['param'],
                                                      data['gv_dict_status'], data['file_info'],
                                                      data['start_timestamp'], data['time_click_start'])
        except Exception:
            # traceback.format_exc()返回字符串,可以说traceback.print_exc()等同于print traceback.format_exc()
            qDAQ_logger.error('raw data consumer error, error msg:' + traceback.format_exc())
            data['gv_dict_status']["code"] = 3000
            data['gv_dict_status']["msg"] = "speed进程出现错误!"
            gv_dict_flag[flag_index_dict["speed-error"]]=1
        # 进程结束的信息，在cmd=4时会获取该信息（表示执行完毕一次测试）
        Q_speed_out.put("speed_process")
        del data


def rawdataConsumer_for_qdaq(Q_speed_nvh_list, param, gv_dict_status,
                             file_info, start_timestamp,
                             time_click_start):
    global gv_dict_flag, trigger_array, rpml_array, rpm_array, isResolver2, shm_vib, shm_cos, shm_speed
    global target_device, target_ai_device, target_ai_info, target_input_mode, target_mode_channels,ftp_queue
    global running_mode
    # 设置自动垃圾回收
    # gc.enable()
    # gc.set_threshold(1, 1, 1)
    qDAQ_logger.debug("rawdataConsumer:pid={},ppid={},thread={}".format(os.getpid(), os.getppid(),
                                                                        threading.current_thread().name))

    gv_dict_status_temp = dict()
    # 获取通道名称信息
    channelNames = param.taskInfo['channelNames']
    allrawdata = None

    if running_mode == "rt" and board == "NI":
        # create a task to read the raw data
        try:
            dtask = DAQTask(param.taskInfo)
            dtask.createtask()
            dtask.start_task()
        except nidaqmx.errors.DaqError:
            # if niDaq error, reset device and create task again
            qDAQ_logger.info('NI Daq error and reset device')
            dtask.stop_task()
            dtask.clear_task()
            reset_ni_device(ni_device)
            dtask = DAQTask(param.taskInfo)
            dtask.createtask()
            dtask.start_task()
        qDAQ_logger.debug("NI task start，type:{},serialNo:{}".format(param.basicInfo['type'],
                                                                     param.basicInfo['serialNo']))
    elif running_mode == "rt" and board == "DT":
        # create a task to read the raw data
        channel_count = len(channelNames)
        frame_counter = 0
        data_reading_counter = 0
        buffer_counter = 0
        buffer_factor_value, frame_data_len, buffer_len, buffer_data = buffer_creator(param.taskInfo)
        target_ai_device = set_daq_task(param.taskInfo, target_ai_info, target_ai_device,
                                        buffer_factor_value, buffer_data,
                                        target_mode_channels, target_input_mode)
        qDAQ_logger.debug("DT task start，type:{},serialNo:{}".format(param.basicInfo['type'],
                                                                     param.basicInfo['serialNo']))
        qDAQ_logger.debug("DT DAQ task start: {}".format(time.time() - time_click_start))
    elif running_mode == "simu":
        all_raw_data = read_raw_data(param.simuInfo["fileName"], channelNames, read_type)
        simu_data_len = len(all_raw_data[channelNames[0]])
        qDAQ_logger.info("time after read all simu rawdata:{}".format(time.time() - time_click_start))

    # 保存原始数据的共享内存
    gv_dict_rawdata = dict()

    if speed_signal == "ttl":
        speedChannel = "Speed"
    else:
        speedChannel = "Sin"

    # 记录原始数据（转速信号）
    gv_dict_rawdata[speedChannel] = np.ndarray(shape=(max_size,), dtype="f", buffer=shm_speed.buf,
                                               offset=0)

    # 统计并记录传感器通道
    vibChannels = param.taskInfo['sensorChan']

    for i in range(sensor_count):
        # 记录原始数据（振动信号或麦克风信号）
        gv_dict_rawdata[vibChannels[i]] = np.ndarray(shape=(max_size,), dtype="f",
                                                     buffer=shm_vib[i].buf,
                                                     offset=0)

    # channelNames中可能存在Torque扭矩信号，单路旋变中可能存在Cos信号等，这些信号只在该进程中使用
    channelNames_in_speed_process = list(set(channelNames).difference(set(vibChannels)))
    channelNames_in_speed_process.remove(speedChannel)
    if isResolver2:
        gv_dict_rawdata["Cos"] = np.ndarray(shape=(max_size,), dtype="f", buffer=shm_cos.buf, offset=0)
        channelNames_in_speed_process.remove("Cos")

    for channelName in channelNames_in_speed_process:
        # 记录除转速和传感器信号之外的信号，如torque
        gv_dict_rawdata[channelName] = np.zeros((max_size,))

    # speed calculation，初始化工况识别
    gv_dict_speedRecog = gv.set_default_speedRecog()
    # 工况评估指标
    RampQuality = None
    # 之前的方案是向nvh传递测试段内的转速曲线，现在传递索引（开始和结束索引）
    speed_start_index = 0
    speed_right_index = 0
    # 初始值（转速计算和工况识别）
    counter_angle = 0
    counter = 0
    icounter = 0  # used for speed recognition with more than 1 test
    recog_index = 0
    result_index = 0
    cdata = dict()
    fileName = param.basicInfo["fileName"]
    sampleRate = param.taskInfo["sampleRate"]
    sampsPerChan = param.taskInfo["sampsPerChan"]
    simu_sleep_time = sampsPerChan / sampleRate * sleep_ratio

    # 之前的代码在向queue中put的时候，太多if/else，
    # 每一个分支中均有put代码，下面的变量用于记录每一个分支中的数据，用于在所有if判断结束后统一put
    right_index_for_put = 0
    # 测试段结束标志
    sectionEnd_for_put = False
    testNameIndex_for_put = 0
    recogIndex_for_put = 0
    # 是否需要传递数据到nvh进程
    None_for_put = True
    temp_rpm = np.array([])
    temp_rpml = np.array([])
    last_frame_status = -1
    # 双路旋变需要的数据
    last_angle_l1f, loc_l1f = 0, None
    last_angle_l2f, loc_l2f = 0, None

    # 共享内存中存入的数组的索引，
    index_rawdata = 0
    # 保存trigger到共享内存的索引
    index_trigger = 0
    # 上一帧转速计算用到的最后一个trigger是第几个trigger
    last_rpm_cal_index = 0
    # 转速曲线保存到了第几个索引
    rpml_array[0] = 0
    rpm_array[0] = 0
    rpm_index = 1
    # 给定某一个trigger的位置，在trigger_array中寻找索引
    trigger_start_index = 0

    temptl = None
    # 是否继续进行转速计算，出错了就置为False并退出转速计算，但是依然进行数据读取
    speedCalFlag = True

    while gv_dict_flag[flag_index_dict["speedCalclation"]]:
        if counter == icounter:
            index_rawdata_backup = index_rawdata
            if running_mode == "rt" and board == "NI":
                data = dtask.read_data()
                len_rawdata_frame = len(data[0])
                for i, channelName in enumerate(channelNames):
                    gv_dict_rawdata[channelName][index_rawdata:len_rawdata_frame + index_rawdata] = \
                        data[i]
                index_rawdata += len_rawdata_frame
            elif running_mode == "rt" and board == "DT":
                status, transfer_status = target_ai_device.get_scan_status()
                next_data_length = (frame_counter + 1) * frame_data_len
                frame_num = (
                                    transfer_status.current_total_count - frame_counter * frame_data_len) // frame_data_len
                for frame_index in range(frame_num):
                    # enough data to read out
                    if transfer_status.current_total_count - frame_counter * frame_data_len <= buffer_len:
                        # no over write
                        rest_part = next_data_length % buffer_len
                        if rest_part < frame_data_len:
                            # take part of data from start and end
                            temp_data = np.concatenate(
                                (buffer_data[0:rest_part], buffer_data[-(frame_data_len - rest_part):]))
                            for chan_index in range(channel_count):
                                gv_dict_rawdata[channelNames[chan_index]][
                                index_rawdata:sampsPerChan + index_rawdata] = temp_data[
                                                                              chan_index:: channel_count]
                            # buffer_counter += 1
                        else:
                            # do not need to take data from start and end
                            start_index = (frame_counter * frame_data_len) % buffer_len
                            end_index = start_index + frame_data_len
                            for chan_index in range(channel_count):
                                gv_dict_rawdata[channelNames[chan_index]][
                                index_rawdata:sampsPerChan + index_rawdata] = buffer_data[
                                                                              start_index + chan_index: end_index: channel_count]
                        index_rawdata += sampsPerChan
                        frame_counter += 1
                    else:
                        raise ValueError("data overflow")
            else:
                if index_rawdata + sampsPerChan <= simu_data_len:
                    for channelName in channelNames:
                        cdata[channelName] = all_raw_data[channelName][
                                             counter * sampsPerChan:(counter + 1) * sampsPerChan]
                        gv_dict_rawdata[channelName][index_rawdata:index_rawdata + sampsPerChan] = \
                            cdata[channelName]
                        # 控制simu模式的数据读取速度
                        time.sleep(simu_sleep_time)
                    index_rawdata += sampsPerChan
                else:
                    gv_dict_flag[flag_index_dict["speed_finish"]]=1
            if index_rawdata_backup == index_rawdata:
                continue
            if speedCalFlag:
                if counter == 0:
                    last_angle_l1f, loc_l1f = 0, None
                    last_angle_l2f, loc_l2f = 0, None
                    last_frame_status = -1
                    left_index = 0
                try:

                    if speed_signal == "ttl":
                        # 该方法返回的location即为从检测开始时记的索引
                        temptl = trigger_detect_for_share(gv_dict_rawdata[speedChannel],
                                                          index_rawdata_backup, index_rawdata,
                                                          param.speedCalcInfo)
                        # 保存trigger信息到共享内存
                        trigger_array[
                        index_trigger:index_trigger + len(temptl)] = temptl + index_rawdata_backup
                        index_trigger += len(temptl)
                    elif speed_signal == "resolver":

                        index_trigger, left_index = single_resolver_butter_filter(
                            gv_dict_rawdata[speedChannel], left_index,
                            index_rawdata, sampleRate, param.speedCalcInfo["coils"], trigger_array,
                            index_trigger,
                            param.speedCalcInfo["ppr"])

                    elif speed_signal == "resolver2":
                        temptl, last_angle_l1f, loc_l1f, last_angle_l2f, loc_l2f, counter_angle = resolver(
                            gv_dict_rawdata['Sin'][index_rawdata_backup:index_rawdata],
                            gv_dict_rawdata['Cos'][index_rawdata_backup:index_rawdata],
                            param.speedCalcInfo['triggerLevel'],
                            param.speedCalcInfo['coils'],
                            param.speedCalcInfo['ppr'],
                            last_angle_l1f,
                            loc_l1f,
                            last_angle_l2f,
                            loc_l2f,
                            counter_angle,
                            sampsPerChan)
                        trigger_array[index_trigger:index_trigger + len(temptl)] = \
                            np.array(temptl) + index_rawdata_backup
                        index_trigger += len(temptl)

                    rpm_index_backup = rpm_index
                    last_rpm_cal_index, rpm_index = rpm_calc_for_share(trigger_array,
                                                                       last_rpm_cal_index,
                                                                       index_trigger - 1,
                                                                       sampleRate,
                                                                       param.speedCalcInfo["averageNum"],
                                                                       param.speedCalcInfo["step"],
                                                                       param.speedCalcInfo["rpmFactor"],
                                                                       rpml_array,
                                                                       rpm_array, rpm_index)
                    # 本帧的数据，用于转速识别
                    temp_rpml = rpml_array[rpm_index_backup:rpm_index]
                    temp_rpm = rpm_array[rpm_index_backup:rpm_index]
                    # 更新status中的转速数据
                    gv_dict_status_temp['data'] = gv_dict_status['data']
                    gv_dict_status_temp['data']['x'] = float(rpml_array[rpm_index - 1])
                    gv_dict_status_temp['data']['y'] = float(rpm_array[rpm_index - 1])
                    gv_dict_status['data'] = gv_dict_status_temp['data']
                except Exception:
                    gv_dict_status["code"] = 3000
                    gv_dict_status["msg"] = "转速计算错误!"
                    qDAQ_logger.error("exec failed, failed msg:" + traceback.format_exc())
                    # 转速计算错误，应该停止计算，但是数据采集应该继续进行,测试段识别结束
                    speedCalFlag = False
                    gv_dict_speedRecog['speedRecogFinish'] = True
        else:
            icounter = counter

        if not gv_dict_speedRecog['speedRecogFinish']:
            # speed recognition
            if not gv_dict_speedRecog['startFlag']:
                # detect the start point of target operating mode
                if param.speedRecogInfo["torqueRecogFlag"][recog_index]:
                    # 如果需要扭矩识别
                    gv_dict_speedRecog = speed_detect_start_with_torque(
                        temp_rpml, temp_rpm, rpm_array[:rpm_index],
                        gv_dict_rawdata['Torque'][index_rawdata_backup:index_rawdata],
                        recog_index,
                        gv_dict_speedRecog,
                        param.speedRecogInfo,
                        sampleRate, counter,
                        sampsPerChan)
                else:
                    # 不需要扭矩识别
                    gv_dict_speedRecog = speed_detect_start(temp_rpml, temp_rpm,
                                                            rpm_array[:rpm_index],
                                                            recog_index,
                                                            gv_dict_speedRecog,
                                                            param.speedRecogInfo)

                # special case: start point and end point in same frame
                if gv_dict_speedRecog['startFlag']:
                    # 记录开始点识别时间
                    qDAQ_logger.debug("test index: {} detect start point at: {}".format(recog_index,
                                                                                        time.time() - time_click_start))
                    # 只有第一次检测到起点才会到这里（确保开始点和结束点在同一帧，一般不会）
                    speed_start_index = rpm_index_backup
                    speed_right_index = rpm_index
                    # 转速曲线上的时间点对应average段的最后一个点
                    # 例 average=5 overlap=3 step=average-overlap=2
                    # 转速曲线索引0 对应 trigger索引5
                    # 转速曲线索引1 对应 trigger索引7
                    trigger_start_index = (gv_dict_speedRecog['startpoint_index'] - 1) * \
                                          param.speedCalcInfo["step"] + \
                                          param.speedCalcInfo["averageNum"]
                    vib_start_index = trigger_array[trigger_start_index]
                    if param.speedRecogInfo["torqueRecogFlag"][recog_index]:
                        gv_dict_speedRecog = speed_detect_end_with_torque(
                            temp_rpm, rpml_array[:rpm_index],
                            rpm_array[:rpm_index],
                            gv_dict_rawdata['Torque'][:index_rawdata],
                            recog_index, gv_dict_speedRecog,
                            param.speedRecogInfo,
                            sampleRate)
                    else:
                        gv_dict_speedRecog = speed_detect_end(temp_rpm,
                                                              rpml_array[:rpm_index],
                                                              rpm_array[:rpm_index],
                                                              recog_index,
                                                              gv_dict_speedRecog,
                                                              param.speedRecogInfo)
                    if gv_dict_speedRecog['endpoint_loc']:
                        """
                        1. once detect the start point put the data into queue, but need to remark this is for which
                        target operating mode
                        2. if this is the first part and detect the end point, take the part of data and put it into
                        the queue
                        """
                        start_index = int(gv_dict_speedRecog['startpoint_loc'] *
                                          sampleRate)
                        end_index = int(
                            gv_dict_speedRecog['endpoint_loc'] * sampleRate)
                        # to check if the test is dummy
                        if param.speedRecogInfo["notDummyFlag"][recog_index]:
                            right_index_for_put = end_index
                            sectionEnd_for_put = True
                            testNameIndex_for_put = result_index
                            recogIndex_for_put = recog_index
                            None_for_put = False
                            startpoint_index = gv_dict_speedRecog['startpoint_index']
                            endpoint_index = gv_dict_speedRecog['endpoint_index']
                            # 计算工况评估指标
                            RampQuality = ramp_quality(rpml_array[startpoint_index:endpoint_index],
                                                       rpm_array[startpoint_index:endpoint_index],
                                                       param.speedRecogInfo, recog_index)

                            # result index is for final result
                            # dummy段不会进行计数
                            result_index += 1
                            # update status
                            gv_dict_status["msg"] = param.speedRecogInfo["testName"][
                                                        recog_index] + "测试段识别完成！"
                            gv_dict_status_temp['data'] = gv_dict_status['data']
                            gv_dict_status_temp["data"]["startX"].append(
                                float(gv_dict_speedRecog["startpoint_loc"]))
                            gv_dict_status_temp["data"]["startY"].append(
                                float(gv_dict_speedRecog["startpoint_speed"]))
                            gv_dict_status_temp["data"]["endX"].append(
                                float(gv_dict_speedRecog["endpoint_loc"]))
                            gv_dict_status_temp["data"]["endY"].append(
                                float(gv_dict_speedRecog["endpoint_speed"]))
                            gv_dict_status_temp["data"]["testName"].append(
                                param.speedRecogInfo["testName"][recog_index])
                            gv_dict_status['data'] = gv_dict_status_temp['data']
                        else:
                            # dummy段不传递数据，但是需要跳过并进行下一个测试段识别
                            sectionEnd_for_put = True

                        # recog index is for test name list of speed recognition
                        recog_index = recog_index + 1

                        if recog_index < len(param.speedRecogInfo['testName']):
                            # set back the parameters for speed recognition if test not finished
                            # 从当前帧开始进行下一个测试段的识别
                            counter = counter - 1
                        else:
                            # 表示工况识别已完成
                            gv_dict_status['code'] = 2
                            gv_dict_status['msg'] = '转速识别完成!'
                            qDAQ_logger.info(
                                'speed recog finished at: ' + datetime.datetime.now().strftime(
                                    "%Y%m%d%H%M%S.%f"))
                            gv_dict_speedRecog['speedRecogFinish'] = True
                    else:
                        # if it's the first part but haven't detect the end,
                        # put part of the data in the queue
                        # 表示已识别到开始点，但是未识别到结束点，所以需要将部分数据传递给nvh进行分析
                        start_index = int(gv_dict_speedRecog['startpoint_loc'] * sampleRate)

                        if param.speedRecogInfo["notDummyFlag"][recog_index]:
                            # 非dummy段的数据传递到nvh进行计算
                            right_index_for_put = index_rawdata
                            sectionEnd_for_put = False
                            testNameIndex_for_put = result_index
                            recogIndex_for_put = recog_index
                            None_for_put = False
                else:
                    # 没有识别到开始点就不传递数据
                    None_for_put = True
            else:
                speed_right_index = rpm_index
                # detect the start point of target operating mode(not the first frame)
                if param.speedRecogInfo["torqueRecogFlag"][recog_index]:
                    gv_dict_speedRecog = speed_detect_end_with_torque(temp_rpm,
                                                                      rpml_array[
                                                                      :rpm_index],
                                                                      rpm_array[
                                                                      :rpm_index],
                                                                      gv_dict_rawdata[
                                                                          'Torque'][
                                                                      :index_rawdata],
                                                                      recog_index,
                                                                      gv_dict_speedRecog,
                                                                      param.speedRecogInfo,
                                                                      sampleRate)
                else:
                    gv_dict_speedRecog = speed_detect_end(temp_rpm,
                                                          rpml_array[:rpm_index],
                                                          rpm_array[:rpm_index],
                                                          recog_index,
                                                          gv_dict_speedRecog,
                                                          param.speedRecogInfo)
                if gv_dict_speedRecog['endpoint_loc']:
                    # if it's not the first part and have detected the end point,
                    # take part of data and put it into the queue
                    end_index = int(gv_dict_speedRecog['endpoint_loc'] * sampleRate)
                    start_index = end_index - (end_index % sampsPerChan)
                    if param.speedRecogInfo["notDummyFlag"][recog_index]:
                        right_index_for_put = end_index
                        sectionEnd_for_put = True
                        testNameIndex_for_put = result_index
                        recogIndex_for_put = recog_index
                        None_for_put = False

                        startpoint_index = gv_dict_speedRecog['startpoint_index']
                        endpoint_index = gv_dict_speedRecog['endpoint_index']
                        # 计算工况评估指标
                        RampQuality = ramp_quality(rpml_array[startpoint_index:endpoint_index],
                                                   rpm_array[startpoint_index:endpoint_index],
                                                   param.speedRecogInfo, recog_index)

                        result_index += 1
                        # put speed recognition
                        gv_dict_status["msg"] =param.speedRecogInfo["testName"][recog_index] + "测试段识别完成！"
                        gv_dict_status_temp['data'] = gv_dict_status['data']
                        gv_dict_status_temp["data"]["startX"].append(
                            float(gv_dict_speedRecog["startpoint_loc"]))
                        gv_dict_status_temp["data"]["startY"].append(
                            float(gv_dict_speedRecog["startpoint_speed"]))
                        gv_dict_status_temp["data"]["endX"].append(
                            float(gv_dict_speedRecog["endpoint_loc"]))
                        gv_dict_status_temp["data"]["endY"].append(
                            float(gv_dict_speedRecog["endpoint_speed"]))
                        gv_dict_status_temp["data"]["testName"].append(
                            param.speedRecogInfo["testName"][recog_index])
                        gv_dict_status['data'] = gv_dict_status_temp['data']
                    else:
                        sectionEnd_for_put = True

                    recog_index = recog_index + 1
                    if recog_index < len(param.speedRecogInfo['testName']):
                        counter = counter - 1
                    else:
                        gv_dict_status["code"] = 2
                        gv_dict_status['msg'] = '转速识别完成!'
                        qDAQ_logger.info('speed recog finished' + datetime.datetime.now().strftime(
                            "%Y%m%d%H%M%S.%f"))
                        gv_dict_speedRecog['speedRecogFinish'] = True
                        # 这里之前的代码会put None,由于测试段识别已经结束，之前的代码这里put None不会出错，
                        # 修改之后由于需要统一put，这里不能添加None_for_put=True
                else:
                    # if it's not the first part and havn't detect the end point,
                    # put the data of this frame into the queue
                    if param.speedRecogInfo["notDummyFlag"][recog_index]:
                        # 要将该次传递进来的数据全部放入，
                        right_index_for_put = index_rawdata
                        sectionEnd_for_put = False
                        testNameIndex_for_put = result_index
                        recogIndex_for_put = recog_index
                        None_for_put = False

            if not None_for_put:
                qdata = {
                    # 测试段开始点vib信号的索引，一个测试段内该值不变
                    'vib_start_index': vib_start_index,
                    # 测试段内原始数据的右端索引，每次put会更新该值,注意右端索引处是没有值的
                    'vib_right_index': right_index_for_put,
                    # 测试段开始时的trigger在trigger_array中的位置，一个测试段内，该值不变
                    'trigger_start_index': trigger_start_index,
                    # 已经计算到了第几个trigger
                    # 'trigger_right_index':rpm_index*param.speedCalcInfo["step"]+param.speedCalcInfo["averageNum"],
                    'trigger_right_index': index_trigger if not sectionEnd_for_put else
                    (gv_dict_speedRecog['endpoint_index'] - 1) * param.speedCalcInfo["step"] +
                    param.speedCalcInfo["averageNum"],
                    # 测试段是否结束
                    'sectionEnd': sectionEnd_for_put,
                    # 有dummy段时，dummy可有多个
                    # 这个索引是去除dummy的索引
                    'testNameIndex': testNameIndex_for_put,
                    # 这个索引是包括dummy的索引
                    'recogIndex': recogIndex_for_put,
                    # 传给nvh的测试段内转速曲线的起始点索引，该段转速曲线的长度比测试段要长，一个测试段内该值不变
                    "speed_start_index": speed_start_index,
                    # 传给nvh的测试段内转速曲线的结束点索引，该段转速曲线的长度比测试段要长，每次put会更新该值
                    "speed_right_index": speed_right_index,
                    # if sectionEnd_for_put 测试段最后一帧才需要speedPattern
                    'speedPattern': param.speedRecogInfo['speedPattern'][
                        recogIndex_for_put],
                    'RampQuality': RampQuality,
                    "ppr": param.speedCalcInfo["ppr"],
                    "speedRatio": param.speedCalcInfo["speedRatio"]
                }

                # 将数据放入queue中
                for Q_speed_nvh in Q_speed_nvh_list:
                    Q_speed_nvh.put(qdata)

                if sectionEnd_for_put:
                    qDAQ_logger.debug("测试段{}识别结束:{}".format(recogIndex_for_put, time.time()))
                    qDAQ_logger.info("test name：{} recognition finished".format(
                        param.speedRecogInfo['testName'][recogIndex_for_put]))
                    qDAQ_logger.info("test name：{} recognition finished".format(
                        param.speedRecogInfo['testName'][recogIndex_for_put]))
                    qDAQ_logger.debug(qdata)
                    qDAQ_logger.debug(gv_dict_speedRecog)
                    gv_dict_speedRecog = gv.set_default_speedRecog()
                    # dummy不可能是最后一段,所有测试段识别完成后将flag置为true，
                    # set_default_speedRecog会将flag置为false
                    if testNameIndex_for_put + 1 == param.speedRecogInfo["test_count_except_dummy"]:
                        qDAQ_logger.info("speed process consumer section end")
                        qDAQ_logger.info("speed process consumer section end :{}".format(
                            time.time() - time_click_start))
                        gv_dict_speedRecog['speedRecogFinish'] = True
                    sectionEnd_for_put = False

        else:
            # recognize finished, need to confirm if some error
            if recog_index < len(param.speedRecogInfo['testName']):
                gv_dict_status["code"] = 3000
                gv_dict_status['msg'] = '转速识别失败，请确认转速曲线!'

        counter = counter + 1
        icounter = icounter + 1

    if running_mode == "rt" and board == "NI":
        dtask.stop_task()
        dtask.clear_task()
        qDAQ_logger.info("NI task stopped!")
    elif running_mode == "rt" and board == "DT":
        target_ai_device.scan_stop()
        qDAQ_logger.info("DT task stopped!")
    try:
        if param.dataSaveFlag["speedCurve"]:
            if save_type == "hdf5":
                speed_result_filename = os.path.join(param.folderInfo["temp"],
                                                     param.basicInfo["type"] + '_' + fileName + '.h5')
                write_hdf5(speed_result_filename, 'speedData', 'speedLoc', rpml_array[:rpm_index])
                write_hdf5(speed_result_filename, 'speedData', 'speedValue', rpm_array[:rpm_index])
                write_hdf5(speed_result_filename, 'trigger', 'trigger', trigger_array[:index_trigger])
            else:
                speed_result_filename = os.path.join(param.folderInfo["temp"],
                                                     param.basicInfo["type"] + '_' + fileName + '.tdms')
                write_tdms(speed_result_filename, 'speedData', 'speedLoc', rpml_array[:rpm_index])
                write_tdms(speed_result_filename, 'speedData', 'speedValue', rpm_array[:rpm_index])
                write_tdms(speed_result_filename, 'trigger', 'trigger', trigger_array[:index_trigger])
            qDAQ_logger.info('speed curve saved, filename: ' + speed_result_filename)
        else:
            qDAQ_logger.info('speed curve not saved!')
    except Exception:
        gv_dict_status["code"] = 3000
        gv_dict_status["msg"] = "转速曲线保存失败!"
        qDAQ_logger.error("speed curve save failed, failed msg:" + traceback.format_exc())

    try:
        if param.dataSaveFlag["rawData"]:
            rawData_length = index_rawdata
            single_folder_confirm(file_info[0])
            # update raw data save properties
            raw_data_save_properties = create_properities(start_timestamp, 1 / sampleRate, sampsPerChan)
            if save_type == "hdf5":
                file_name = fileName + '.h5'
                raw_data_filename = os.path.join(file_info[0], fileName + '.h5')
                # 保存原始数据
                for i in range(len(param.taskInfo['channelNames'])):
                    raw_data_save_properties['NI_ChannelName'] = param.taskInfo['channelNames'][i]
                    raw_data_save_properties['NI_UnitDescription'] = param.taskInfo['units'][i]
                    write_hdf5(raw_data_filename, 'AIData',
                               param.taskInfo['channelNames'][i],
                               gv_dict_rawdata[param.taskInfo['channelNames'][i]][:rawData_length],
                               raw_data_save_properties)
                # 保存转速曲线和trigger信息
                write_hdf5(raw_data_filename, 'speedData', 'speedLoc', rpml_array[:rpm_index])
                write_hdf5(raw_data_filename, 'speedData', 'speedValue', rpm_array[:rpm_index])
                write_hdf5(raw_data_filename, 'triggerData', 'Trigger', trigger_array[:index_trigger])
            else:
                file_name = fileName + '.tdms'
                raw_data_filename = os.path.join(file_info[0], fileName + '.tdms')
                # 保存原始数据
                for i in range(len(param.taskInfo['channelNames'])):
                    raw_data_save_properties['NI_ChannelName'] = param.taskInfo['channelNames'][i]
                    raw_data_save_properties['NI_UnitDescription'] = param.taskInfo['units'][i]
                    write_tdms(raw_data_filename, 'AIData',
                               param.taskInfo['channelNames'][i],
                               gv_dict_rawdata[param.taskInfo['channelNames'][i]][:rawData_length],
                               raw_data_save_properties)
                # 保存转速曲线和trigger信息
                write_tdms(raw_data_filename, 'speedData', 'speedLoc', rpml_array[:rpm_index])
                write_tdms(raw_data_filename, 'speedData', 'speedValue', rpm_array[:rpm_index])
                write_tdms(raw_data_filename, 'triggerData', 'Trigger', trigger_array[:index_trigger])
            qDAQ_logger.info("raw data saved, filename: " + raw_data_filename)

            # 原始数据保存成功，传入参数开始执行数据上传和请求解析
            url_info = param.ftpUploadInfo
            folder_info = param.folderInfo
            folder_info["localFolder"] = file_info[0]
            data_info = dict()
            data_info['fileName'] = file_name
            data_info['system'] = param.basicInfo['systemNo']
            data_info['type'] = param.basicInfo['type']
            data_info['serial'] = param.basicInfo['serialNo']
            data_info['timestamp'] = timestamp_to_time(start_timestamp)
            format_timestamp = start_timestamp.strftime("%Y%m%d%H%M%S")
            ftp_queue.put({"urlInfo": url_info, "folderInfo": folder_info, "dataInfo": data_info,
                           "formatTimeStamp": format_timestamp})
            qDAQ_logger.debug("ftp upload request send out")

        else:
            qDAQ_logger.info("raw data not saved!")
    except Exception:
        gv_dict_status["code"] = 3000
        gv_dict_status["msg"] = "原始数据保存失败!"
        qDAQ_logger.error("raw data save failed, failed msg:" + traceback.format_exc())

    del cdata
    del gv_dict_rawdata
    del gv_dict_speedRecog
    del counter_angle
    del counter
    del icounter
    del recog_index
    del result_index
    del channelNames
    del right_index_for_put
    del sectionEnd_for_put
    del testNameIndex_for_put
    del recogIndex_for_put
    del None_for_put
    del raw_data_filename
    del raw_data_save_properties
    del temptl
    del temp_rpm
    del temp_rpml
    del param
    del file_info
    del start_timestamp
    del time_click_start
    # gc.collect()


def rawdataConsumer_for_byd(Q_speed_nvh_list, param, gv_dict_status,
                            file_info, start_timestamp,
                            time_click_start):
    global gv_dict_flag, trigger_array, rpml_array, rpm_array, isResolver2, shm_vib, shm_cos, shm_speed, \
        shm_sin, shm_ttl
    global target_device, target_ai_device, target_ai_info, target_input_mode, target_mode_channels,ftp_queue
    global running_mode
    # 设置自动垃圾回收
    # gc.enable()
    # gc.set_threshold(1, 1, 1)
    qDAQ_logger.debug("rawdataConsumer:pid={},ppid={},thread={}".format(os.getpid(), os.getppid(),
                                                                        threading.current_thread().name))

    gv_dict_status_temp = dict()
    # 获取通道名称信息
    channelNames = param.taskInfo['channelNames']
    allrawdata = None

    if running_mode == "rt" and board == "NI":
        # create a task to read the raw data
        try:
            dtask = DAQTask(param.taskInfo)
            dtask.createtask()
            dtask.start_task()
        except nidaqmx.errors.DaqError:
            # if niDaq error, reset device and create task again
            qDAQ_logger.info('NI Daq error and reset device')
            dtask.stop_task()
            dtask.clear_task()
            reset_ni_device(ni_device)
            dtask = DAQTask(param.taskInfo)
            dtask.createtask()
            dtask.start_task()
        qDAQ_logger.debug("NI task start，type:{},serialNo:{}".format(param.basicInfo['type'],
                                                                     param.basicInfo['serialNo']))
    elif running_mode == "rt" and board == "DT":
        # create a task to read the raw data
        channel_count = len(channelNames)
        frame_counter = 0
        data_reading_counter = 0
        buffer_counter = 0
        buffer_factor_value, frame_data_len, buffer_len, buffer_data = buffer_creator(param.taskInfo)
        target_ai_device = set_daq_task(param.taskInfo, target_ai_info, target_ai_device,
                                        buffer_factor_value, buffer_data,
                                        target_mode_channels, target_input_mode)
        qDAQ_logger.debug("DT task start，type:{},serialNo:{}".format(param.basicInfo['type'],
                                                                     param.basicInfo['serialNo']))
        qDAQ_logger.debug("DT DAQ task start: {}".format(time.time() - time_click_start))
    elif running_mode == "simu":
        all_raw_data = read_raw_data(param.simuInfo["fileName"], channelNames, read_type)
        simu_data_len = len(all_raw_data[channelNames[0]])
        qDAQ_logger.info("time after read all simu rawdata:{}".format(time.time() - time_click_start))

    # 保存原始数据的共享内存
    gv_dict_rawdata = dict()

    # ttl信号的ppr
    ppr_for_ttl = param.speedCalcInfo["ppr"]
    # 旋变信号的ppr，包括单路旋变和双路旋变
    ppr_for_resolver = param.speedCalcInfo["pprForBydResolver"]
    triggerLevel_for_resolver2 = param.speedCalcInfo["triggerLevelForBydResolver"]

    # 参数配置中的speed_signal决定是否开启Cos通道，即isResolver2变量，这里更新speed_signal
    # 在切换的测试段识别结束后再次切换
    speed_signal = param.speedCalcInfo["signalBeforeSwitch"]

    if speed_signal == "ttl":
        speedChannel = "Speed"
        ppr = param.speedCalcInfo["ppr"]
        triggerLevel = param.speedCalcInfo["triggerLevel"]
        overlap = param.speedCalcInfo["overlap"]
        averageNum = param.speedCalcInfo["averageNum"]
    else:
        speedChannel = "Sin"
        ppr = param.speedCalcInfo["pprForBydResolver"]
        triggerLevel = param.speedCalcInfo["triggerLevelForBydResolver"]
        overlap = param.speedCalcInfo["overlapForBydResolver"]
        averageNum = param.speedCalcInfo["averageNumForBydResolver"]

    # 切换之前的转速比，速度乘以该转速比得到观察轴的速度，阶次除以该转速比得到观察轴的阶次
    speedRatio = param.speedCalcInfo["speedRatioForByd"][0]

    gv_dict_rawdata["Speed"] = np.ndarray(shape=(max_size,), dtype="f", buffer=shm_ttl.buf, offset=0)
    gv_dict_rawdata["Sin"] = np.ndarray(shape=(max_size,), dtype="f", buffer=shm_sin.buf, offset=0)

    # 统计并记录传感器通道
    vibChannels = param.taskInfo['sensorChan']

    for i in range(sensor_count):
        # 记录原始数据（振动信号或麦克风信号）
        gv_dict_rawdata[vibChannels[i]] = np.ndarray(shape=(max_size,), dtype="f",
                                                     buffer=shm_vib[i].buf,
                                                     offset=0)

    # channelNames中可能存在Torque扭矩信号，单路旋变中可能存在Cos信号等，这些信号只在该进程中使用
    channelNames_in_speed_process = list(set(channelNames).difference(set(vibChannels)))
    channelNames_in_speed_process.remove(speedChannel)
    if isResolver2:
        gv_dict_rawdata["Cos"] = np.ndarray(shape=(max_size,), dtype="f", buffer=shm_cos.buf, offset=0)
        channelNames_in_speed_process.remove("Cos")

    for channelName in channelNames_in_speed_process:
        # 记录除转速和传感器信号之外的信号，如torque
        gv_dict_rawdata[channelName] = np.zeros((max_size,))

    # speed calculation，初始化工况识别
    gv_dict_speedRecog = gv.set_default_speedRecog()
    # 工况评估指标
    RampQuality = None
    # 之前的方案是向nvh传递测试段内的转速曲线，现在传递索引（开始和结束索引）
    speed_start_index = 0
    speed_right_index = 0
    # 初始值（转速计算和工况识别）
    counter_angle = 0
    counter = 0
    icounter = 0  # used for speed recognition with more than 1 test
    recog_index = 0
    result_index = 0
    cdata = dict()
    fileName = param.basicInfo["fileName"]
    sampleRate = param.taskInfo["sampleRate"]
    sampsPerChan = param.taskInfo["sampsPerChan"]
    simu_sleep_time = sampsPerChan / sampleRate * sleep_ratio

    # 之前的代码在向queue中put的时候，太多if/else，
    # 每一个分支中均有put代码，下面的变量用于记录每一个分支中的数据，用于在所有if判断结束后统一put
    right_index_for_put = 0
    # 测试段结束标志
    sectionEnd_for_put = False
    testNameIndex_for_put = 0
    recogIndex_for_put = 0
    # 是否需要传递数据到nvh进程
    None_for_put = True
    temp_rpm = np.array([])
    temp_rpml = np.array([])
    last_frame_status = -1
    # 双路旋变需要的数据
    last_angle_l1f, loc_l1f = 0, None
    last_angle_l2f, loc_l2f = 0, None

    # 共享内存中存入的数组的索引，
    index_rawdata = 0
    # 保存trigger到共享内存的索引
    index_trigger = 0
    # 上一帧转速计算用到的最后一个trigger是第几个trigger
    last_rpm_cal_index = 0
    # 转速曲线保存到了第几个索引
    rpml_array[0] = 0
    rpm_array[0] = 0
    rpm_index = 1
    # 给定某一个trigger的位置，在trigger_array中寻找索引
    trigger_start_index = 0

    temptl = None
    # 是否继续进行转速计算，出错了就置为False并退出转速计算，但是依然进行数据读取
    speedCalFlag = True

    # 是否是第一次计算转速，由于byd转速信号发生改变，改变后要将该值置为True
    is_first_speed_calc = True

    # 记录上一次转速信号计算到哪一个trigger以及rpm,该index处没有值，
    # 即为下一个转速信号的0位置
    index_trigger_last_speed_signal = index_trigger
    index_rpm_last_speed_signal = rpm_index

    while gv_dict_flag[flag_index_dict["speedCalclation"]]:
        if counter == icounter:
            index_rawdata_backup = index_rawdata
            if running_mode == "rt" and board == "NI":
                data = dtask.read_data()
                len_rawdata_frame = len(data[0])
                for i, channelName in enumerate(channelNames):
                    gv_dict_rawdata[channelName][index_rawdata:len_rawdata_frame + index_rawdata] = \
                        data[i]
                index_rawdata += len_rawdata_frame
            elif running_mode == "rt" and board == "DT":
                status, transfer_status = target_ai_device.get_scan_status()
                next_data_length = (frame_counter + 1) * frame_data_len
                frame_num = (
                                        transfer_status.current_total_count - frame_counter * frame_data_len) // frame_data_len
                for frame_index in range(frame_num):
                    # enough data to read out
                    if transfer_status.current_total_count - frame_counter * frame_data_len <= buffer_len:
                        # no over write
                        rest_part = next_data_length % buffer_len
                        if rest_part < frame_data_len:
                            # take part of data from start and end
                            temp_data = np.concatenate(
                                (buffer_data[0:rest_part], buffer_data[-(frame_data_len - rest_part):]))
                            for chan_index in range(channel_count):
                                gv_dict_rawdata[channelNames[chan_index]][
                                index_rawdata:sampsPerChan + index_rawdata] = temp_data[
                                                                              chan_index:: channel_count]
                            # buffer_counter += 1
                        else:
                            # do not need to take data from start and end
                            start_index = (frame_counter * frame_data_len) % buffer_len
                            end_index = start_index + frame_data_len
                            for chan_index in range(channel_count):
                                gv_dict_rawdata[channelNames[chan_index]][
                                index_rawdata:sampsPerChan + index_rawdata] = buffer_data[
                                                                              start_index + chan_index: end_index: channel_count]
                        index_rawdata += sampsPerChan
                        frame_counter += 1
                    else:
                        raise ValueError("data overflow")
            else:
                if index_rawdata + sampsPerChan <= simu_data_len:
                    for channelName in channelNames:
                        cdata[channelName] = all_raw_data[channelName][
                                             counter * sampsPerChan:(counter + 1) * sampsPerChan]
                        gv_dict_rawdata[channelName][index_rawdata:index_rawdata + sampsPerChan] = \
                            cdata[channelName]
                        # 控制simu模式的数据读取速度
                        time.sleep(simu_sleep_time)
                    index_rawdata += sampsPerChan
                else:
                    gv_dict_flag[flag_index_dict["speed_finish"]] = 1
            if index_rawdata_backup == index_rawdata:
                continue
            if speedCalFlag:
                if counter == 0:
                    last_angle_l1f, loc_l1f = 0, None
                    last_angle_l2f, loc_l2f = 0, None
                    last_frame_status = -1
                    left_index = 0
                try:

                    if speed_signal == "ttl":
                        # 该方法返回的location即为从检测开始时记的索引
                        temptl = trigger_detect_for_share(gv_dict_rawdata[speedChannel],
                                                          index_rawdata_backup, index_rawdata,
                                                          param.speedCalcInfo)
                        # 保存trigger信息到共享内存
                        trigger_array[
                        index_trigger:index_trigger + len(temptl)] = temptl + index_rawdata_backup
                        index_trigger += len(temptl)
                    elif speed_signal == "resolver":
                        index_trigger, left_index = single_resolver_butter_filter(
                            gv_dict_rawdata[speedChannel], left_index, index_rawdata, sampleRate,
                            param.speedCalcInfo["coils"], trigger_array, index_trigger, ppr)
                    elif speed_signal == "resolver2":
                        temptl, last_angle_l1f, loc_l1f, last_angle_l2f, loc_l2f, counter_angle = resolver(
                            gv_dict_rawdata['Sin'][index_rawdata_backup:index_rawdata],
                            gv_dict_rawdata['Cos'][index_rawdata_backup:index_rawdata],
                            triggerLevel_for_resolver2,
                            param.speedCalcInfo['coils'],
                            ppr_for_resolver,
                            last_angle_l1f,
                            loc_l1f,
                            last_angle_l2f,
                            loc_l2f,
                            counter_angle,
                            sampsPerChan)
                        trigger_array[index_trigger:index_trigger + len(temptl)] = \
                            np.array(temptl) + index_rawdata_backup
                        index_trigger += len(temptl)

                    rpm_index_backup = rpm_index
                    last_rpm_cal_index, rpm_index, is_first_speed_calc = rpm_calc_for_share_for_byd(
                        trigger_array,
                        last_rpm_cal_index,
                        index_trigger - 1,
                        sampleRate,
                        averageNum,
                        averageNum - overlap,
                        60 * averageNum / ppr,
                        rpml_array,
                        rpm_array, rpm_index,
                        is_first_speed_calc)
                    # 本帧的数据，用于转速识别
                    temp_rpml = rpml_array[rpm_index_backup:rpm_index]
                    temp_rpm = rpm_array[rpm_index_backup:rpm_index]
                    # 更新status中的转速数据
                    gv_dict_status_temp['data'] = gv_dict_status['data']
                    gv_dict_status_temp['data']['x'] = float(rpml_array[rpm_index - 1])
                    gv_dict_status_temp['data']['y'] = float(rpm_array[rpm_index - 1])
                    gv_dict_status['data'] = gv_dict_status_temp['data']
                except Exception:
                    gv_dict_status["code"] = 3000
                    gv_dict_status["msg"] = "转速计算错误!"
                    qDAQ_logger.error("exec failed, failed msg:" + traceback.format_exc())
                    # 转速计算错误，应该停止计算，但是数据采集应该继续进行,测试段识别结束
                    speedCalFlag = False
                    gv_dict_speedRecog['speedRecogFinish'] = True
        else:
            icounter = counter

        if not gv_dict_speedRecog['speedRecogFinish']:
            # speed recognition
            if not gv_dict_speedRecog['startFlag']:
                # detect the start point of target operating mode
                if param.speedRecogInfo["torqueRecogFlag"][recog_index]:
                    # 如果需要扭矩识别
                    gv_dict_speedRecog = speed_detect_start_with_torque(
                        temp_rpml, temp_rpm, rpm_array[:rpm_index],
                        gv_dict_rawdata['Torque'][index_rawdata_backup:index_rawdata],
                        recog_index,
                        gv_dict_speedRecog,
                        param.speedRecogInfo,
                        sampleRate, counter,
                        sampsPerChan)
                else:
                    # 不需要扭矩识别
                    gv_dict_speedRecog = speed_detect_start(temp_rpml, temp_rpm,
                                                            rpm_array[:rpm_index],
                                                            recog_index,
                                                            gv_dict_speedRecog,
                                                            param.speedRecogInfo)

                # special case: start point and end point in same frame
                if gv_dict_speedRecog['startFlag']:
                    if recog_index == 1:
                        time.sleep(0.1)
                    # 记录开始点识别时间
                    qDAQ_logger.debug("test index: {} detect start point at: {}".format(recog_index,
                                                                                        time.time() - time_click_start))
                    # 只有第一次检测到起点才会到这里（确保开始点和结束点在同一帧，一般不会）
                    speed_start_index = rpm_index_backup
                    speed_right_index = rpm_index
                    # 转速曲线上的时间点对应average段的最后一个点
                    # 例 average=5 overlap=3 step=average-overlap=2
                    # 转速曲线索引0 对应 trigger索引5
                    # 转速曲线索引1 对应 trigger索引7
                    trigger_start_index = (gv_dict_speedRecog[
                                               'startpoint_index'] - index_rpm_last_speed_signal) * \
                                          (
                                                  averageNum - overlap) + averageNum + index_trigger_last_speed_signal
                    vib_start_index = trigger_array[trigger_start_index]
                    if param.speedRecogInfo["torqueRecogFlag"][recog_index]:
                        gv_dict_speedRecog = speed_detect_end_with_torque(
                            temp_rpm, rpml_array[:rpm_index],
                            rpm_array[:rpm_index],
                            gv_dict_rawdata['Torque'][:index_rawdata],
                            recog_index, gv_dict_speedRecog,
                            param.speedRecogInfo,
                            sampleRate)
                    else:
                        gv_dict_speedRecog = speed_detect_end(temp_rpm,
                                                              rpml_array[:rpm_index],
                                                              rpm_array[:rpm_index],
                                                              recog_index,
                                                              gv_dict_speedRecog,
                                                              param.speedRecogInfo)
                    if gv_dict_speedRecog['endpoint_loc']:
                        """
                        1. once detect the start point put the data into queue, but need to remark this is for which
                        target operating mode
                        2. if this is the first part and detect the end point, take the part of data and put it into
                        the queue
                        """
                        start_index = int(gv_dict_speedRecog['startpoint_loc'] *
                                          sampleRate)
                        end_index = int(
                            gv_dict_speedRecog['endpoint_loc'] * sampleRate)
                        # to check if the test is dummy
                        if param.speedRecogInfo["notDummyFlag"][recog_index]:
                            right_index_for_put = end_index
                            sectionEnd_for_put = True
                            testNameIndex_for_put = result_index
                            recogIndex_for_put = recog_index
                            None_for_put = False
                            startpoint_index = gv_dict_speedRecog['startpoint_index']
                            endpoint_index = gv_dict_speedRecog['endpoint_index']
                            # 计算工况评估指标
                            RampQuality = ramp_quality(rpml_array[startpoint_index:endpoint_index],
                                                       rpm_array[startpoint_index:endpoint_index],
                                                       param.speedRecogInfo, recog_index)

                            # result index is for final result
                            # dummy段不会进行计数
                            result_index += 1
                            # update status
                            gv_dict_status["msg"] = param.speedRecogInfo["testName"][
                                                        recog_index] + "测试段识别完成！"
                            gv_dict_status_temp['data'] = gv_dict_status['data']
                            gv_dict_status_temp["data"]["startX"].append(
                                float(gv_dict_speedRecog["startpoint_loc"]))
                            gv_dict_status_temp["data"]["startY"].append(
                                float(gv_dict_speedRecog["startpoint_speed"]))
                            gv_dict_status_temp["data"]["endX"].append(
                                float(gv_dict_speedRecog["endpoint_loc"]))
                            gv_dict_status_temp["data"]["endY"].append(
                                float(gv_dict_speedRecog["endpoint_speed"]))
                            gv_dict_status_temp["data"]["testName"].append(
                                param.speedRecogInfo["testName"][recog_index])
                            gv_dict_status['data'] = gv_dict_status_temp['data']
                        else:
                            # dummy段不传递数据，但是需要跳过并进行下一个测试段识别
                            sectionEnd_for_put = True

                        # recog index is for test name list of speed recognition
                        recog_index = recog_index + 1

                        if recog_index < len(param.speedRecogInfo['testName']):
                            # set back the parameters for speed recognition if test not finished
                            # 从当前帧开始进行下一个测试段的识别
                            counter = counter - 1
                        else:
                            # 表示工况识别已完成
                            gv_dict_status['code'] = 2
                            gv_dict_status['msg'] = '转速识别完成!'
                            qDAQ_logger.info(
                                'speed recog finished at: ' + datetime.datetime.now().strftime(
                                    "%Y%m%d%H%M%S.%f"))
                            gv_dict_speedRecog['speedRecogFinish'] = True
                    else:
                        # if it's the first part but haven't detect the end,
                        # put part of the data in the queue
                        # 表示已识别到开始点，但是未识别到结束点，所以需要将部分数据传递给nvh进行分析
                        start_index = int(gv_dict_speedRecog['startpoint_loc'] * sampleRate)

                        if param.speedRecogInfo["notDummyFlag"][recog_index]:
                            # 非dummy段的数据传递到nvh进行计算
                            right_index_for_put = index_rawdata
                            sectionEnd_for_put = False
                            testNameIndex_for_put = result_index
                            recogIndex_for_put = recog_index
                            None_for_put = False
                else:
                    # 没有识别到开始点就不传递数据
                    None_for_put = True
            else:
                speed_right_index = rpm_index
                # detect the start point of target operating mode(not the first frame)
                if param.speedRecogInfo["torqueRecogFlag"][recog_index]:
                    gv_dict_speedRecog = speed_detect_end_with_torque(temp_rpm,
                                                                      rpml_array[
                                                                      :rpm_index],
                                                                      rpm_array[
                                                                      :rpm_index],
                                                                      gv_dict_rawdata[
                                                                          'Torque'][
                                                                      :index_rawdata],
                                                                      recog_index,
                                                                      gv_dict_speedRecog,
                                                                      param.speedRecogInfo,
                                                                      sampleRate)
                else:
                    gv_dict_speedRecog = speed_detect_end(temp_rpm,
                                                          rpml_array[:rpm_index],
                                                          rpm_array[:rpm_index],
                                                          recog_index,
                                                          gv_dict_speedRecog,
                                                          param.speedRecogInfo)
                if gv_dict_speedRecog['endpoint_loc']:
                    # if it's not the first part and have detected the end point,
                    # take part of data and put it into the queue
                    end_index = int(gv_dict_speedRecog['endpoint_loc'] * sampleRate)
                    start_index = end_index - (end_index % sampsPerChan)
                    if param.speedRecogInfo["notDummyFlag"][recog_index]:
                        right_index_for_put = end_index
                        sectionEnd_for_put = True
                        testNameIndex_for_put = result_index
                        recogIndex_for_put = recog_index
                        None_for_put = False

                        startpoint_index = gv_dict_speedRecog['startpoint_index']
                        endpoint_index = gv_dict_speedRecog['endpoint_index']
                        # 计算工况评估指标
                        RampQuality = ramp_quality(rpml_array[startpoint_index:endpoint_index],
                                                   rpm_array[startpoint_index:endpoint_index],
                                                   param.speedRecogInfo, recog_index)

                        result_index += 1
                        # put speed recognition
                        gv_dict_status["msg"] = \
                            param.speedRecogInfo["testName"][
                                recog_index] + "测试段识别完成！"
                        gv_dict_status_temp['data'] = gv_dict_status['data']
                        gv_dict_status_temp["data"]["startX"].append(
                            float(gv_dict_speedRecog["startpoint_loc"]))
                        gv_dict_status_temp["data"]["startY"].append(
                            float(gv_dict_speedRecog["startpoint_speed"]))
                        gv_dict_status_temp["data"]["endX"].append(
                            float(gv_dict_speedRecog["endpoint_loc"]))
                        gv_dict_status_temp["data"]["endY"].append(
                            float(gv_dict_speedRecog["endpoint_speed"]))
                        gv_dict_status_temp["data"]["testName"].append(
                            param.speedRecogInfo["testName"][recog_index])
                        gv_dict_status['data'] = gv_dict_status_temp['data']
                    else:
                        sectionEnd_for_put = True

                    recog_index = recog_index + 1
                    if recog_index < len(param.speedRecogInfo['testName']):
                        counter = counter - 1
                    else:
                        gv_dict_status["code"] = 2
                        gv_dict_status['msg'] = '转速识别完成!'
                        qDAQ_logger.info('speed recog finished' + datetime.datetime.now().strftime(
                            "%Y%m%d%H%M%S.%f"))
                        gv_dict_speedRecog['speedRecogFinish'] = True
                        # 这里之前的代码会put None,由于测试段识别已经结束，之前的代码这里put None不会出错，
                        # 修改之后由于需要统一put，这里不能添加None_for_put=True
                else:
                    # if it's not the first part and havn't detect the end point,
                    # put the data of this frame into the queue
                    if param.speedRecogInfo["notDummyFlag"][recog_index]:
                        # 要将该次传递进来的数据全部放入，
                        right_index_for_put = index_rawdata
                        sectionEnd_for_put = False
                        testNameIndex_for_put = result_index
                        recogIndex_for_put = recog_index
                        None_for_put = False

            if not None_for_put:
                qdata = {
                    # 测试段开始点vib信号的索引，一个测试段内该值不变
                    'vib_start_index': vib_start_index,
                    # 测试段内原始数据的右端索引，每次put会更新该值,注意右端索引处是没有值的
                    'vib_right_index': right_index_for_put,
                    # 测试段开始时的trigger在trigger_array中的位置，一个测试段内，该值不变
                    'trigger_start_index': trigger_start_index,
                    # 已经计算到了第几个trigger
                    # 'trigger_right_index':rpm_index*param.speedCalcInfo["step"]+param.speedCalcInfo["averageNum"],
                    'trigger_right_index': index_trigger if not sectionEnd_for_put else
                    (gv_dict_speedRecog['endpoint_index'] - index_rpm_last_speed_signal) * (
                            averageNum - overlap) +
                    averageNum + index_trigger_last_speed_signal,
                    # 测试段是否结束
                    'sectionEnd': sectionEnd_for_put,
                    # 有dummy段时，dummy可有多个
                    # 这个索引是去除dummy的索引
                    'testNameIndex': testNameIndex_for_put,
                    # 这个索引是包括dummy的索引
                    'recogIndex': recogIndex_for_put,
                    # 传给nvh的测试段内转速曲线的起始点索引，该段转速曲线的长度比测试段要长，一个测试段内该值不变
                    "speed_start_index": speed_start_index,
                    # 传给nvh的测试段内转速曲线的结束点索引，该段转速曲线的长度比测试段要长，每次put会更新该值
                    "speed_right_index": speed_right_index,
                    # if sectionEnd_for_put 测试段最后一帧才需要speedPattern
                    'speedPattern': param.speedRecogInfo['speedPattern'][
                        recogIndex_for_put],
                    'RampQuality': RampQuality,
                    "ppr": ppr,
                    "speedRatio": speedRatio
                }

                # print(qdata)
                # 将数据放入queue中
                for Q_speed_nvh in Q_speed_nvh_list:
                    Q_speed_nvh.put(qdata)

                if sectionEnd_for_put:
                    qDAQ_logger.debug("测试段{}识别结束:{}".format(recogIndex_for_put, time.time()))
                    qDAQ_logger.info("test name：{} recognition finished".format(
                        param.speedRecogInfo['testName'][recogIndex_for_put]))
                    qDAQ_logger.debug(qdata)
                    qDAQ_logger.debug(gv_dict_speedRecog)
                    print(gv_dict_speedRecog)
                    gv_dict_speedRecog = gv.set_default_speedRecog()
                    # dummy不可能是最后一段,所有测试段识别完成后将flag置为true，
                    # set_default_speedRecog会将flag置为false
                    if testNameIndex_for_put + 1 == param.speedCalcInfo["switchSection"]:
                        speed_signal = param.speedCalcInfo["signalAfterSwitch"]
                        if speed_signal == "ttl":
                            speedChannel = "Speed"
                            ppr = param.speedCalcInfo["ppr"]
                            triggerLevel = param.speedCalcInfo["triggerLevel"]
                            overlap = param.speedCalcInfo["overlap"]
                            averageNum = param.speedCalcInfo["averageNum"]
                        else:
                            speedChannel = "Sin"
                            ppr = param.speedCalcInfo["pprForBydResolver"]
                            triggerLevel = param.speedCalcInfo["triggerLevelForBydResolver"]
                            overlap = param.speedCalcInfo["overlapForBydResolver"]
                            averageNum = param.speedCalcInfo["averageNumForBydResolver"]
                            # 更新单路旋变的左索引
                            left_index = index_rawdata_backup

                        # 更新该值，上一种计算方法的到的trigger不用于转速切换后的转速计算,是第一次计算转速
                        last_rpm_cal_index = index_trigger
                        is_first_speed_calc = True

                        # 切换之后的转速比 速度乘以该转速比得到观察轴的速度，阶次除以该转速比得到观察轴的阶次
                        speedRatio = param.speedCalcInfo["speedRatioForByd"][1]

                        # 记录上一次转速信号计算到哪一个trigger以及rpm,该index处没有值，
                        # 即为下一个转速信号的0位置
                        index_trigger_last_speed_signal = index_trigger
                        index_rpm_last_speed_signal = rpm_index

                    if testNameIndex_for_put + 1 == param.speedRecogInfo["test_count_except_dummy"]:
                        qDAQ_logger.info("speed process consumer section end")
                        qDAQ_logger.info("speed process consumer section end :{}".format(
                            time.time() - time_click_start))
                        gv_dict_speedRecog['speedRecogFinish'] = True
                    sectionEnd_for_put = False

        else:
            # recognize finished, need to confirm if some error
            if recog_index < len(param.speedRecogInfo['testName']):
                gv_dict_status["code"] = 3000
                gv_dict_status['msg'] = '转速识别失败，请确认转速曲线!'

        counter = counter + 1
        icounter = icounter + 1

    if running_mode == "rt" and board == "NI":
        dtask.stop_task()
        dtask.clear_task()
        qDAQ_logger.info("NI task stopped!")
    elif running_mode == "rt" and board == "DT":
        target_ai_device.scan_stop()
        qDAQ_logger.info("DT task stopped!")
    try:
        if param.dataSaveFlag["speedCurve"]:
            if save_type == "hdf5":
                speed_result_filename = os.path.join(param.folderInfo["temp"],
                                                     param.basicInfo["type"] + '_' + fileName + '.h5')
                write_hdf5(speed_result_filename, 'speedData', 'speedLoc', rpml_array[:rpm_index])
                write_hdf5(speed_result_filename, 'speedData', 'speedValue', rpm_array[:rpm_index])
                write_hdf5(speed_result_filename, 'trigger', 'trigger', trigger_array[:index_trigger])
            else:
                speed_result_filename = os.path.join(param.folderInfo["temp"],
                                                     param.basicInfo["type"] + '_' + fileName + '.tdms')
                write_tdms(speed_result_filename, 'speedData', 'speedLoc', rpml_array[:rpm_index])
                write_tdms(speed_result_filename, 'speedData', 'speedValue', rpm_array[:rpm_index])
                write_tdms(speed_result_filename, 'trigger', 'trigger', trigger_array[:index_trigger])
            qDAQ_logger.info('speed curve saved, filename: ' + speed_result_filename)
        else:
            qDAQ_logger.info('speed curve not saved!')
    except Exception:
        gv_dict_status["code"] = 3000
        gv_dict_status["msg"] = "转速曲线保存失败!"
        qDAQ_logger.error("speed curve save failed, failed msg:" + traceback.format_exc())

    try:
        if param.dataSaveFlag["rawData"]:
            rawData_length = index_rawdata
            single_folder_confirm(file_info[0])
            # update raw data save properties
            raw_data_save_properties = create_properities(start_timestamp, 1 / sampleRate, sampsPerChan)
            if save_type == "hdf5":
                file_name = fileName + '.h5'
                raw_data_filename = os.path.join(file_info[0], fileName + '.h5')
                # 保存原始数据
                for i in range(len(param.taskInfo['channelNames'])):
                    raw_data_save_properties['NI_ChannelName'] = param.taskInfo['channelNames'][i]
                    raw_data_save_properties['NI_UnitDescription'] = param.taskInfo['units'][i]
                    write_hdf5(raw_data_filename, 'AIData',
                               param.taskInfo['channelNames'][i],
                               gv_dict_rawdata[param.taskInfo['channelNames'][i]][:rawData_length],
                               raw_data_save_properties)
                # 保存转速曲线和trigger信息
                write_hdf5(raw_data_filename, 'speedData', 'speedLoc', rpml_array[:rpm_index])
                write_hdf5(raw_data_filename, 'speedData', 'speedValue', rpm_array[:rpm_index])
                write_hdf5(raw_data_filename, 'triggerData', 'Trigger', trigger_array[:index_trigger])
            else:
                file_name = fileName + '.tdms'
                raw_data_filename = os.path.join(file_info[0], fileName + '.tdms')
                # 保存原始数据
                for i in range(len(param.taskInfo['channelNames'])):
                    raw_data_save_properties['NI_ChannelName'] = param.taskInfo['channelNames'][i]
                    raw_data_save_properties['NI_UnitDescription'] = param.taskInfo['units'][i]
                    write_tdms(raw_data_filename, 'AIData',
                               param.taskInfo['channelNames'][i],
                               gv_dict_rawdata[param.taskInfo['channelNames'][i]][:rawData_length],
                               raw_data_save_properties)
                # 保存转速曲线和trigger信息
                write_tdms(raw_data_filename, 'speedData', 'speedLoc', rpml_array[:rpm_index])
                write_tdms(raw_data_filename, 'speedData', 'speedValue', rpm_array[:rpm_index])
                write_tdms(raw_data_filename, 'triggerData', 'Trigger', trigger_array[:index_trigger])
            qDAQ_logger.info("raw data saved, filename: " + raw_data_filename)
            # 原始数据保存成功，传入参数开始执行数据上传和请求解析
            url_info = param.ftpUploadInfo
            folder_info = param.folderInfo
            folder_info["localFolder"] = file_info[0]
            data_info = dict()
            data_info['fileName'] = file_name
            data_info['system'] = param.basicInfo['systemNo']
            data_info['type'] = param.basicInfo['type']
            data_info['serial'] = param.basicInfo['serialNo']
            data_info['timestamp'] = timestamp_to_time(start_timestamp)
            format_timestamp = start_timestamp.strftime("%Y%m%d%H%M%S")
            ftp_queue.put({"urlInfo": url_info, "folderInfo": folder_info, "dataInfo": data_info,
                           "formatTimeStamp": format_timestamp})
            qDAQ_logger.debug("ftp upload request send out")
        else:
            qDAQ_logger.info("raw data not saved!")
    except Exception:
        gv_dict_status["code"] = 3000
        gv_dict_status["msg"] = "原始数据保存失败!"
        qDAQ_logger.error("raw data save failed, failed msg:" + traceback.format_exc())

    del cdata
    del gv_dict_rawdata
    del gv_dict_speedRecog
    del counter_angle
    del counter
    del icounter
    del recog_index
    del result_index
    del channelNames
    del right_index_for_put
    del sectionEnd_for_put
    del testNameIndex_for_put
    del recogIndex_for_put
    del None_for_put
    del raw_data_filename
    del raw_data_save_properties
    del temptl
    del temp_rpm
    del temp_rpml
    del param
    del file_info
    del start_timestamp
    del time_click_start
    # gc.collect()



def rawdataConsumer_for_fluctuation_norsp(Q_speed_nvh_list, param, gv_dict_status,
                                          file_info, start_timestamp,
                                          time_click_start):
    global gv_dict_flag, shm_vib
    global running_mode
    # 设置自动垃圾回收
    gc.enable()
    gc.set_threshold(1, 1, 1)
    qDAQ_logger.debug("rawdataConsumer:pid={},ppid={},thread={}".format(os.getpid(), os.getppid(),
                                                                        threading.current_thread().name))

    gv_dict_status_temp = dict()
    # 获取通道名称信息
    channelNames = param.taskInfo['channelNames']
    allrawdata = None

    fileName = param.basicInfo["fileName"]
    sampleRate = param.taskInfo["sampleRate"]
    sampsPerChan = param.taskInfo["sampsPerChan"]
    speedRecogInfo = param.speedRecogInfo
    simu_sleep_time = sampsPerChan / sampleRate * sleep_ratio

    if running_mode == "rt" and board == "NI":
        # create a task to read the raw data
        try:
            dtask = DAQTask(param.taskInfo)
            dtask.createtask()
            dtask.start_task()
        except nidaqmx.errors.DaqError:
            # if niDaq error, reset device and create task again
            qDAQ_logger.info('NI Daq error and reset device')
            dtask.stop_task()
            dtask.clear_task()
            reset_ni_device(ni_device)
            dtask = DAQTask(param.taskInfo)
            dtask.createtask()
            dtask.start_task()
        qDAQ_logger.debug("NI task start，type:{},serialNo:{}".format(param.basicInfo['type'],
                                                                     param.basicInfo['serialNo']))
    elif running_mode == "rt" and board == "Umic":
        if sensor_count != len(Umic_names):
            raise Exception(
                "传感器个数(sensor_count):{}和Umic个数(Umic_names):{}不匹配".format(sensor_count, len(Umic_names)))
        if bit_depth==24:
            sample_format = pyaudio.paInt24  # 24 bits per sample
        elif bit_depth==32:
            sample_format = pyaudio.paInt32  # 32 bits per sample
        else:
            raise Exception("Umic仅支持24位或32位，不支持{}位".format(bit_depth))
        stream_list = list()
        p = pyaudio.PyAudio()
        n_device = p.get_device_count()
        for umic_i in range(len(Umic_names)):
            device_index = None
            for i in range(n_device):
                info = p.get_device_info_by_index(i)
                name, hostapi = info['name'], info['hostApi']
                if name == Umic_names[umic_i] and hostapi == Umic_hostapis[umic_i]:
                    device_index = i
                    break
            if device_index == None:
                raise Exception(
                    "找不到对应的Umic设备,name:{},hostapi:{}".format(Umic_names[umic_i], Umic_hostapis[umic_i]))
            stream = p.open(format=sample_format,
                            channels=1,
                            rate=sampleRate,
                            frames_per_buffer=sampsPerChan,
                            input_device_index=device_index,
                            input=True)
            stream_list.append(stream)
        qDAQ_logger.info("umic stream建立完成")
    elif running_mode == "simu":

        allrawdata = read_raw_data(param.simuInfo["fileName"], channelNames, read_type)
        simu_data_len = len(allrawdata[channelNames[0]])
        qDAQ_logger.info("time after read all simu rawdata:{}".format(time.time() - time_click_start))

    # 保存原始数据的共享内存
    gv_dict_rawdata = dict()
    # 统计并记录传感器通道
    vibChannels = param.taskInfo['sensorChan']
    # 每一个通道均为振动通道
    # vibChannels = param.taskInfo['channelNames']

    for i in range(sensor_count):
        gv_dict_rawdata[vibChannels[i]] = np.ndarray(shape=(max_size,), dtype="f",
                                                     buffer=shm_vib[i].buf,
                                                     offset=0)
    # speed calculation
    gv_dict_speedRecog = gv.set_default_speedRecog()
    counter = 0
    cdata = dict()

    # 共享内存中存入的数组的索引，
    index_rawdata = 0

    # 下一个将要识别的测试段是第几个测试段
    index_test = 0

    rpm = list()
    rpml = list()

    # 第几次向queue中放入数据
    index_queue_put = 0
    while gv_dict_flag[flag_index_dict["speedCalclation"]]:

        # 获取原始数据，存入共享内存
        index_rawdata_backup = index_rawdata
        if running_mode == "rt" and board == "NI":
            data = dtask.read_data()
            if len(channelNames) == 1:
                len_rawdata_frame = len(data)
                gv_dict_rawdata[channelNames[0]][index_rawdata:len_rawdata_frame + index_rawdata] = data
                index_rawdata += len_rawdata_frame
            else:
                len_rawdata_frame = len(data[0])
                for i, channelName in enumerate(channelNames):
                    gv_dict_rawdata[channelName][index_rawdata:len_rawdata_frame + index_rawdata] = \
                        data[i]
                index_rawdata += len_rawdata_frame
        elif running_mode == "rt" and board == "Umic":
            for i, stream in enumerate(stream_list):
                data_bit24 = stream.read(sampsPerChan)
                decode_data = np.array(bit_to_int(data_bit24,bit_depth)) * param.taskInfo["sensitivity"][i] / (
                        2 ** (bit_depth-1) - 1)
                len_rawdata_frame = len(decode_data)
                gv_dict_rawdata[channelNames[i]][
                index_rawdata:len_rawdata_frame + index_rawdata] = decode_data
            index_rawdata += len_rawdata_frame
        else:
            if index_rawdata + sampsPerChan <= simu_data_len:
                for channelName in channelNames:
                    cdata[channelName] = allrawdata[channelName][
                                         counter * sampsPerChan:(counter + 1) * sampsPerChan]
                    len_rawdata_frame = len(cdata[channelName])
                    gv_dict_rawdata[channelName][index_rawdata:index_rawdata + len_rawdata_frame] = \
                        cdata[channelName]
                    # 控制simu模式的数据读取速度
                    # time.sleep(simu_sleep_time)
                index_rawdata += len_rawdata_frame
            else:
                gv_dict_flag[flag_index_dict["speed_finish"]] = 1

        # 如果测试段识别尚未结束，进行测试段识别
        if not gv_dict_speedRecog['speedRecogFinish']:

            # 测试段尚未开始
            if not gv_dict_speedRecog['startFlag']:
                if index_rawdata / sampleRate < speedRecogInfo["startTime"][index_test]:
                    None_for_put = True
                    sectionEnd_for_put = False
                else:
                    # 检测到开始点
                    None_for_put = False
                    sectionEnd_for_put = False
                    # 暂不考虑帧内测试段结束的情况
                    right_index_for_put = index_rawdata
                    gv_dict_speedRecog['startFlag'] = True
                    qDAQ_logger.debug("检测到开始点")
            else:
                if (index_rawdata / sampleRate) > speedRecogInfo["endTime"][index_test]:
                    # 检测到结束点
                    None_for_put = False
                    sectionEnd_for_put = True
                    right_index_for_put = int(speedRecogInfo["endTime"][index_test] * sampleRate)
                    gv_dict_speedRecog['startFlag'] = False
                else:
                    # 该帧数据全在测试段内
                    None_for_put = False
                    sectionEnd_for_put = False
                    right_index_for_put = index_rawdata

            if not None_for_put:
                # speed = speed_calibration(gv_dict_rawdata[channelNames[0]][
                #                           index_rawdata - len_rawdata_frame:index_rawdata],
                #                           speedRecogInfo["startSpeed"][index_test],
                #                           speedRecogInfo["endSpeed"][index_test],
                #                           speedRecogInfo["order"][index_test], sampleRate)
                # x_value=(index_rawdata - len_rawdata_frame / 2) / sampleRate
                # rpm.append(speed)
                # rpml.append(x_value)
                # gv_dict_status_temp['data'] = gv_dict_status['data']
                # gv_dict_status_temp['data']['x'] = float(rpml[-1])
                # gv_dict_status_temp['data']['y'] = float(rpm[-1])
                # gv_dict_status['data'] = gv_dict_status_temp['data']
                qdata = {
                    # 测试段开始点vib信号的索引，一个测试段内该值不变
                    'vib_start_index': int(speedRecogInfo["startTime"][index_test] * sampleRate),
                    # 测试段内原始数据的右端索引，每次put会更新该值,注意右端索引处是没有值的
                    'vib_right_index': right_index_for_put,
                    "testNameIndex": index_test,
                    "sectionEnd": sectionEnd_for_put
                }
                # qDAQ_logger.debug(qdata)
                # 将数据放入queue中
                for Q_speed_nvh in Q_speed_nvh_list:
                    Q_speed_nvh.put(qdata)
                index_queue_put += 1

                if sectionEnd_for_put:
                    qDAQ_logger.info("section:{} detect end".format(index_test))
                    if index_test + 1 == param.speedRecogInfo["test_count_except_dummy"]:
                        gv_dict_speedRecog['speedRecogFinish'] = True
                        qDAQ_logger.info("测试段识别结束")

                    # gv_dict_status_temp['data'] = gv_dict_status['data']
                    # # 起始时刻从参数配置中拿就好
                    # gv_dict_status_temp["data"]["startX"].append(
                    #     float(speedRecogInfo["startTime"][index_test]))
                    # # 没必要记下起始转速，从参数配置中拿转速，前端可以正常显示就好
                    # gv_dict_status_temp["data"]["startY"].append(
                    #     float(speedRecogInfo["startSpeed"][index_test] + speedRecogInfo["endSpeed"][
                    #         index_test]) / 2)
                    # gv_dict_status_temp["data"]["endX"].append(
                    #     float(rpml[-1]))
                    # gv_dict_status_temp["data"]["endY"].append(
                    #     float(rpm[-1]))
                    # gv_dict_status_temp["data"]["testName"].append(
                    #     param.speedRecogInfo["testName"][index_test])
                    # gv_dict_status['data'] = gv_dict_status_temp['data']

                    index_test += 1
                    index_queue_put = 0

        else:
            pass

        counter += 1

    if running_mode == "rt":
        dtask.stop_task()
        dtask.clear_task()
        qDAQ_logger.info("NI task stopped!")

    # 保存原始数据
    try:
        if param.dataSaveFlag["rawData"]:
            rawData_length = index_rawdata
            single_folder_confirm(file_info[0])
            # update raw data save properties
            raw_data_save_properties = create_properities(start_timestamp, 1 / sampleRate, sampsPerChan)
            if save_type == "hdf5":
                file_name = fileName + '.h5'
                raw_data_filename = os.path.join(file_info[0], fileName + '.h5')
                for i in range(len(param.taskInfo['channelNames'])):
                    raw_data_save_properties['NI_ChannelName'] = param.taskInfo['channelNames'][i]
                    raw_data_save_properties['NI_UnitDescription'] = param.taskInfo['units'][i]
                    write_hdf5(raw_data_filename, 'AIData',
                               param.taskInfo['channelNames'][i],
                               gv_dict_rawdata[param.taskInfo['channelNames'][i]][:rawData_length],
                               raw_data_save_properties)
            else:
                file_name = fileName + '.tdms'
                raw_data_filename = os.path.join(file_info[0], fileName + '.tdms')
                for i in range(len(param.taskInfo['channelNames'])):
                    raw_data_save_properties['NI_ChannelName'] = param.taskInfo['channelNames'][i]
                    raw_data_save_properties['NI_UnitDescription'] = param.taskInfo['units'][i]
                    write_tdms(raw_data_filename, 'AIData',
                               param.taskInfo['channelNames'][i],
                               gv_dict_rawdata[param.taskInfo['channelNames'][i]][:rawData_length],
                               raw_data_save_properties)
            qDAQ_logger.info("raw data saved, filename: " + raw_data_filename)
            # 原始数据保存成功，传入参数开始执行数据上传和请求解析
            url_info = param.ftpUploadInfo
            folder_info = param.folderInfo
            folder_info["localFolder"] = file_info[0]
            data_info = dict()
            data_info['fileName'] = file_name
            data_info['system'] = param.basicInfo['systemNo']
            data_info['type'] = param.basicInfo['type']
            data_info['serial'] = param.basicInfo['serialNo']
            data_info['timestamp'] = timestamp_to_time(start_timestamp)
            format_timestamp = start_timestamp.strftime("%Y%m%d%H%M%S")
            ftp_queue.put({"urlInfo": url_info, "folderInfo": folder_info, "dataInfo": data_info,
                           "formatTimeStamp": format_timestamp})
            qDAQ_logger.debug("ftp upload request send out")
        else:
            qDAQ_logger.info("raw data not saved!")
    except Exception:
        gv_dict_status["code"] = 3000
        gv_dict_status["msg"] = "原始数据保存失败!"
        qDAQ_logger.warning("raw data save failed, failed msg:" + traceback.format_exc())
    # 保存转速曲线
    # try:
    #     if param.dataSaveFlag["speedCurve"]:
    #         if save_type == "hdf5":
    #             speed_result_filename = os.path.join(file_info[0], fileName + '.h5')
    #             write_hdf5(speed_result_filename, 'speedData', 'speedLoc', rpml)
    #             write_hdf5(speed_result_filename, 'speedData', 'speedValue', rpm)
    #         else:
    #             speed_result_filename = os.path.join(file_info[0], fileName + '.tdms')
    #             write_tdms(speed_result_filename, 'speedData', 'speedLoc', rpml)
    #             write_tdms(speed_result_filename, 'speedData', 'speedValue', rpm)
    #         qDAQ_logger.info('speed curve saved, filename: ' + speed_result_filename)
    #     else:
    #         qDAQ_logger.info('speed curve not saved!')
    # except Exception:
    #     gv_dict_status["code"] = 3000
    #     gv_dict_status["msg"] = "转速曲线保存失败!"
    #     qDAQ_logger.warning("speed curve save failed, failed msg:" + traceback.format_exc())

    # simu时需要删除读到内存中的allrawdata
    del allrawdata
    del cdata
    del gv_dict_rawdata
    del gv_dict_speedRecog
    del counter
    del channelNames
    del None_for_put
    del raw_data_filename
    del raw_data_save_properties
    del param
    del file_info
    del start_timestamp
    del time_click_start
    gc.collect()
