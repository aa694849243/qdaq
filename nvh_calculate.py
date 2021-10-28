#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2020/9/24 23:42
@Author : Wall@Synovate
@Email  : qxu@sonustc.com
@Function: real time speed recognition and NVH calculation
"""
import threading
import gc
import numpy as np
import time
import os
import logging
import traceback
from queue import Empty, Full
from common_info import max_size, flag_index_dict, save_type, qDAQ_logger, version, ramp_quality_flag
from initial import create_empty_twodtd_for_share, \
    create_empty_temptd_for_share, create_empty_twodsf_for_share, \
    create_empty_tempsf_for_share, update_nvh_data_for_thread, \
    create_empty_threedos_for_share, create_empty_twodoc_for_share, single_folder_confirm, \
    create_empty_twodsf_for_const, create_empty_tempsf_for_const
from indicator_tools import oned_time_domain, average_oned_stat_factor, \
    twod_time_domain_for_share, twod_stat_factor_for_share, \
    order_spectrum_for_share, convert_time_speed_for_share, angular_resampling, db_convertion, \
    order_spectrum_for_const, twod_stat_factor_for_const, convert_time_speed_for_const, \
    order_spectrum_for_const_fluctuation, twod_stat_factor_for_const_fluctuation, oned_stat_factor, \
    oned_stat_factor_mean, oned_stat_factor_mean_for_const
from indicator_tools import twod_order_spectrum, oned_order_spectrum, cepstrum
from parameters import filter_win
from speed_tools import ramp_quality_for_const, speed_calibration, speed_calibration_with_orderlist
from utils import write_hdf5, write_tdms, tdms_data_confirm, hdf5_data_confirm
from multiprocessing import shared_memory
from pyinstrument import Profiler

# test_result被进行了初始化，并且主进程中在传递给子进程后没有更改，
# param在主进程中初始化后不会再产生更改
# 这两者在子进程中会随着检测时间不断增加


def nvh_process(Q_nvh_in, Q_nvh_out, Q_speed_nvh, Q_nvh_datapack, sensor_index,
                lock_for_tdms):
    global gv_dict_flag, trigger_ndarray, rpml_array, rpm_array, vib_ndarray
    try:
        if version == 1 or version == 2:
            # 每个进程是否工作的flag所在的共享内存
            shm_flag = shared_memory.SharedMemory(name="shm_flag")
            gv_dict_flag = shm_flag.buf
            # 上升沿/下降沿所在的共享内存
            shm_trigger = shared_memory.SharedMemory(name="shm_trigger")
            trigger_ndarray = np.ndarray(shape=(max_size // 2 + 1,), dtype="i", buffer=shm_trigger.buf)
            # 该传感器（sensor_index）的振动信号所在的共享内存
            shm_vib = shared_memory.SharedMemory(name="shm_vib" + str(sensor_index))
            vib_ndarray = np.ndarray(shape=(max_size,), dtype='f', buffer=shm_vib.buf)
            # 转速曲线所在的共享内存 shm_rpml:时间 shm_rpm:转速值
            shm_rpml = shared_memory.SharedMemory(name='shm_rpml')
            shm_rpm = shared_memory.SharedMemory(name='shm_rpm')
            rpml_array = np.ndarray(shape=(max_size // 2 + 1,), dtype="f", buffer=shm_rpml.buf)
            rpm_array = np.ndarray(shape=(max_size // 2 + 1,), dtype="f", buffer=shm_rpm.buf)
        elif version == 3 or version == 4:
            # 每个进程是否工作的flag所在的共享内存
            shm_flag = shared_memory.SharedMemory(name="shm_flag")
            gv_dict_flag = shm_flag.buf

            # 该传感器（sensor_index）的振动信号所在的共享内存
            shm_vib = shared_memory.SharedMemory(name="shm_vib" + str(sensor_index))
            vib_ndarray = np.ndarray(shape=(max_size,), dtype='f', buffer=shm_vib.buf)

        qDAQ_logger.info("nvh process memory success")
    except Exception:
        qDAQ_logger.error("nvh process memory error, error msg:" + traceback.format_exc())

    while True:
        data = Q_nvh_in.get()
        try:
            if version == 1 or version == 2:
                nvh_calc_process_for_qdaq(Q_speed_nvh, Q_nvh_datapack,
                                          data['gv_dict_status'],
                                          data['param'],
                                          data['test_result'], data['sensor_index'],
                                          data['time_click_start'],
                                          data['file_info'],
                                          data['colormap_save_properties'],
                                          lock_for_tdms)

            elif version == 4:
                nvh_calc_process_for_fluctuation_norsp(Q_speed_nvh, Q_nvh_datapack,
                                                       data['gv_dict_status'],
                                                       data['param'],
                                                       data['test_result'], data['sensor_index'],
                                                       data['time_click_start'],
                                                       data['file_info'],
                                                       data['colormap_save_properties'],
                                                       lock_for_tdms)
        except Exception:
            qDAQ_logger.error('nvh calculation error, error msg:' + traceback.format_exc())
            data['gv_dict_status']["code"] = 3000
            data['gv_dict_status']["msg"] = "nvh{}进程出现错误!".format(data['sensor_index'])
            gv_dict_flag[flag_index_dict["nvh_error"]]=1
        Q_nvh_out.put(data['sensor_index'])  # 自己put会被自己get
        del data


def nvh_calc_process_for_qdaq(in_Q, out_Q, gv_dict_status,
                              param, test_result, sensor_index,
                              time_click_start,
                              file_info, colormap_save_properties, lock_for_tdms):
    global gv_dict_flag, trigger_ndarray, rpml_array, rpm_array, vib_ndarray

    # gc.enable()
    qDAQ_logger.debug("nvh_calc_process:pid={},ppid={},thread={}".format(os.getpid(), os.getppid(),
                                                                         threading.current_thread().name))

    # 时域指标计算帧长
    calSize = param.timeDomainCalcInfo["calSize"]
    # 角度重采样两边均多取arPoints个点
    arPoints = param.orderSpectrumCalcInfo["arPoints"]
    # 采样率
    sampleRate = param.taskInfo["sampleRate"]
    # 帧长
    sampsPerChan = param.taskInfo["sampsPerChan"]

    # 该测试段从queue中取得的第几个data （从1开始）
    counter_data_queue = 0
    # 时域的计算帧长为calSize,该索引记录进行了几次计算，
    # 用于twod_td以及temp_td,和之前的counter_td意义相同
    index_twod_td = 0

    while gv_dict_flag[flag_index_dict['nvhCalclation']]:
        try:
            data = in_Q.get(block=True, timeout=1)
        except Empty:
            continue

        if not data:
            continue

        if data['sectionEnd']:
            time_get_sectionend = time.time()
            qDAQ_logger.debug(
                "sensor {} nvh_get_sectionend:{}".format(sensor_index, time.time() - time_click_start))

        if counter_data_queue == 0:
            qDAQ_logger.debug(
                "sensor {} nvh_start:{}".format(sensor_index, time.time() - time_click_start))
            # 测试段的第一帧数据
            # 左索引
            vib_start_index = data['vib_start_index']
            # 更新右索引，右端索引处没有值
            vib_right_index = data['vib_right_index']
            # 更新trigger的左索引
            trigger_start_index = data['trigger_start_index']
            trigger_right_index = data['trigger_right_index']

            # maxT是测试段的最长时间，测试时可能比该时长稍长，暂时取1.5倍
            maxT = param.speedRecogInfo["maxT"][data['recogIndex']] * 1.5
            # maxSpeed是测试段内的最大速度
            maxSpeed = max(
                param.speedRecogInfo["startSpeed"][data['recogIndex']],
                param.speedRecogInfo["endSpeed"][data['recogIndex']])
            # maxLen是该测试段时域计算时按帧长calSize最多计算多少次
            maxLen = int(maxT * sampleRate // calSize) + 1
            # 创建空的twodtd和空的temp_td，创建的容量保证能放下测试段内的数据，
            # 将来测试段结束后再进行截取，然后再进行计算
            twod_td = create_empty_twodtd_for_share(param.timeDomainCalcInfo, sensor_index, maxLen,
                                                    indicator_diagnostic=
                                                    param.speedRecogInfo['initial_indicator_diagnostic'][
                                                        data['recogIndex']])
            temp_td = create_empty_temptd_for_share(maxLen)
            # 一个测试段内计算了多少个twod_td计算了多少次 ，新测试段开始时赋值为0
            index_twod_td = 0

            # 该测试段最长时间*最大转速=最多转了多少圈
            # 最多多少圈/重采样后两点之间的圈数间隔=重采样数据最大有多少个
            rspMaxLen = int(maxT * maxSpeed / 60 // param.orderSpectrumCalcInfo['dr_af']) + 1
            # 角度重采样后得到的数据，事先开辟足够的空间，以免每一帧要进行extend
            vib_rsp_time = np.zeros(rspMaxLen)
            vib_rsp = np.zeros(rspMaxLen)
            # 写到了第几个vib_rsp ，索引为vib_rsp_index的值是下一个要写入的点
            vib_rsp_index = 0

            # 记录按圈计算过程中每一个轴计算过的次数
            twod_sf_counter = dict()
            twod_sf = create_empty_twodsf_for_share(param.statFactorCalcInfo, sensor_index, rspMaxLen,
                                                    indicator_diagnostic=
                                                    param.speedRecogInfo['initial_indicator_diagnostic'][
                                                        data['recogIndex']])
            temp_sf = create_empty_tempsf_for_share(param.statFactorCalcInfo,sensor_index,rspMaxLen)

            # 每32圈计算一次，overlapRatio为0.75 即不仅长度为8
            osMaxLen = round(maxT * maxSpeed / 60 // (param.orderSpectrumCalcInfo['revNum'] * (
                    1 - param.orderSpectrumCalcInfo['overlapRatio']))) + 1
            threed_os = create_empty_threedos_for_share(param.orderSpectrumCalcInfo, param.taskInfo,
                                                        sensor_index, osMaxLen,
                                                        db_flag=param.basicInfo['dBFlag'],
                                                        indicator_diagnostic=param.speedRecogInfo[
                                                            'initial_indicator_diagnostic'][
                                                            data['recogIndex']])
            twod_oc = create_empty_twodoc_for_share(param.orderCutCalcInfo,
                                                    param.taskInfo,
                                                    sensor_index, osMaxLen,
                                                    db_flag=param.basicInfo['dBFlag'],
                                                    indicator_diagnostic=
                                                    param.speedRecogInfo['initial_indicator_diagnostic'][
                                                        data['recogIndex']])
            # 每次重采样取的trigger对应的圈数和时间从该位置截取
            trigger_right_index_backup = 0
            # twod_td计算的左索引，测试段开始时更新为测试段开始振动点
            last_calc_index = vib_start_index
            # 进行了几帧角度重采样，
            counter_ar = 0
            # 进行了几帧时间域的计算，并不一定等于时间域计算方法的运行次数，
            # 调用一次方法，在方法内可能进行了多帧运算
            counter_td = 0
            # 进行了几帧阶次谱的运算
            counter_or = 0

            # 由于byd的需求，不同的测试段有不同的ppr
            ppr = data["ppr"]
            speedRatio=data["speedRatio"]
        else:
            # 不是测试段的第一帧数据，更新振动信号和trigger的右索引
            vib_right_index = data['vib_right_index']
            trigger_right_index = data['trigger_right_index']
        # 测试段内的第几个data
        counter_data_queue += 1

        # 进行测试段内数据的计算
        # 以传来的第一个值作为测试段的0时刻点，作为0圈数点
        # 角度重采样，一测试段内传进来的第一个vib点为0时刻点，0圈数点

        # 测试段内vib的长度
        len_vib = trigger_ndarray[trigger_right_index - 1] - trigger_ndarray[trigger_start_index]
        if counter_ar > 0:
            # 不是第一帧，向前截取一段trigger，
            rev_trigger_left = trigger_right_index - 2 * (
                    trigger_right_index - trigger_right_index_backup)
        else:
            # 第一帧
            rev_trigger_left = trigger_start_index
        while (counter_ar + 1) * sampsPerChan < len_vib:
            if counter_ar > 0:
                # 不是第一帧需要往前一帧多取一些点（滤波带来的边缘效应），然后由下一次计算的结果补上
                s_index = counter_ar * sampsPerChan - arPoints * 2
            else:
                # 第一帧无法往前取点
                s_index = 0
            # 确定结束点的索引
            e_index = (counter_ar + 1) * sampsPerChan

            # trigger对应的圈数
            rev = (np.arange(rev_trigger_left - trigger_start_index,
                             trigger_right_index - trigger_start_index)) / ppr
            # trigger对应的时刻点，以测试段开始点为0时刻
            rev_time = (trigger_ndarray[rev_trigger_left:trigger_right_index] - trigger_ndarray[
                trigger_start_index]) / sampleRate
            # 振动信号
            target_vib = vib_ndarray[trigger_ndarray[trigger_start_index] + s_index:
                                     trigger_ndarray[trigger_start_index] + e_index]
            # 振动信号对应的时刻点
            target_time = np.arange(s_index, e_index) / sampleRate
            try:
                vib_rsp_frame, vib_rsp_time_frame = angular_resampling(rev, rev_time, target_vib,
                                                                       target_time, counter_ar,
                                                                       arPoints,
                                                                       param.orderSpectrumCalcInfo[
                                                                           'dr_bf'],
                                                                       param.orderSpectrumCalcInfo[
                                                                           'dr_af'])
            except Exception:
                gv_dict_status["code"] = 3000
                gv_dict_status["msg"] = "角度域重采样错误！"
                qDAQ_logger.error("test: " + param.speedRecogInfo['testName'][
                    data['recogIndex']] +
                                  "angular resampling failed, failed msg:" + traceback.format_exc())
                break
            counter_ar += 1
            # 将本帧重采样数据放入整个ndarray中
            vib_rsp[vib_rsp_index:vib_rsp_index + len(vib_rsp_frame)] = vib_rsp_frame
            vib_rsp_time[vib_rsp_index:vib_rsp_index + len(vib_rsp_frame)] = \
                vib_rsp_time_frame + trigger_ndarray[trigger_start_index] / sampleRate
            vib_rsp_index = vib_rsp_index + len(vib_rsp_frame)
        # 保存本帧右trigger点
        trigger_right_index_backup = trigger_right_index

        # 分帧计算时域信息
        try:
            # time domain indicators
            twod_td, temp_td, counter_td, index_twod_td, last_calc_index = twod_time_domain_for_share(
                vib_ndarray,
                last_calc_index,
                vib_right_index,
                calSize,
                param.timeDomainCalcInfo["indicatorNestedList"][sensor_index],
                param.timeDomainCalcInfo["refValue"][sensor_index],
                twod_td,
                temp_td,
                counter_td,
                sampleRate,
                index_twod_td)
        except Exception:
            gv_dict_status["code"] = 3000
            gv_dict_status["msg"] = "二维时间域指标计算错误!"
            qDAQ_logger.error(
                "sensor " + str(
                    sensor_index + 1) + ", test: " + (
                    param.speedRecogInfo['testName'][data['recogIndex']]) +
                "2D time domain indicators calculation failed, failed msg:" + traceback.format_exc())
            break

        try:
            # order domain indicators
            threed_os, twod_oc, counter_or = order_spectrum_for_share(threed_os,
                                                                      twod_oc,
                                                                      counter_or,
                                                                      vib_rsp[:vib_rsp_index],
                                                                      vib_rsp_time[:vib_rsp_index],
                                                                      param.orderSpectrumCalcInfo,
                                                                      param.orderCutCalcInfo,
                                                                      sensor_index,
                                                                      db_flag=param.basicInfo['dBFlag'])
        except Exception:
            gv_dict_status["code"] = 3000
            gv_dict_status["msg"] = "阶次域指标计算错误!"
            qDAQ_logger.error(
                "sensor " + str(
                    sensor_index + 1) + ", test: " + (
                    param.speedRecogInfo['testName'][data['recogIndex']]) +
                ": 2/3D order domain indicators calculation failed, failed msg:" + traceback.format_exc())
            break

        try:
            # 按圈计算
            twod_sf, temp_sf = twod_stat_factor_for_share(twod_sf, temp_sf, vib_rsp[:vib_rsp_index],
                                                          vib_rsp_time[:vib_rsp_index], twod_sf_counter,
                                                          sensor_index,param.statFactorCalcInfo)
        except Exception:
            gv_dict_status["code"] = 3000
            gv_dict_status["msg"] = "二维按圈计算指标错误!"
            qDAQ_logger.error("sensor " + str(sensor_index + 1) + ", test: " + (
                param.speedRecogInfo['testName'][data[
                    'recogIndex']]) + "2D indicators by revolution calculation failed, failed msg:"
                              + traceback.format_exc())
            break

        if data['sectionEnd']:
            twod_td, temp_td, twod_sf, temp_sf, threed_os, twod_oc = update_nvh_data_for_thread(
                twod_td, temp_td, index_twod_td,
                twod_sf, temp_sf, twod_sf_counter,
                threed_os, twod_oc, counter_or)
            speed_start_index = data['speed_start_index']
            speed_right_index = data['speed_right_index']

            # prepare test finished
            try:
                # update 1D time domain indicators into final test result
                if test_result['resultData'][sensor_index]['dataSection'][data['testNameIndex']][
                    'onedData'] is None:
                    test_result['resultData'][sensor_index]['dataSection'][data['testNameIndex']][
                        'onedData'] = list()
                test_result['resultData'][sensor_index]['dataSection'][data['testNameIndex']][
                    'onedData'].extend(
                    oned_time_domain(temp_td, calSize,
                                     param.timeDomainCalcInfo, sensor_index,
                                     db_flag=param.basicInfo['dBFlag']))
                qDAQ_logger.info(
                    "1D time domain indicators calculation of sensor{}, test: {} finished".format(
                        sensor_index + 1,
                        param.speedRecogInfo['testName'][
                            data['recogIndex']]))
            except Exception:
                gv_dict_status["code"] = 3000
                gv_dict_status["msg"] = "一维时间域指标计算错误!"
                qDAQ_logger.error(
                    "sensor " + str(
                        sensor_index + 1) + ", test: " + (
                        param.speedRecogInfo['testName'][data['recogIndex']]) +
                    ": 1D time domain indicators calculation failed, failed msg:" + traceback.format_exc())

            try:
                # update 1D indicators by revolution into final test result
                if test_result['resultData'][sensor_index]['dataSection'][
                    data['testNameIndex']]['onedData'] is None:
                    test_result['resultData'][sensor_index]['dataSection'][
                        data['testNameIndex']][
                        'onedData'] = list()
                # 计算一维按圈计算指标并更新二维结果
                # twod_sf, oned_sf = average_oned_stat_factor(twod_sf, param.statFactorCalcInfo,
                #                                             sensor_index)
                oned_sf,twod_sf=oned_stat_factor_mean(temp_sf, twod_sf, param.statFactorCalcInfo, sensor_index)
                test_result['resultData'][sensor_index]['dataSection'][
                    data['testNameIndex']]['onedData'].extend(oned_sf)
                qDAQ_logger.info(
                    "1D indicators by revolution calculation of sensor{}, test: {} finished".format(
                        sensor_index + 1,
                        param.speedRecogInfo['testName'][
                            data['recogIndex']]))
            except Exception:
                gv_dict_status["code"] = 3000
                gv_dict_status["msg"] = "一维按圈计算指标计算错误!"
                qDAQ_logger.error(
                    "sensor " + str(
                        sensor_index + 1) + ", test: " + (
                        param.speedRecogInfo['testName'][data['recogIndex']]) +
                    ": 1D indicators by revolution calculation failed, failed msg:" + traceback.format_exc())

            # 更新二维时间域结果的x轴
            twod_td = convert_time_speed_for_share(twod_td,
                                                   rpml_array[speed_start_index:speed_right_index],
                                                   rpm_array[speed_start_index:speed_right_index],
                                                   data['speedPattern'],
                                                   speedRatio)
            test_result['resultData'][sensor_index]['dataSection'][data['testNameIndex']][
                'twodTD'] = twod_td
            qDAQ_logger.info(
                "2D time domain indicators calculation of sensor{}, test: {} finished".format(
                    sensor_index + 1,
                    param.speedRecogInfo['testName'][
                        data['recogIndex']]))
            # 更新二维按圈计算指标的x轴
            twod_sf = convert_time_speed_for_share(twod_sf,
                                                   rpml_array[speed_start_index:speed_right_index],
                                                   rpm_array[speed_start_index:speed_right_index],
                                                   data['speedPattern'],
                                                   speedRatio,
                                                   indicator_num=param.statFactorCalcInfo[
                                                       'indicatorNum'][sensor_index])
            # 添加二维按圈计算指标到二维时间域指标中
            test_result['resultData'][sensor_index]['dataSection'][data['testNameIndex']][
                'twodTD'].extend(twod_sf)
            qDAQ_logger.info(
                "2D indicators by revolution calculation of sensor{} finished".format(sensor_index + 1))
            # update stop and end time of present test into final test result
            test_result['resultData'][sensor_index]['dataSection'][data['testNameIndex']]['startTime'] = \
                data['vib_start_index'] / sampleRate
            test_result['resultData'][sensor_index]['dataSection'][data['testNameIndex']]['endTime'] = \
                data['vib_right_index'] / sampleRate

            try:
                # update 2D order spectrum into final test result
                twod_os = twod_order_spectrum(threed_os, param.taskInfo, sensor_index)
                qDAQ_logger.info(
                    "2D order spectrum calculation of sensor{}, test: {} finished".format(
                        sensor_index + 1,
                        param.speedRecogInfo[
                            'testName'][
                            data[
                                'recogIndex']]))
            except Exception:
                gv_dict_status["code"] = 3000
                gv_dict_status["msg"] = "二维阶次谱计算错误!"
                qDAQ_logger.error(
                    "sensor " + str(
                        sensor_index + 1) + ", test: " + (
                        param.speedRecogInfo['testName'][data['recogIndex']]) +
                    ": 2D order spectrum calculation failed, failed msg:" + traceback.format_exc())

            try:
                # update 2D order cepstrum into final test result
                logging.info("len(threed_os['xValue']):{}".format(len(threed_os['xValue'])))
                logging.info("len(twod_oc['xValue']):{}".format(len(twod_oc[0]['xValue'])))
                test_result['resultData'][sensor_index]['dataSection'][
                    data['testNameIndex']]['twodCeps'] = \
                    cepstrum(twod_os, param.cepstrumCalcInfo, param.taskInfo,
                             sensor_index, db_flag=param.basicInfo['dBFlag'])
                qDAQ_logger.info(
                    "2D order cepstrum calculation of sensor{}, test: {} finished".format(
                        sensor_index + 1,
                        param.speedRecogInfo[
                            'testName'][
                            data[
                                'recogIndex']]))
            except Exception:
                gv_dict_status["code"] = 3000
                gv_dict_status["msg"] = "二维倒阶次谱计算错误!"
                qDAQ_logger.error(
                    "sensor " + str(
                        sensor_index + 1) + ", test: " + (
                        param.speedRecogInfo['testName'][data['recogIndex']]) +
                    ": 2D order cepstrum calculation failed, failed msg:" + traceback.format_exc())

            try:
                # update 1D order domain indicators into final test result
                test_result['resultData'][sensor_index]['dataSection'][
                    data['testNameIndex']]['onedData'] += oned_order_spectrum(twod_os,
                                                                              param.onedOSCalcInfo,
                                                                              param.taskInfo,
                                                                              param.modulationDepthCalcInfo,
                                                                              sensor_index,
                                                                              db_flag=param.basicInfo[
                                                                                  'dBFlag'])
                # 放入工况评估指标
                if ramp_quality_flag:
                    test_result['resultData'][sensor_index]['dataSection'][
                        data['testNameIndex']]['onedData'].append(data['RampQuality'])
                qDAQ_logger.info(
                    "1D order indicators calculation of sensor{}, test: {} finished".format(
                        sensor_index + 1,
                        param.speedRecogInfo['testName'][
                            data['recogIndex']]))
            except Exception:
                gv_dict_status["code"] = 3000
                gv_dict_status["msg"] = "一维阶次切片计算错误!"
                qDAQ_logger.error(
                    "sensor " + str(
                        sensor_index + 1) + ", test: " + (
                        param.speedRecogInfo['testName'][data['recogIndex']]) +
                    ": 1D order cutting calculation failed, failed msg:" + traceback.format_exc())
            try:
                # 更新二维阶次谱和三维阶次谱的阶次信息（一定要在阶次切片结果计算完成以后）
                if speedRatio != 1:
                    if twod_os:
                        # twod_os[0]['xValue'] = param.orderSpectrumCalcInfo['convertOrder']
                        # threed_os['yValue'] = param.orderSpectrumCalcInfo['convertOrder']

                        twod_os[0]["xValue"] = (np.array(twod_os[0]["xValue"])/speedRatio).tolist()
                        threed_os['yValue'] = (np.array(threed_os['yValue'])/speedRatio).tolist()
                # 更新结果到结果数据中
                # 更新结果到结果数据中（包括dB转换）
                if param.basicInfo['dBFlag']:
                    twod_os[0]['yUnit'] = 'dB'
                    twod_os[0]['yValue'] = db_convertion(twod_os[0]['yValue'],
                                                         param.taskInfo['refValue'][
                                                             sensor_index])
                # 更新yValue由ndarray变为list(),
                # 之前会用twod_os[0]['yValue']进行运算，在运算完成之后进行更新
                twod_os[0]['yValue'] = twod_os[0]['yValue'].tolist()
                test_result['resultData'][sensor_index]['dataSection'][data['testNameIndex']][
                    'twodOS'] = twod_os
                qDAQ_logger.info(
                    "order conversion of sensor{}, test: {} finished".format(sensor_index + 1,
                                                                             param.speedRecogInfo[
                                                                                 'testName'][
                                                                                 data[
                                                                                     'recogIndex']]))
            except Exception:
                gv_dict_status["code"] = 3000
                gv_dict_status["msg"] = "阶次转换错误!"
                qDAQ_logger.error(
                    "sensor " + str(
                        sensor_index + 1) + ", test: " + (
                        param.speedRecogInfo['testName'][data['recogIndex']]) +
                    ": order convertion failed, failed msg:" + traceback.format_exc())
            # 更新二维阶次切片和三维阶次谱的X轴信息
            twod_oc, threed_os = convert_time_speed_for_share(twod_oc,
                                                              rpml_array[
                                                              speed_start_index:speed_right_index],
                                                              rpm_array[
                                                              speed_start_index:speed_right_index],
                                                              data['speedPattern'],
                                                              speedRatio,
                                                              threed_os=threed_os)
            test_result['resultData'][sensor_index]['dataSection'][
                data['testNameIndex']]['twodOC'] = twod_oc
            qDAQ_logger.info(
                "nvh calculation of sensor{}, test: {} finished ".format(sensor_index + 1,
                                                                         param.speedRecogInfo[
                                                                             'testName'][
                                                                             data['recogIndex']]))
            del threed_os["tempXi2"]

            if twod_os:
                error_flag = 0
            else:
                error_flag = 1
            while gv_dict_flag[flag_index_dict['dataPack']]:
                try:
                    # 将test_result内的数据变为可以json化的数据
                    out_Q.put(
                        {"sensorIndex": sensor_index,
                         "testNameIndex": data['testNameIndex'],
                         "data": test_result,
                         "error": error_flag}, block=True, timeout=1)
                    break
                except Full:
                    continue
            qDAQ_logger.debug("从nvh获取到测试段最后一帧到已经将数据放入queue总时间:{}".format(time.time() - time_click_start))
            qDAQ_logger.debug("nvh获取到测试段最后一帧到已经将数据放入queue:{}".format(time.time() - time_get_sectionend))

            if param.dataSaveFlag['colorMap']:
                # store the colormap
                try:
                    # 2 sensor can not deal with colormap data like 1 sensor
                    if save_type == 'hdf5':
                        colormap_filename = os.path.join(file_info[1],
                                                         param.basicInfo["fileName"] + '.h5')
                    else:
                        colormap_filename = os.path.join(file_info[1],
                                                         param.basicInfo["fileName"] + '.tdms')

                    group_name = 'ATEOLOSMAP_' + param.speedRecogInfo['testName'][data['recogIndex']]
                    single_folder_confirm(file_info[1])
                    # 每个传感器的彩图数据需要分开写，所以写入一个传感器的时候需要防止其它进程写入
                    lock_for_tdms.acquire()
                    # 要判断是否存在Speed通道（别的传感器可能会先写入了speed）
                    if save_type == 'hdf5':
                        # 确认Speed是否已存在
                        if hdf5_data_confirm(colormap_filename, group_name, 'Speed'):
                            write_hdf5(colormap_filename, group_name, 'Speed', threed_os['xValue'])
                        else:
                            pass
                    else:
                        if tdms_data_confirm(colormap_filename, group_name, 'Speed'):
                            write_tdms(colormap_filename, group_name, 'Speed', threed_os['xValue'])
                    # sensor_index是0或1\

                    for i in range(len(threed_os['zValue'])):
                        if save_type == 'hdf5':
                            write_hdf5(colormap_filename, group_name,
                                       'Vib' + str(sensor_index) + '[' + str(i) + ']',
                                       threed_os['zValue'][i], colormap_save_properties)
                        else:
                            write_tdms(colormap_filename, group_name,
                                       'Vib' + str(sensor_index) + '[' + str(i) + ']',
                                       threed_os['zValue'][i], colormap_save_properties)
                    qDAQ_logger.info(
                        'order colormap of sensor{} saved, filepath:{}'.format(sensor_index + 1,
                                                                               colormap_filename))
                except Exception:
                    gv_dict_status["code"] = 3000
                    gv_dict_status["msg"] = "彩图数据保存失败!"
                    traceback.print_exc()
                    qDAQ_logger.error(
                        "colormap data save failed, failed msg:" + traceback.format_exc())
                finally:
                    # 释放锁以便其它传感器的数据可以写入
                    lock_for_tdms.release()
                    qDAQ_logger.info("order colormap save operation done")
            else:
                qDAQ_logger.info("order colormap not saved!")

            if data['testNameIndex'] == param.speedRecogInfo["test_count_except_dummy"]:
                qDAQ_logger.info("nvh process finish")
                gv_dict_flag[flag_index_dict["nvh_finish"]]=1
                break

            counter_data_queue = 0

            del twod_td
            del temp_td
            del twod_sf
            del temp_sf
            del threed_os
            del twod_oc
            del vib_rsp
            del vib_rsp_time
            del rev
            del rev_time
            del counter_td
            del counter_ar
            del counter_or

    # gc.collect()


def nvh_calc_process_for_fluctuation_norsp(in_Q, out_Q, gv_dict_status,
                                           param, test_result, sensor_index,
                                           time_click_start,
                                           file_info, colormap_save_properties, lock_for_tdms):
    global gv_dict_flag, trigger_ndarray, rpml_array, rpm_array, vib_ndarray

    gc.enable()
    qDAQ_logger.debug("nvh_calc_process:pid={},ppid={},thread={}".format(os.getpid(), os.getppid(),
                                                                         threading.current_thread().name))

    # 时域指标计算帧长
    calSize = param.timeDomainCalcInfo["calSize"]

    # 采样率
    sampleRate = param.taskInfo["sampleRate"]
    # 帧长
    sampsPerChan = param.taskInfo["sampsPerChan"]
    # 转速识别信息
    speedRecogInfo = param.speedRecogInfo

    gv_dict_status_temp = dict()

    # 该测试段从queue中取得的第几个data （从1开始）
    counter_data_queue = 0
    # 时域的计算帧长为calSize,该索引记录进行了几次计算，
    # 用于twod_td以及temp_td,和之前的counter_td意义相同
    index_twod_td = 0

    # 每一帧的振动信号长度
    len_vib_frame = 0

    while gv_dict_flag[flag_index_dict['nvhCalclation']]:
        try:
            data = in_Q.get(block=True, timeout=1)
        except Empty:
            continue

        if not data:
            continue

        if data['sectionEnd']:
            time_get_sectionend = time.time()
            qDAQ_logger.debug("nvh_get_sectionend:{}".format(time.time() - time_click_start))

        if counter_data_queue == 0:
            profiler = Profiler()
            profiler.start()
            time_get_sectionstart=time.time()
            # 测试段的第一帧数据
            # 左索引
            vib_start_index = data['vib_start_index']
            # 帧内左索引
            vib_left_index = vib_start_index
            # 更新右索引，右端索引处没有值
            vib_right_index = data['vib_right_index']

            testNameIndex = data["testNameIndex"]
            data['recogIndex'] = data["testNameIndex"]
            # maxT是测试段的最长时间
            maxT = speedRecogInfo["endTime"][testNameIndex] - speedRecogInfo["startTime"][testNameIndex]

            # endSpeed是测试段内的最大速度
            maxSpeed = speedRecogInfo["endSpeed"][testNameIndex]
            minSpeed = speedRecogInfo["startSpeed"][testNameIndex]

            # nvh计算的帧长,该值与参数配置/speed进程中的值不同，该值在不同的测试段不同
            # # 每一帧大约计算两次阶次谱
            # sampsPerChan = int(param.orderSpectrumCalcInfo["revNum"] * (
            #         2 - param.orderSpectrumCalcInfo["overlapRatio"]) / (minSpeed / 60) * sampleRate)
            # 每一帧大约计算一次阶次谱
            sampsPerChan = int(param.orderSpectrumCalcInfo["revNum"] / (minSpeed / 60) * sampleRate)

            # maxLen是该测试段时域计算时按帧长calSize最多计算多少次
            maxLen = int(maxT * sampleRate // calSize) + 2
            # 创建空的twodtd和空的temp_td，创建的容量保证能放下测试段内的数据，
            # 将来测试段结束后再进行截取，然后再进行计算
            twod_td = create_empty_twodtd_for_share(param.timeDomainCalcInfo, sensor_index, maxLen,
                                                    indicator_diagnostic=
                                                    param.speedRecogInfo['initial_indicator_diagnostic'][
                                                        data['recogIndex']])
            temp_td = create_empty_temptd_for_share(maxLen)
            # 一个测试段内计算了多少个twod_td计算了多少次 ，新测试段开始时赋值为0
            index_twod_td = 0

            # 该测试段
            max_circle = maxT * maxSpeed / 60
            # 记录按圈计算过程中每一个轴计算过的次数
            twod_sf_counter = dict()
            twod_sf = create_empty_twodsf_for_const(param.statFactorCalcInfo, sensor_index, max_circle,
                                                    indicator_diagnostic=
                                                    param.speedRecogInfo['initial_indicator_diagnostic'][
                                                        data['recogIndex']])
            temp_sf = create_empty_tempsf_for_const(param.statFactorCalcInfo,sensor_index, max_circle)

            # 每32圈计算一次，overlapRatio为0.75 即步进长度为8
            # osMaxLen = int(maxT * maxSpeed / 60 // (param.orderSpectrumCalcInfo['revNum'] * (
            #         1 - param.orderSpectrumCalcInfo['overlapRatio']))) + 1
            osMaxLen = int(maxT * sampleRate // sampsPerChan) + 2
            threed_os = create_empty_threedos_for_share(param.orderSpectrumCalcInfo, param.taskInfo,
                                                        sensor_index, osMaxLen,
                                                        db_flag=param.basicInfo['dBFlag'],
                                                        indicator_diagnostic=param.speedRecogInfo[
                                                            'initial_indicator_diagnostic'][
                                                            data['recogIndex']])
            twod_oc = create_empty_twodoc_for_share(param.orderCutCalcInfo,
                                                    param.taskInfo,
                                                    sensor_index, osMaxLen,
                                                    db_flag=param.basicInfo['dBFlag'],
                                                    indicator_diagnostic=
                                                    param.speedRecogInfo['initial_indicator_diagnostic'][
                                                        data['recogIndex']])
            # 每次重采样取的trigger对应的圈数和时间从该位置截取
            trigger_right_index_backup = 0
            # twod_td计算的左索引，测试段开始时更新为测试段开始振动点
            last_calc_index = vib_start_index
            # 进行了几帧角度重采样，
            counter_ar = 0
            # 进行了几帧时间域的计算，并不一定等于时间域计算方法的运行次数，
            # 调用一次方法，在方法内可能进行了多帧运算
            counter_td = 0
            # 进行了几帧阶次谱的运算
            counter_or = 0

            # 测试段内每个振动点对应的圈数，第一个振动点为0圈
            rev = np.zeros(int(maxT * sampleRate) + 1)
            len_vib_frame = vib_right_index - vib_start_index
            rev_index = 0
            # rev[rev_index:(rev_index + len_vib_frame)] = np.arange(len_vib_frame) * (
            #         speed / 60 / sampleRate)
            # rev_index += len_vib_frame

            # 保存测试段内的转速曲线
            rpm = list()
            rpml = list()


        else:
            # 不是测试段的第一帧数据，更新振动信号和trigger的右索引
            vib_left_index = vib_right_index
            vib_right_index = data['vib_right_index']
            len_vib_frame = vib_right_index - vib_left_index
            # rev[rev_index:(rev_index + len_vib_frame)] = rev[rev_index - 1] + np.arange(1,
            #                                                                             len_vib_frame + 1) * (
            #                                                      speed / 60 / sampleRate)
            # rev_index += len_vib_frame

            # 也许将来要配置dummy段
            data['recogIndex'] = data["testNameIndex"]

        #
        # qDAQ_logger.debug(data["speed"])
        # 测试段内的第几个data
        counter_data_queue += 1

        # 进行测试段内数据的计算
        # 以传来的第一个值作为测试段的0时刻点，作为0圈数点
        # 角度重采样，一测试段内传进来的第一个vib点为0时刻点，0圈数点

        # 测试段内vib的长度
        len_vib = vib_right_index - vib_start_index

        # 分帧计算时域信息
        try:
            # time domain indicators
            twod_td, temp_td, counter_td, index_twod_td, last_calc_index = twod_time_domain_for_share(
                vib_ndarray,
                last_calc_index,
                vib_right_index,
                calSize,
                param.timeDomainCalcInfo["indicatorNestedList"][sensor_index],
                param.timeDomainCalcInfo["refValue"][sensor_index],
                twod_td,
                temp_td,
                counter_td,
                sampleRate,
                index_twod_td)
        except Exception:
            gv_dict_status["code"] = 3000
            gv_dict_status["msg"] = "二维时间域指标计算错误!"
            qDAQ_logger.warning(
                "sensor " + str(
                    sensor_index + 1) + ", test: " + (
                    param.speedRecogInfo['testName'][data['recogIndex']]) +
                "2D time domain indicators calculation failed, failed msg:" + traceback.format_exc())
            break

        try:
            # 转速计算
            while vib_start_index + (counter_or + 1) * sampsPerChan <= vib_right_index:
                # speed = speed_calibration(vib_ndarray[vib_start_index + counter_or * sampsPerChan:
                #                                       vib_start_index + (counter_or + 1) * sampsPerChan],
                #                           speedRecogInfo["startSpeed"][testNameIndex],
                #                           speedRecogInfo["endSpeed"][testNameIndex],
                #                           speedRecogInfo["order"][testNameIndex], sampleRate)
                speed=speed_calibration_with_orderlist(vib_ndarray[vib_start_index + counter_or * sampsPerChan:
                                                      vib_start_index + (counter_or + 1) * sampsPerChan],
                                                       speedRecogInfo["startSpeed"][testNameIndex],
                                                       speedRecogInfo["endSpeed"][testNameIndex],
                                                       speedRecogInfo["order"][testNameIndex], sampleRate
                                                       )
                # speed=(speedRecogInfo["startSpeed"][testNameIndex]+speedRecogInfo["endSpeed"][testNameIndex])/2

                rpm.append(speed)
                rpml.append(((counter_or + 0.5) * sampsPerChan + vib_start_index) / sampleRate)

                if counter_or == 0:

                    rev[rev_index:(rev_index + sampsPerChan)] = np.arange(sampsPerChan) * (
                            speed / 60 / sampleRate)
                    rev_index += sampsPerChan
                else:
                    rev[rev_index:(rev_index + sampsPerChan)] = rev[rev_index - 1] + np.arange(1,
                                                                                               sampsPerChan + 1) * (
                                                                        speed / 60 / sampleRate)
                    rev_index += sampsPerChan

                if sensor_index == 0:
                    gv_dict_status_temp['data'] = gv_dict_status['data']
                    gv_dict_status_temp['data']['x'] = float(rpml[-1])
                    gv_dict_status_temp['data']['y'] = float(rpm[-1])
                    gv_dict_status['data'] = gv_dict_status_temp['data']

                # 阶次谱计算
                threed_os, twod_oc, counter_or = order_spectrum_for_const_fluctuation(threed_os,
                                                                                      twod_oc,
                                                                                      counter_or,
                                                                                      vib_ndarray[
                                                                                      vib_start_index + counter_or * sampsPerChan:
                                                                                      vib_start_index + (
                                                                                              counter_or + 1) * sampsPerChan],
                                                                                      param.orderSpectrumCalcInfo,
                                                                                      param.orderCutCalcInfo,
                                                                                      sensor_index,
                                                                                      vib_start_index + counter_or * sampsPerChan,
                                                                                      sampleRate,
                                                                                      speed,
                                                                                      db_flag=
                                                                                      param.basicInfo[
                                                                                          'dBFlag'])
            if sensor_index == 0 and data['sectionEnd']:
                gv_dict_status_temp['data'] = gv_dict_status['data']
                # 起始时刻从参数配置中拿就好
                gv_dict_status_temp["data"]["startX"].append(
                    float(speedRecogInfo["startTime"][testNameIndex]))
                # 没必要记下起始转速，从参数配置中拿转速，前端可以正常显示就好
                gv_dict_status_temp["data"]["startY"].append(
                    float(speedRecogInfo["startSpeed"][testNameIndex] + speedRecogInfo["endSpeed"][
                        testNameIndex]) / 2)
                gv_dict_status_temp["data"]["endX"].append(
                    float(rpml[-1]))
                gv_dict_status_temp["data"]["endY"].append(
                    float(rpm[-1]))
                gv_dict_status_temp["data"]["testName"].append(
                    param.speedRecogInfo["testName"][testNameIndex])
                gv_dict_status['data'] = gv_dict_status_temp['data']

        except Exception:
            gv_dict_status["code"] = 3000
            gv_dict_status["msg"] = "阶次域指标计算错误!"
            qDAQ_logger.warning(
                "sensor " + str(
                    sensor_index + 1) + ", test: " + (
                    param.speedRecogInfo['testName'][data['recogIndex']]) +
                ": 2/3D order domain indicators calculation failed, failed msg:" + traceback.format_exc())
            break

        try:
            # 按圈计算
            # twod_sf, temp_sf = twod_stat_factor_for_const_fluctuation(twod_sf, temp_sf, rev, rev_index,
            #                                                           vib_ndarray[
            #                                                           vib_start_index:vib_right_index],
            #                                                           twod_sf_counter,
            #                                                           vib_start_index,
            #                                                           sampleRate,
            #                                                           sensor_index,
            #                                                           param.statFactorCalcInfo)
            twod_sf, temp_sf = twod_stat_factor_for_const_fluctuation(twod_sf, temp_sf, rev, rev_index,
                                                                      vib_ndarray[
                                                                      vib_start_index:vib_right_index],
                                                                      twod_sf_counter,
                                                                      vib_start_index,
                                                                      sampleRate,
                                                                      sensor_index,
                                                                      param.statFactorCalcInfo)

        except Exception:
            gv_dict_status["code"] = 3000
            gv_dict_status["msg"] = "二维按圈计算指标错误!"
            qDAQ_logger.warning("sensor " + str(sensor_index + 1) + ", test: " + (
                param.speedRecogInfo['testName'][data[
                    'recogIndex']]) + "2D indicators by revolution calculation failed, failed msg:"
                                + traceback.format_exc())
            break

        if data['sectionEnd']:

            twod_td, temp_td, twod_sf, temp_sf, threed_os, twod_oc = update_nvh_data_for_thread(
                twod_td, temp_td, index_twod_td,
                twod_sf, temp_sf, twod_sf_counter,
                threed_os, twod_oc, counter_or)

            # prepare test finished
            try:
                # update 1D time domain indicators into final test result
                if test_result['resultData'][sensor_index]['dataSection'][data['testNameIndex']][
                    'onedData'] is None:
                    test_result['resultData'][sensor_index]['dataSection'][data['testNameIndex']][
                        'onedData'] = list()
                test_result['resultData'][sensor_index]['dataSection'][data['testNameIndex']][
                    'onedData'].extend(
                    oned_time_domain(temp_td, calSize,
                                     param.timeDomainCalcInfo, sensor_index,
                                     db_flag=param.basicInfo['dBFlag']))
                qDAQ_logger.info(
                    "1D time domain indicators calculation of sensor{}, test: {} finished".format(
                        sensor_index + 1,
                        param.speedRecogInfo['testName'][
                            data['recogIndex']]))
            except Exception:
                gv_dict_status["code"] = 3000
                gv_dict_status["msg"] = "一维时间域指标计算错误!"
                qDAQ_logger.warning(
                    "sensor " + str(
                        sensor_index + 1) + ", test: " + (
                        param.speedRecogInfo['testName'][data['recogIndex']]) +
                    ": 1D time domain indicators calculation failed, failed msg:" + traceback.format_exc())
            rampQuality = ramp_quality_for_const(rpml, rpm, speedRecogInfo, data["recogIndex"])
            test_result['resultData'][sensor_index]['dataSection'][data['testNameIndex']][
                'onedData'].append(rampQuality)
            test_result['resultData'][sensor_index]['dataSection'][data['testNameIndex']][
                'onedData'].append(
                {"name": "Speed", "unit": "", "value": np.mean(rpm), "indicatorDiagnostic": 1}
            )
            try:
                # update 1D indicators by revolution into final test result
                if test_result['resultData'][sensor_index]['dataSection'][
                    data['testNameIndex']]['onedData'] is None:
                    test_result['resultData'][sensor_index]['dataSection'][
                        data['testNameIndex']][
                        'onedData'] = list()
                # 计算一维按圈计算指标并更新二维结果
                # twod_sf, oned_sf = average_oned_stat_factor(twod_sf, param.statFactorCalcInfo,
                #                                             sensor_index)
                # oned_sf=oned_stat_factor(temp_sf,param.statFactorCalcInfo,sensor_index)
                oned_sf,twod_sf=oned_stat_factor_mean_for_const(temp_sf, twod_sf, param.statFactorCalcInfo, sensor_index)
                test_result['resultData'][sensor_index]['dataSection'][
                    data['testNameIndex']]['onedData'].extend(oned_sf)
                qDAQ_logger.info(
                    "1D indicators by revolution calculation of sensor{}, test: {} finished".format(
                        sensor_index + 1,
                        param.speedRecogInfo['testName'][
                            data['recogIndex']]))
            except Exception:
                gv_dict_status["code"] = 3000
                gv_dict_status["msg"] = "一维按圈计算指标计算错误!"
                qDAQ_logger.warning(
                    "sensor " + str(
                        sensor_index + 1) + ", test: " + (
                        param.speedRecogInfo['testName'][data['recogIndex']]) +
                    ": 1D indicators by revolution calculation failed, failed msg:" + traceback.format_exc())

            # 更新二维时间域结果的x轴
            twod_td = convert_time_speed_for_const(twod_td)
            test_result['resultData'][sensor_index]['dataSection'][data['testNameIndex']][
                'twodTD'] = twod_td
            qDAQ_logger.info(
                "2D time domain indicators calculation of sensor{}, test: {} finished".format(
                    sensor_index + 1,
                    param.speedRecogInfo['testName'][
                        data['recogIndex']]))
            # 更新二维按圈计算指标的x轴
            twod_sf = convert_time_speed_for_const(twod_sf, indicator_num=param.statFactorCalcInfo[
                'indicatorNum'][sensor_index])
            # 添加二维按圈计算指标到二维时间域指标中
            test_result['resultData'][sensor_index]['dataSection'][data['testNameIndex']][
                'twodTD'].extend(twod_sf)
            qDAQ_logger.info(
                "2D indicators by revolution calculation of sensor{} finished".format(sensor_index + 1))

            # 向二维结果数据中保存转速曲线
            test_result['resultData'][sensor_index]['dataSection'][data['testNameIndex']][
                'twodTD'].append({
                "xName": "Time",
                "xUnit": "s",
                "xValue": rpml,
                "yName": "Speed",
                "yUnit": "RPM",
                "yValue": rpm,
                "indicatorDiagnostic": 1
            })

            # update stop and end time of present test into final test result
            test_result['resultData'][sensor_index]['dataSection'][data['testNameIndex']]['startTime'] = \
                data['vib_start_index'] / sampleRate
            test_result['resultData'][sensor_index]['dataSection'][data['testNameIndex']]['endTime'] = \
                data['vib_right_index'] / sampleRate

            try:
                # update 2D order spectrum into final test result
                twod_os = twod_order_spectrum(threed_os, param.taskInfo, sensor_index)
                qDAQ_logger.info(
                    "2D order spectrum calculation of sensor{}, test: {} finished".format(
                        sensor_index + 1,
                        param.speedRecogInfo[
                            'testName'][
                            data[
                                'recogIndex']]))
            except Exception:
                gv_dict_status["code"] = 3000
                gv_dict_status["msg"] = "二维阶次谱计算错误!"
                qDAQ_logger.warning(
                    "sensor " + str(
                        sensor_index + 1) + ", test: " + (
                        param.speedRecogInfo['testName'][data['recogIndex']]) +
                    ": 2D order spectrum calculation failed, failed msg:" + traceback.format_exc())

            try:
                # update 2D order cepstrum into final test result
                logging.info("len(threed_os['xValue']):{}".format(len(threed_os['xValue'])))
                logging.info("len(twod_oc['xValue']):{}".format(len(twod_oc[0]['xValue'])))
                test_result['resultData'][sensor_index]['dataSection'][
                    data['testNameIndex']]['twodCeps'] = \
                    cepstrum(twod_os, param.cepstrumCalcInfo, param.taskInfo,
                             sensor_index, db_flag=param.basicInfo['dBFlag'])
                qDAQ_logger.info(
                    "2D order cepstrum calculation of sensor{}, test: {} finished".format(
                        sensor_index + 1,
                        param.speedRecogInfo[
                            'testName'][
                            data[
                                'recogIndex']]))
            except Exception:
                gv_dict_status["code"] = 3000
                gv_dict_status["msg"] = "二维倒阶次谱计算错误!"
                qDAQ_logger.warning(
                    "sensor " + str(
                        sensor_index + 1) + ", test: " + (
                        param.speedRecogInfo['testName'][data['recogIndex']]) +
                    ": 2D order cepstrum calculation failed, failed msg:" + traceback.format_exc())

            try:
                # update 1D order domain indicators into final test result
                test_result['resultData'][sensor_index]['dataSection'][
                    data['testNameIndex']]['onedData'] += oned_order_spectrum(twod_os,
                                                                              param.onedOSCalcInfo,
                                                                              param.taskInfo,
                                                                              param.modulationDepthCalcInfo,
                                                                              sensor_index,
                                                                              db_flag=param.basicInfo[
                                                                                  'dBFlag'])
                # 放入工况评估指标
                # test_result['resultData'][sensor_index]['dataSection'][
                #     data['testNameIndex']]['onedData'].append(data['RampQuality'])
                qDAQ_logger.info(
                    "1D order indicators calculation of sensor{}, test: {} finished".format(
                        sensor_index + 1,
                        param.speedRecogInfo['testName'][
                            data['recogIndex']]))
            except Exception:
                gv_dict_status["code"] = 3000
                gv_dict_status["msg"] = "一维阶次谱计算错误!"
                qDAQ_logger.warning(
                    "sensor " + str(
                        sensor_index + 1) + ", test: " + (
                        param.speedRecogInfo['testName'][data['recogIndex']]) +
                    ": 1D order cutting calculation failed, failed msg:" + traceback.format_exc())
            try:
                # 更新二维阶次谱和三维阶次谱的阶次信息（一定要在阶次切片结果计算完成以后）
                if param.speedCalcInfo['speedRatio'] != 1:
                    if twod_os:
                        twod_os[0]['xValue'] /= param.orderSpectrumCalcInfo['convertOrder']
                        threed_os['yValue'] = param.orderSpectrumCalcInfo['convertOrder']
                # 更新结果到结果数据中
                # 更新结果到结果数据中
                # 更新结果到结果数据中（包括dB转换）
                if param.basicInfo['dBFlag']:
                    twod_os[0]['yUnit'] = 'dB'
                    twod_os[0]['yValue'] = db_convertion(twod_os[0]['yValue'],
                                                         param.taskInfo['refValue'][
                                                             sensor_index])
                # 更新yValue由ndarray变为list(),
                # 之前会用twod_os[0]['yValue']进行运算，在运算完成之后进行更新
                twod_os[0]['yValue'] = twod_os[0]['yValue'].tolist()
                test_result['resultData'][sensor_index]['dataSection'][data['testNameIndex']][
                    'twodOS'] = twod_os
                qDAQ_logger.info(
                    "order conversion of sensor{}, test: {} finished".format(sensor_index + 1,
                                                                             param.speedRecogInfo[
                                                                                 'testName'][
                                                                                 data[
                                                                                     'recogIndex']]))
            except Exception:
                gv_dict_status["code"] = 3000
                gv_dict_status["msg"] = "阶次转换错误!"
                qDAQ_logger.warning(
                    "sensor " + str(
                        sensor_index + 1) + ", test: " + (
                        param.speedRecogInfo['testName'][data['recogIndex']]) +
                    ": order convertion failed, failed msg:" + traceback.format_exc())
            # 更新二维阶次切片和三维阶次谱的X轴信息
            twod_oc, threed_os = convert_time_speed_for_const(twod_oc, threed_os=threed_os)
            test_result['resultData'][sensor_index]['dataSection'][data['testNameIndex']][
                'twodOC'] = twod_oc
            qDAQ_logger.info(
                "nvh calculation of sensor{}, test: {} finished ".format(sensor_index + 1,
                                                                         param.speedRecogInfo[
                                                                             'testName'][
                                                                             data['recogIndex']]))
            del threed_os["tempXi2"]

            if twod_os:
                error_flag = 0
            else:
                error_flag = 1
            while gv_dict_flag[flag_index_dict['dataPack']]:
                try:
                    # 将test_result内的数据变为可以json化的数据
                    out_Q.put(
                        {"sensorIndex": sensor_index,
                         "testNameIndex": data['testNameIndex'],
                         "data": test_result,
                         "error": error_flag}, block=True, timeout=1)
                    break
                except Full:
                    continue

            qDAQ_logger.debug("nvh获取到测试段{}最后一帧到已经将数据放入queue:{}".format(data["testNameIndex"],time.time() - time_get_sectionend))
            qDAQ_logger.debug("拿到测试段{}第一帧到已经将数据放入queue:{}".format(data["testNameIndex"],time.time() - time_get_sectionstart))
            profiler.stop()
            profiler.print(show_all=True)
            if param.dataSaveFlag['colorMap']:
                # store the colormap
                try:
                    # 2 sensor can not deal with colormap data like 1 sensor
                    if save_type == 'hdf5':
                        colormap_filename = os.path.join(file_info[1],
                                                         param.basicInfo["fileName"] + '.h5')
                    else:
                        colormap_filename = os.path.join(file_info[1],
                                                         param.basicInfo["fileName"] + '.tdms')
                    group_name = 'ATEOLOSMAP_' + param.speedRecogInfo['testName'][data['recogIndex']]
                    single_folder_confirm(file_info[1])
                    # 每个传感器的彩图数据需要分开写，所以写入一个传感器的时候需要防止其它进程写入
                    lock_for_tdms.acquire()
                    # 要判断是否存在Speed通道（别的传感器可能会先写入了speed）
                    if save_type == 'hdf5':
                        # 确认Speed是否已存在
                        if hdf5_data_confirm(colormap_filename, group_name, 'Speed'):
                            write_hdf5(colormap_filename, group_name, 'Speed', threed_os['xValue'])
                        else:
                            pass
                    else:
                        if tdms_data_confirm(colormap_filename, group_name, 'Speed'):
                            write_tdms(colormap_filename, group_name, 'Speed', threed_os['xValue'])
                    # sensor_index是0或1\

                    for i in range(len(threed_os['zValue'])):
                        if save_type == 'hdf5':
                            write_hdf5(colormap_filename, group_name,
                                       'Vib' + str(sensor_index) + '[' + str(i) + ']',
                                       threed_os['zValue'][i], colormap_save_properties)
                        else:
                            write_tdms(colormap_filename, group_name,
                                       'Vib' + str(sensor_index) + '[' + str(i) + ']',
                                       threed_os['zValue'][i], colormap_save_properties)
                    qDAQ_logger.info(
                        'order colormap of sensor{} saved, filepath:{}'.format(sensor_index + 1,
                                                                               colormap_filename))
                except Exception:
                    gv_dict_status["code"] = 3000
                    gv_dict_status["msg"] = "彩图数据保存失败!"
                    traceback.print_exc()
                    qDAQ_logger.warning(
                        "colormap data save failed, failed msg:" + traceback.format_exc())
                finally:
                    # 释放锁以便其它传感器的数据可以写入
                    lock_for_tdms.release()
                    qDAQ_logger.info("order colormap save operation done")
            else:
                qDAQ_logger.info("order colormap not saved!")

            if data['testNameIndex'] == param.speedRecogInfo["test_count_except_dummy"]:
                qDAQ_logger.info("nvh process finish")
                gv_dict_flag[flag_index_dict["nvh_finish"]]=1
                break

            counter_data_queue = 0

            del twod_td
            del temp_td
            del twod_sf
            del temp_sf
            del threed_os
            del twod_oc
            del counter_td
            del counter_ar
            del counter_or

    gc.collect()
