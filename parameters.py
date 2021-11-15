#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2021/05/27 16:44
@Author : Wall@Synovate
@Email  : qxu@sonustc.com
update: 2021/06/16 16:01
功能：该模块主要包含了配置参数的类定义和读取方法
"""

import sys
import os
import numpy as np
from scipy import signal
import logging
import traceback
import configparser
from utils import decrypt_data, read_json
from common_info import baseFolder, ni_device, xName, xUnit, sensor_count, speed_signal, encrypt_flag, \
    Cryptor, server_ip, version, constant_initial_indicator_diagnostic, \
    drive_initial_indicator_diagnostic, coast_initial_indicator_diagnostic, cut_off_freq
from cytoolz import pluck, concat
from collections import defaultdict
from numba import njit, jit, prange


def folder_info_update():
    """
    # update more detailed folder info
    # qDAQ运行所需要的的文件夹信息，用于存储不同的数据，包括:
    1. Data：原始数据保存路径
    2. Report：彩图数据保存路径
    3. JSON_InfoMissing：信息缺失导致的发送失败的结果数据保存路径
    4. JSON_NetError：网络故障导致发送失败的结果数据保存路径
    5. temp：临时数据的保存路径（包括转速曲线，XML，结果数据）
    返回：
    1. folder_info：文件夹目录信息
    """
    folder_info = dict()
    folder_info["rawData"] = os.path.join(baseFolder, "Data")
    folder_info["reportData"] = os.path.join(baseFolder, "Report")
    # 结果数据上传失败时本地保存的位置
    folder_info["reportInfoMissing"] = os.path.join(baseFolder, "Error/JSON_InfoMissing")
    folder_info["reportNetError"] = os.path.join(baseFolder, "Error/JSON_NetError")
    # ftp或本地数据移动失败是本地保存的位置
    folder_info["uploadError"] = os.path.join(baseFolder, "Error/rawData_uploadError")
    folder_info["analysisError"] = os.path.join(baseFolder, "Error/rawData_analysisError")
    folder_info["temp"] = os.path.join(baseFolder, "temp")
    return folder_info


def basic_info_update(basic_info):
    """
    功能：确认并更新基础信息
    输入：配置文件中的基础信息
    输出：确认后的基础信息
    """
    if "unit" not in basic_info.keys():
        # 不存在这个信息则直接认为不需要转换
        basic_info['dBFlag'] = 0
    else:
        if basic_info['unit'].lower() == 'db':
            # 如果单位为dB， 则认为需要转换
            basic_info['dBFlag'] = 1
        else:
            basic_info['dBFlag'] = 0
    return basic_info


def task_info_update(task_info):
    """
    update the taskInfo based on raw information from config file
    1. add ni device info for physicalChannels
    2. 更新单位索引信息
    功能：完善数据采集信息，主要是结合本地配置文件（config.ini）完善采集设备信息
    返回：更新后的NI DAQmx数据采集任务配置信息
    """
    task_info = defaultdict(list, task_info)
    # 参考振动值
    ref_g = 1e-6 / 9.8  # unit: m/s^2
    # 参考声压
    ref_sound_pressure = 20e-6  # unit: Pa
    for i in range(len(task_info["physicalChannels"])):
        task_info["physicalChannels"][i] = "/".join(
            [ni_device, task_info["physicalChannels"][i]])
    # 根据通道名称更新单位索引信息(目前只支持振动信号（Vib开头）和声压信号（Mic开头）),索引信息的数量最终会与基础信息里的传感器数量一致
    task_info['indicatorsUnitChanIndex'] = list()
    for channel_index, channel_name in enumerate(task_info['channelNames']):
        if channel_name.lower().startswith('vib'):
            task_info['indicatorsUnitChanIndex'].append(channel_index)
            task_info['sensorChan'].append(channel_name)
            task_info['refValue'].append(ref_g)
        elif channel_name.lower().startswith('mic'):
            task_info['indicatorsUnitChanIndex'].append(channel_index)
            task_info['sensorChan'].append(channel_name)
            task_info['refValue'].append(ref_sound_pressure)
        elif channel_name.lower().startswith('umic'):
            task_info['indicatorsUnitChanIndex'].append(channel_index)
            task_info['sensorChan'].append(channel_name)
            task_info['refValue'].append(ref_sound_pressure)
    # 设置线性换算参数
    if "lineScale" not in task_info.keys():
        # 如果不存在的话
        task_info["lineScale"] = dict()
        task_info["lineScale"]["flag"] = list()
        task_info["lineScale"]["slope"] = list()
        task_info["lineScale"]["intercept"] = list()
        task_info["lineScale"]["scaleUnits"] = list()
        for unit in task_info["units"]:
            task_info["lineScale"]["scaleUnits"].append(unit)
            task_info["lineScale"]["flag"].append(0)
            task_info["lineScale"]["slope"].append(1)
            task_info["lineScale"]["intercept"].append(0)
    # 更新通道单位
    for i, unit in enumerate(task_info["lineScale"]["scaleUnits"]):
        if task_info["lineScale"]["flag"][i]:
            # 如果需要换算则将通道的单位更新（根据换算后的单位）
            task_info["units"][i] = unit
    if {'Speed', 'Sin', 'Cos'} & set(task_info['channelNames']):
        # 存在转速信号
        task_info['speedFlag'] = 1
    else:
        task_info['speedFlag'] = 0
    task_info['sensorName'] = [name for name in task_info['channelNames'] if name.lower()[:3] in ('mic', 'umi', 'vib')]
    return task_info


def speed_calc_info_update(speed_calc_info):
    """
    update speed calculation info, add 2 factors to avoid re calculation inside loop
    功能：提前计算好需要重复计算的参数以加快实时模式的速度，包括：
    1. 转速计算参数（rpmFactor），60*平均点数/ppr
    2. 取点参数（resampleFactor），该参数决定了转速曲线的点数
    3. 转速比（speedRatio），用于结果数据中的转速转换，默认为1.0
    返回：更新后的转速计算信息
    """
    for i in range(testSectionNum := len(speed_calc_info)):
        # speed_calc_info[i] = defaultdict(list, speed_calc_info[i])
        speed_calc_info[i]['rpmFactor'] = 60 * speed_calc_info[i]['averageNum'] / speed_calc_info[i][
            'ppr']
        speed_calc_info[i]['step'] = speed_calc_info[i]['averageNum'] - speed_calc_info[i]['overlap']
        if speed_calc_info[i]['overlap'] < speed_calc_info[i]['averageNum']:
            speed_calc_info[i]['resampleFactor'] = speed_calc_info[i]['averageNum'] - \
                                                   speed_calc_info[i]['overlap']
        else:
            # 避免overlap过大导致无法进行计算（不能超过平均点数）
            speed_calc_info[i]['overlap'] = 0
            speed_calc_info[i]['resampleFactor'] = speed_calc_info[i]['averageNum']
        # 确认转速比
        if 'speedRatio' not in speed_calc_info[i]:
            # 若没有设置转速比则默认是1.0
            speed_calc_info[i]['speedRatio'] = 1.0
        if speed_calc_info[i]['speedRatio'] <= 0:
            # 若转速比异常（小于等于0）则认为设置出错，强制设置为1
            speed_calc_info[i]['speedRatio'] = 1.0
        return speed_calc_info


def speed_recog_info_update(speed_recog_info):
    """
    # just update some speed recognition info to avoid re calculation inside loop
    功能：提前计算转速识别所需要的的参数，包括：
    1. minRange 和 maxRange：变速段对应的上上下限
    2. slope：目标转速的斜率（预期）
    3. speedPattern：转速类型，1表示恒速段；2表示升速段；3表示降速段
    4. minSpeed：测试段最小转速
    返回：
    1. 更新后的工况识别参数
    """
    # speed_recog_info['minRange'] = list()
    # speed_recog_info['maxRange'] = list()
    # speed_recog_info['slope'] = list()
    # speed_recog_info['speedPattern'] = list()
    # speed_recog_info['minSpeed'] = list()
    # speed_recog_info['notDummyFlag'] = list()
    # speed_recog_info['initial_indicator_diagnostic'] = list()
    # speed_recog_info["minTorque"] = list()
    # speed_recog_info["maxTorque"] = list()
    # 使用默认字典精简代码
    speed_recog_info = defaultdict(list, speed_recog_info)
    # 确认扭矩识别信息
    speed_recog_info.setdefault('torqueRecogFlag', list(np.zeros(len(speed_recog_info["testName"]), dtype='int')))
    # 定义转速范围（用于区分恒速和变速段）
    speed_recog_info.setdefault('speedRange', 100)
    for i in range(len(speed_recog_info['startSpeed'])):
        # 统计有多少个测试段（除dummy）
        if speed_recog_info["testName"][i].lower().startswith("dummy"):
            # 如果是dummy段
            speed_recog_info['notDummyFlag'].append(0)
        else:
            speed_recog_info['minSpeed'].append(
                min(speed_recog_info['startSpeed'][i], speed_recog_info['endSpeed'][i]))
            speed_recog_info['notDummyFlag'].append(1)
        if abs(speed_recog_info['endSpeed'][i] - speed_recog_info['startSpeed'][i]) <= speed_recog_info[
            'speedRange']:
            # 恒速段
            # 设置指标的初始化结果
            speed_recog_info['initial_indicator_diagnostic'].append(
                constant_initial_indicator_diagnostic)
            # 设置转速类型（1为恒速，2为升速，3为降速）
            speed_recog_info['speedPattern'].append(1)
            speed_recog_info['minRange'].append(0)
            speed_recog_info['maxRange'].append(0)
            speed_recog_info['slope'].append(0)
            if speed_recog_info["torqueRecogFlag"][i]:
                # 如果需要扭矩识别
                speed_recog_info["minTorque"].append(
                    min(speed_recog_info['startTorque'][i], speed_recog_info['endTorque'][i]))
                speed_recog_info["maxTorque"].append(
                    max(speed_recog_info['startTorque'][i], speed_recog_info['endTorque'][i]))
            else:
                speed_recog_info["minTorque"].append(0)
                speed_recog_info["maxTorque"].append(0)
        else:
            # 变速段
            # 针对变速段限制不用扭矩识别（如果开启就是认为设置错误，强制修正）
            speed_recog_info["torqueRecogFlag"][i] = 0
            speed_recog_info["minTorque"].append(0)
            speed_recog_info["maxTorque"].append(0)
            slope = (speed_recog_info['endSpeed'][i] - speed_recog_info['startSpeed'][i]) / \
                    speed_recog_info['expectT'][
                        i]
            speed_recog_info['slope'].append(slope)
            speed_recog_info['minRange'].append(
                slope * (speed_recog_info['expectT'][i] - speed_recog_info['minT'][i]))
            speed_recog_info['maxRange'].append(
                slope * (speed_recog_info['expectT'][i] - speed_recog_info['maxT'][i]))
            if speed_recog_info['endSpeed'][i] - speed_recog_info['startSpeed'][i] > speed_recog_info[
                'speedRange']:
                # 升速段
                speed_recog_info['speedPattern'].append(2)
                # 设置指标的初始化结果
                speed_recog_info['initial_indicator_diagnostic'].append(
                    drive_initial_indicator_diagnostic)
            else:
                # 降速段
                speed_recog_info['speedPattern'].append(3)
                # 设置指标的初始化结果
                speed_recog_info['initial_indicator_diagnostic'].append(
                    coast_initial_indicator_diagnostic)
    speed_recog_info['overallMinSpeed'] = min(speed_recog_info['minSpeed'])
    speed_recog_info["test_count_except_dummy"] = sum(speed_recog_info['notDummyFlag'])

    # 对于恒速电机，要更新时间
    if "waitTime" in speed_recog_info:
        speed_recog_info["startTime"].append(speed_recog_info["waitTime"][0])
        speed_recog_info["endTime"].append(speed_recog_info["startTime"][0] + speed_recog_info["expectT"][0])

        for i in range(1, len(speed_recog_info["testName"])):
            speed_recog_info["startTime"].append(speed_recog_info["waitTime"][i] + speed_recog_info["endTime"][i - 1])
            speed_recog_info["endTime"].append(speed_recog_info["startTime"][i] + speed_recog_info["expectT"][i])
    return speed_recog_info


def filter_win(nfft, wtype):
    """
    # create window applied on signal before fft
    功能：生成窗函数（根据类型和点数），类型包括：
    1. hanning：汉宁窗
    2. hamming：海明窗
    3. kaiser：9阶凯塞窗
    4. blackman：布雷克曼窗
    5. bartlett：巴雷特窗
    6. flattop：平顶窗
    7. 其他：矩形窗（相当于不加窗）
    返回：
    1. 窗函数序列
    2. 修正因子（主要用于能量修正，以加窗后的能量可保持与未加窗时相同）
    """
    if wtype == 'hanning':
        win = np.hanning(nfft)
        win_factor = 1.633
    elif wtype == 'hamming':
        win = np.hamming(nfft)
        win_factor = 1.586
    elif wtype == 'kaiser':
        win = np.kaiser(nfft, 9)
        win_factor = 1.81
    elif wtype == 'blackman':
        win = np.blackman(nfft)
        win_factor = 1.812
    elif wtype == 'bartlett':
        # triangle window
        win = np.bartlett(nfft)
        win_factor = 1.732
    elif wtype == 'flattop':
        win = signal.flattop(nfft, sym=False)
        win_factor = 1.069
    elif wtype == 'hanning1':
        # create hanning window manually
        win = np.array(
            [0.5 - 0.5 * np.cos(2 * np.pi * n / (nfft - 1)) for n in range(nfft)])
        win_factor = 1.633
    else:
        win = np.ones(nfft)
        win_factor = 1.0
    return win, win_factor


def time_domain_calc_info_update(time_domian_calc_info, task_info, basic_info):
    """
    # update the RMS, Crest, Kurtosis, and Skewness unit
    功能：更新RMS，Crest，Kurtosis和Skewness，以及SPL或SPLA的单位，具体信息如下：
    1. RMS：单位来源于指定的采集通道
    2. Crest，Kurtosis，Skewness：无单位
    3. SPL：声压级，单位为dB
    4. SPL(A)：A计权声压级，单位为dB(A)
    返回：更新后的时间域指标计算参数
    """
    # time_domian_calc_info['indicatorUnit'] = list()
    # time_domian_calc_info["indicatorNestedList"] = list()
    for i in range(test_section_num := len(time_domian_calc_info)):
        time_domian_calc_info[i] = defaultdict(list, time_domian_calc_info[i])
        time_domian_calc_info[i]['refValue'] = task_info['refValue']
        if "Speed" in time_domian_calc_info[i]['vibrationIndicatorList']:
            time_domian_calc_info['vibrationIndicatorList'].pop("Speed")
        if "Speed" in time_domian_calc_info[i]['soundIndicatorList']:
            time_domian_calc_info['soundIndicatorList'].pop("Speed")
        for indicator in ['vibrationIndicatorList', 'soundIndicatorList']:
            for vib_info in time_domian_calc_info[i][indicator]:
                if vib_info['index'].lower()[:3] == 'rms':
                    if basic_info['dBFlag']:
                        vib_info['unit'] = ['dB']
                    else:
                        vib_info['unit'] = [task_info['units'][k] for k in task_info['indicatorsUnitChanIndex']]
                elif vib_info['index'].lower()[:6] == 'spl(a)': #todo 继续
                    vib_info['unit'] = ['dB(A)']
                elif vib_info['index'].lower()[:3] == 'spl':
                    vib_info['unit'] = ['dB']
                else:
                    vib_info['unit'] = ['']
        time_domian_calc_info[i]['xName'] = xName
        time_domian_calc_info[i]['xUnit'] = xUnit
        time_domian_calc_info[i]['calSize'] = int(task_info["sampleRate"] / time_domian_calc_info[i]["calRate"])
    return time_domian_calc_info


def order_spectrum_calc_info_update(order_spectrum_calc_info, speed_calc_info, speed_recog_info,
                                    task_info):
    """
    # just update some order spectrum calculation info to avoid re calculation inside loop
    功能：提前计算阶次谱计算的参数，包括：
    1. 确认重叠比例是否合理（overlapRatio）
    2. 角度域重采样间隔（dr_af)
    3. fft变换点数（nfft）
    4. fft变换步进点数（nstep）
    5. fft变换的窗函数（win），不同通道的振动信号可以加不同的窗函数
    6. fft变化的归一化因子（normFactor），可以决定需不需要补偿（根据winCorrectFlag来决定）
    7. 阶次谱x轴（order），根据变换点数和分辨率决定，然后根据设定的最大关注阶次进行截取
    返回：更新后的阶次谱计算参数
    """
    # 确认最大关注阶次是否合理
    if version == 1 or version == 2:
        max_order_available = (60 * task_info['sampleRate']) / (
                speed_recog_info['overallMinSpeed'] * 2 * 1.6384)
    else:
        max_order_available = (60 * task_info['sampleRate']) / (
                speed_recog_info['overallMinSpeed'] * 2)
    for i in range(testSectionNum := len(order_spectrum_calc_info)):
        order_spectrum_calc_info[i] = defaultdict(list, order_spectrum_calc_info[i])
        if order_spectrum_calc_info[i]['maxOrder'] > max_order_available:
            raise ValueError(
                f"test section {i}: max order {order_spectrum_calc_info[i]['maxOrder']} set error, should less than {max_order_available}")
        if order_spectrum_calc_info[i]['overlapRatio'] >= 1:
            order_spectrum_calc_info[i]['overlapRatio'] = 0
        # 定义角度域降采样之前的采样点角度间隔
        order_spectrum_calc_info[i]['dr_bf'] = speed_recog_info['overallMinSpeed'] / 60 / task_info[
            'sampleRate']
        # 定义角度域降采样之后的采样点角度间隔
        order_spectrum_calc_info[i]['dr_af'] = 1 / order_spectrum_calc_info[i][
            'maxOrder'] / 2 / 1.6384  # 1.6384=1.28*1.28
        # order_spectrum_calc_info[i]['dr_af'] = 1 / order_spectrum_calc_info[i][
        #     'maxOrder'] / 2 / 1.28  # 1.6384=1.28*1.28
        order_spectrum_calc_info[i]['nfft'] = int(
            order_spectrum_calc_info[i]['revNum'] / order_spectrum_calc_info[i]['dr_af'])
        order_spectrum_calc_info[i]['nstep'] = int(
            order_spectrum_calc_info[i]['nfft'] * (1 - order_spectrum_calc_info[i]['overlapRatio']))
        # 构建窗函数
        if len(order_spectrum_calc_info[i]['window']) != len(task_info['sensorChan']):
            raise ValueError("window count and sensor count not matched")
        if 'window' in order_spectrum_calc_info[i]:
            for j, winType in enumerate(order_spectrum_calc_info[i]['window']):
                win_temp = filter_win(order_spectrum_calc_info[i]['nfft'], winType)
                order_spectrum_calc_info[i]['win'].append(win_temp[0])
                # 是否开启修正系数 0是不开启 赋值会被压下来
                if order_spectrum_calc_info[i]['winCorrectFlag'][j]:
                    order_spectrum_calc_info[i]['normFactor'].append(
                        1.414 / order_spectrum_calc_info[i]['nfft'] * win_temp[1])
                else:
                    order_spectrum_calc_info[i]['normFactor'].append(
                        1.414 / order_spectrum_calc_info[i]['nfft'])
        order_spectrum_calc_info[i]['order'] = (
            (np.fft.rfftfreq(order_spectrum_calc_info[i]['nfft']) *
             order_spectrum_calc_info[i]['orderResolution'] *
             order_spectrum_calc_info[i]['nfft']))
        if 'maxOrder' in order_spectrum_calc_info[i]:
            order_spectrum_calc_info[i]['order'] = (order_spectrum_calc_info[i]['order'][:(
                    order_spectrum_calc_info[i]['revNum'] * order_spectrum_calc_info[i]['maxOrder'])])
        if speed_calc_info[i]['speedRatio'] != 1:
            # 提前计算需要更换的阶次轴（若转速比不为1才需要进行转换）
            order_spectrum_calc_info[i]['convertOrder'] = (
                    order_spectrum_calc_info[i]['order'] / speed_calc_info[i]['speedRatio']).tolist()
        order_spectrum_calc_info[i]['ppr'] = speed_calc_info[i]['ppr']
        order_spectrum_calc_info[i]['refValue'] = task_info['refValue']
        # 计算出每个测试端多少个点需要设置为参考值（去除低阶成分）
        order_spectrum_calc_info[i]['cutOffNum'] = [
            round(cut_off_freq * 60 / x / order_spectrum_calc_info[i]['orderResolution']) for x in
            speed_recog_info['minSpeed']]

        order_spectrum_calc_info[i]['xName'] = xName
        order_spectrum_calc_info[i]['xUnit'] = xUnit
        arPoints = round(task_info["sampsPerChan"] / 200)
        order_spectrum_calc_info[i]["arPoints"] = arPoints if arPoints > 10 else 10
    return order_spectrum_calc_info


def order_cut_calc_info_update(order_cut_calc_info, order_spectrum_calc_info):
    """
    功能：提前计算二维阶次切片所需要的参数，主要是限制目标阶次的边界，包括：
    1. 最小阶次（orderMin）：提取阶次切片时的目标阶次应大于该最小阶次，与阶次切片的宽度有关（左右的点数）
    2. 最大阶次（orderMax）：提取阶次切片时的目标阶次应小于该最大阶次，与阶次切片的宽度有关（左右的点数）
    返回：更新后的二维阶次切片计算参数
    """
    # update the order boundary for target order confirm
    for i in range(testSectionNum := len(order_spectrum_calc_info)):
        min_order_available = order_spectrum_calc_info[i]['orderResolution'] * (order_cut_calc_info[i]['pointNum'] // 2)
        max_order_available = order_spectrum_calc_info[i]['maxOrder'] - order_spectrum_calc_info[i][
            'orderResolution'] * (order_cut_calc_info[i]['pointNum'] // 2 + 1)
        # 找各测试段最大和最小的阶次，pluck为提取列表中每个字典的key为'value'的值，concat则将二维列表flatten成一维
        min_order = min(concat(pluck('value', order_cut_calc_info[i]['indicatorInfoList'])))
        max_order = max(concat(pluck('value', order_cut_calc_info[i]['indicatorInfoList'])))
        # 校验关注阶次
        if min_order < min_order_available:
            raise ValueError(
                f'test section {i}:min order of 2D order slice: {min_order} set is out of range, should bigger than: {min_order_available}')
        if max_order > max_order_available:
            raise ValueError(
                f'test section {i}:max order of 2D order slice: {min_order} set is out of range, should smaller than: {min_order_available}')
        order_cut_calc_info[i]['xName'] = xName
        order_cut_calc_info[i]['xUnit'] = xUnit
    return order_cut_calc_info


def oned_os_calc_info_update(oned_os_calc_info, order_spectrum_calc_info):
    """
    功能：提前计算一维阶次切片指标所需要的参数，主要是限制目标阶次的边界，包括：
    1. 最小阶次（orderMin）：提取阶次切片时的目标阶次应大于该最小阶次，与阶次切片的宽度有关（左右的点数）
    2. 最大阶次（orderMax）：提取阶次切片时的目标阶次应小于该最大阶次，与阶次切片的宽度有关（左右的点数）
    返回：更新后的一维阶次切片计算参数
    """
    for i in range(testSectionNum := len(order_spectrum_calc_info)):
        min_order_available = order_spectrum_calc_info[i]['orderResolution'] * (oned_os_calc_info[i]['pointNum'] // 2)
        max_order_available = order_spectrum_calc_info[i]['maxOrder'] - (
                order_spectrum_calc_info[i]['orderResolution'] * (oned_os_calc_info[i]['pointNum'] // 2 + 1))
        min_order = min(concat(pluck('value', oned_os_calc_info[i]['indicatorInfoList'])))
        max_order = max(concat(pluck('value', oned_os_calc_info[i]['indicatorInfoList'])))
        # 校验关注阶次
        if min_order < min_order_available:
            raise ValueError(
                f'test section {i}:min order of 1D order indicator: {min_order} set is out of range, should bigger than: {min_order_available}')
        if max_order > max_order_available:
            raise ValueError(
                f'test section {i}:max order of 1D order indicator: {max_order} set is out of range, should bigger than: {max_order_available}')
    return oned_os_calc_info


def cepstrum_calc_info_update(order_spectrum_calc_info):
    """
    功能：生成倒阶次谱所需要的计算参数，包括：
    1. 圈数（revNum），主要用于形成倒阶次谱的x轴信息，由阶次谱的阶次分辨率得到
    返回：倒阶次谱计算的参数信息
    """
    cepstrum_calc_info = [{'revNum': 1 / order_spectrum_calc_info[i]['orderResolution']} for i in
                          range(len(order_spectrum_calc_info))]
    return cepstrum_calc_info


def time_freq_map_calc_info_update(time_freq_map_calc_info):
    """
    功能：提前计算和确认时频分析所需要的参数，包括：
    1. 每次用于fft计算的时域数据的点数（nperseg），即输入数据的长度
    2. fft变换的输出点数（nfft），一般与nperseg相等
    3. 归一化参数（normFactor），最后的结果为rms值，归一化公式为fft_result/(nfft/2)*0.707，即fft_result*1.414/nfft/2
    4. 每帧的重叠点数（noverlap），一般为输入点数的一半（默认值）
    5. 步进点数（nstep），等于输入点数-重叠点数
    6. 窗函数（win），每帧数据所加的窗
    返回：更新后的时频分析参数
    """
    # update time frequency map calculation info
    if time_freq_map_calc_info['nperseg'] is None:
        time_freq_map_calc_info['nperseg'] = 256
    if time_freq_map_calc_info['nfft'] is None:
        time_freq_map_calc_info['nfft'] = time_freq_map_calc_info['nperseg']
    time_freq_map_calc_info['normFactor'] = 1.414 / time_freq_map_calc_info['nfft']
    if time_freq_map_calc_info['noverlap'] is None:
        time_freq_map_calc_info['noverlap'] = time_freq_map_calc_info['nperseg'] // 2
    time_freq_map_calc_info['nstep'] = time_freq_map_calc_info['nperseg'] - time_freq_map_calc_info['noverlap']
    if time_freq_map_calc_info['window']:
        time_freq_map_calc_info['win'], _ = filter_win(time_freq_map_calc_info['nperseg'],
                                                       time_freq_map_calc_info['window'])
    time_freq_map_calc_info['xName'] = xName
    time_freq_map_calc_info['xUnit'] = xUnit
    return time_freq_map_calc_info


def ssa_calc_info_update(ssa_calc_info, speed_calc_info):
    """
    功能：确认和提前计算SSA分析所需要的的配置参数（按圈计算的指标也会用到），包括：
    1. 生成每根轴对应的名称（gearName），比如输入轴，中间轴，输出轴
    2. 生成每根轴对应的ppr（pprNum），由转速来源轴和齿数比决定
    3. 更新每根轴的齿数比（gearRatio），主要是插入输入轴的比值，例如3个齿数比对应的是4根轴
    返回：更新后的SSA参数
    """
    # 默认只有输入轴，更新每个转轴名称
    ssa_calc_info['gearName'] = list()
    ssa_calc_info['pprNum'] = list()
    ssa_calc_info['factors'] = list()
    # 确认是否输入轴和输出轴为同一个
    if len(ssa_calc_info['gearRatio']) == 1 and ssa_calc_info['gearRatio'][0] == 1:
        ssa_calc_info['gearRatio'] = list()
    ssa_calc_info['gearRatio'].insert(0, 1.0)
    if ssa_calc_info['gearRatio']:
        if ssa_calc_info['onInputShaft'] == 0:
            # shaftIndex主要是用于每根轴的ppr计算（正向还是逆向），转速来源于输出轴则为逆向
            ssa_calc_info['shaftIndex'] = len(ssa_calc_info['gearRatio']) - 1
        else:
            # 转速来源于输入轴则为正向
            ssa_calc_info['shaftIndex'] = 0
        for i in range(len(ssa_calc_info['gearRatio'])):
            # 更新ppr信息
            factor = np.prod(ssa_calc_info['gearRatio'][:i + 1]) / np.prod(
                ssa_calc_info['gearRatio'][:ssa_calc_info['shaftIndex'] + 1])
            ssa_calc_info['pprNum'].append(speed_calc_info['ppr'] * factor)
            ssa_calc_info['factors'].append(factor)
            # 更新转轴名称
            if i == 0:
                ssa_calc_info['gearName'].append('InputShaft')
            elif i + 1 == len(ssa_calc_info['gearRatio']):
                ssa_calc_info['gearName'].append('OutputShaft')
            elif i == 1 and len(ssa_calc_info['gearRatio']) == 3:
                ssa_calc_info['gearName'].append('CounterShaft')
            else:
                ssa_calc_info['gearName'].append('CounterShaft' + str(i))
    ssa_calc_info["xName"] = "angle"
    ssa_calc_info["xUnit"] = "°"
    return ssa_calc_info


def stat_factor_calc_info_update(stat_factor_calc_info, order_spectrum_calc_info, task_info, basic_info):
    """
    功能：统计学指标按圈计算参数（指标名称参考时间域指标），包括：
    1. 计算圈数（revNum），即多少圈计算一次。默认为1
    2. 重叠比例（overlapRatio），默认为0.5
    3. 计算的点数（pointsNum），根据转一圈需要的点数依次对应不同的轴
    4. 步进长度（stepPoints），每次步进的点数，由重叠比例决定，每根轴对应不同的值
    输入：
    1. 时间域指标参数信息
    2. 阶次谱计算参数信息
    3. 数据采集参数信息
    4.基本信息
    返回：按圈计算的参数信息
    """
    # 更新统计学指标按圈计算参数
    for i in range(testSectionNum := len(stat_factor_calc_info)):
        if i != stat_factor_calc_info[i]['testSectionIndex']:  # 如果测试段对不上则跳过
            stat_factor_calc_info[i] = defaultdict(list)
            continue
        stat_factor_calc_info[i] = defaultdict(list, stat_factor_calc_info[i])
        if 'revNum' not in stat_factor_calc_info[i]:
            # 每次计算的圈数未设置则默认为1
            stat_factor_calc_info[i]['revNum'] = 1
        if 'overlapRev' not in stat_factor_calc_info[i]:
            # 重叠比例未设置则默认为0.5
            stat_factor_calc_info[i]['overlapRev'] = 0
        stat_factor_calc_info[i]['overlapRatio'] = stat_factor_calc_info[i]['overlapRev'] / stat_factor_calc_info[i][
            'revNum']
        # 确定重叠比率是否设置合理
        if stat_factor_calc_info[i]['overlapRatio'] >= 1:
            # 若重叠比率超过1则强制归零
            stat_factor_calc_info[i]['overlapRatio'] = 0
        speedSourceOrder = stat_factor_calc_info[i]['speedSourceOrder']
        # 基于转速来源轴和齿轮副信息得到每个轴转过固定圈数所需要的数据点数
        # 基于角度域重采样后的振动信号
        stat_factor_calc_info[i]['sampleRate'] = round(1 / order_spectrum_calc_info[i]['dr_af'])
        temp_num = int(stat_factor_calc_info[i]['revNum'] / order_spectrum_calc_info[i]['dr_af'])
        if stat_factor_calc_info[i]["vibrationIndicatorList"]:
            for j in range(len(stat_factor_calc_info[i]['vibrationIndicatorList'])):
                stat_factor_calc_info[i]['vibrationIndicatorList'][j]['pointsNum'] = round(
                    temp_num * speedSourceOrder / stat_factor_calc_info[i]['vibrationIndicatorList'][j]['value'])
                stat_factor_calc_info[i]['vibrationIndicatorList'][j]['stepPoints'] = round(
                    stat_factor_calc_info[i]['vibrationIndicatorList'][j]['pointsNum'] * (
                            1 - stat_factor_calc_info[i]['overlapRatio']))

                # 其它轴每转过revNum，转速来源轴转了多少圈
                stat_factor_calc_info[i]['vibrationIndicatorList'][j]['revNums'] = speedSourceOrder / \
                                                                                   stat_factor_calc_info[i][
                                                                                       'vibrationIndicatorList'][j][
                                                                                       'value']
                stat_factor_calc_info[i]['vibrationIndicatorList'][j]['stepNums'] = \
                    stat_factor_calc_info[i]['vibrationIndicatorList'][j]['revNums'] * (
                            1 - stat_factor_calc_info[i]['overlapRatio'])
                if stat_factor_calc_info[i]['vibrationIndicatorList'][j]['index'].lower()[:3] == 'rms':
                    if basic_info['dBFlag']:
                        stat_factor_calc_info[i]['vibrationIndicatorList'][j]['unit'] = ['dB']
                    else:
                        stat_factor_calc_info[i]['vibrationIndicatorList'][j]['unit'] = [task_info['units'][k] for k in
                                                                                         task_info[
                                                                                             'indicatorsUnitChanIndex']]
                elif stat_factor_calc_info[i]['vibrationIndicatorList'][j]['index'].lower()[:6] == 'spl(a)':
                    stat_factor_calc_info[i]['vibrationIndicatorList'][j]['unit'] = ['dB(A)']
                elif stat_factor_calc_info[i]['vibrationIndicatorList'][j]['index'].lower()[:3] == 'spl':
                    stat_factor_calc_info[i]['vibrationIndicatorList'][j]['unit'] = ['dB']
                else:
                    stat_factor_calc_info[i]['vibrationIndicatorList'][j]['unit'] = ['']

        if stat_factor_calc_info[i]["soundIndicatorList"]:
            for j in range(len(stat_factor_calc_info[i]['soundIndicatorList'])):
                stat_factor_calc_info[i]['soundIndicatorList'][j]['pointsNum'] = round(
                    temp_num * stat_factor_calc_info[i]['soundIndicatorList'][j]['value'] / speedSourceOrder)
                stat_factor_calc_info[i]['soundIndicatorList'][j]['stepPoints'] = round(
                    stat_factor_calc_info[i]['soundIndicatorList'][j]['pointsNum'] * (
                            1 - stat_factor_calc_info[i]['overlapRatio']))

                # 其它轴每转过revNum，转速来源轴转了多少圈
                stat_factor_calc_info[i]['soundIndicatorList'][j]['revNums'] = speedSourceOrder / \
                                                                               stat_factor_calc_info[i][
                                                                                   'soundIndicatorList'][j][
                                                                                   'value']
                stat_factor_calc_info[i]['soundIndicatorList'][j]['stepNums'] = \
                    stat_factor_calc_info[i]['soundIndicatorList'][j]['revNums'] * (
                            1 - stat_factor_calc_info[i]['overlapRatio'])
                if stat_factor_calc_info[i]['soundIndicatorList'][j]['index'].lower()[:3] == 'RMS':
                    if basic_info['dBFlag']:
                        stat_factor_calc_info[i]['soundIndicatorList'][j]['unit'] = ['dB']
                    else:
                        stat_factor_calc_info[i]['soundIndicatorList'][j]['unit'] = [task_info['units'][k] for k in
                                                                                     task_info[
                                                                                         'indicatorsUnitChanIndex']]
                elif stat_factor_calc_info[i]['soundIndicatorList'][j]['index'].lower()[:6] == 'spl(a)':
                    stat_factor_calc_info[i]['soundIndicatorList'][j]['unit'] = ['dB(A)']
                elif stat_factor_calc_info[i]['soundIndicatorList'][j]['index'].lower()[:3] == 'spl':
                    stat_factor_calc_info[i]['soundIndicatorList'][j]['unit'] = ['dB']
                else:
                    stat_factor_calc_info[i]['soundIndicatorList'][j]['unit'] = ['']
        stat_factor_calc_info[i]['refValue'] = task_info['refValue']
        stat_factor_calc_info[i]['xName'] = xName
        stat_factor_calc_info[i]['xUnit'] = xUnit
    return stat_factor_calc_info


def simu_info_update(params):
    """
    功能：更新或设置simu的信息，确保有入口可以更改simu信息，包括：
    1. simu数据所在的目录
    2. 数据组名称
    3. 通道名称
    4. 每次读取的点数
    """
    # 确认是否存在simu信息，存在就直接读取
    if "simuInfo" in params.keys():
        simuinfo = params["simuInfo"]
    else:
        # 不存在就创建默认信息, 只获取文件位置，其他信息从task info中获取
        simuinfo = {"fileFolder": "D:/qdaq/Simu"}
    return simuinfo


def sensor_confirm(task_info, basic_info):
    """
    功能：确认传感器信息设置是否正确（如果设置的传感器数量和采集的不一致则报错）
    输入：
    1. 数据采集参数信息， 里面包含每个通道的名称
    2. 基础信息，包含传感器id信息
    返回：记录信息错误并报错
    """
    if len(task_info["indicatorsUnitChanIndex"]) != len(basic_info["sensorId"]):
        raise ValueError("sensor id and data acquisition info not matched")
    if sensor_count != len(basic_info["sensorId"]):
        raise ValueError("sensor count in config and sensor id in param info not matched")


def speed_signal_confirm(task_info):
    """
    功能：确认软件设定的转速信号类型是否与数据采集信息匹配
    输入：数据采集参数信息， 里面包含每个通道的名称
    返回：记录错误信息并报错
    """
    if speed_signal == 'ttl':
        if not ('Speed' in task_info['channelNames']):
            raise ValueError(
                "speed signal defined: {}, should have Speed channel in taskInfo".format(speed_signal))
    elif speed_signal == 'resolver2':
        if not ('Sin' in task_info['channelNames'] and 'Cos' in task_info['channelNames']):
            raise ValueError(
                "speed signal defined: {}, should have Sin and Cos channel in taskInfo".format(
                    speed_signal))
    elif speed_signal == 'resolver':
        if not ('Sin' in task_info['channelNames']):
            raise ValueError(
                "speed signal defined: {}, should have Sin channel in taskInfo".format(
                    speed_signal))


class Parameters:
    """
    定义参数的类，包含所需要的的配置参数信息，比如数据采集，阶次谱计算等
    """

    # define class to read in tha parameters by type
    def __init__(self, filename):
        # read the json file of parameters
        if encrypt_flag:
            data = decrypt_data(Cryptor, filename)
        else:
            data = read_json(filename)
        # take out the individual parameters module
        self.basicInfo = basic_info_update(data['basicInfo'])
        self.folderInfo = folder_info_update()
        # convert into readable value for NI task
        self.taskInfo = task_info_update(task_info=data['taskInfo'])
        if self.taskInfo["speedFlag"]:
            # 确认转速信号信息
            speed_signal_confirm(self.taskInfo)
        elif version in [1, 2]:
            # 如果没有转速信号则认为只进行信号采集
            self.dataSaveFlag = {"rawData": 1}
        if self.taskInfo["speedFlag"] or (version in [3, 4, 5]):
            # 确认传感器信息
            sensor_confirm(self.taskInfo, self.basicInfo)
            # 确认转速信号信息
            self.speedCalcInfo = speed_calc_info_update(speed_calc_info=data["speedCalcInfo"])
            self.speedRecogInfo = speed_recog_info_update(speed_recog_info=data["speedRecogInfo"])
            self.timeDomainCalcInfo = time_domain_calc_info_update(data["timeDomainCalcInfo"], self.taskInfo,
                                                                   self.basicInfo)
            self.orderSpectrumCalcInfo = order_spectrum_calc_info_update(data["orderSpectrumCalcInfo"],
                                                                         self.speedCalcInfo, self.speedRecogInfo,
                                                                         self.taskInfo)
            self.orderCutCalcInfo = order_cut_calc_info_update(data["orderCutCalcInfo"], data["orderSpectrumCalcInfo"])
            self.onedOSCalcInfo = oned_os_calc_info_update(data["onedOSCalcInfo"], data["orderSpectrumCalcInfo"])
            self.modulationDepthCalcInfo = data["modulationDepthCalcInfo"]
            self.cepstrumCalcInfo = cepstrum_calc_info_update(data['orderSpectrumCalcInfo'])
            self.limitCompareFlag = data["limitCompareFlag"]
            self.dataSaveFlag = data["dataSaveFlag"]
            self.sendResultInfo = "http://" + server_ip + ":8081/api/storage/resultData"
            # read simu info for data play back
            self.simuInfo = simu_info_update(data)
            self.filterInfo = data["filterInfo"]
            self.timeFreqMapCalcInfo = time_freq_map_calc_info_update(data["timeFreqMapCalcInfo"])
            # self.ssaCalcInfo = ssa_calc_info_update(data["ssaCalcInfo"], self.speedCalcInfo) #todo ssa后期再算
            self.statFactorCalcInfo = stat_factor_calc_info_update(data['indicatorByRevCalcInfo'],
                                                                   self.orderSpectrumCalcInfo, self.taskInfo,
                                                                   self.basicInfo)
            # 更新原始数据上传的信息
            self.ftpUploadInfo = dict()
            self.ftpUploadInfo['dataStatus'] = "http://" + server_ip + ":8081/api/storage/originData/status"
            self.ftpUploadInfo['dataAnalysis'] = "http://" + server_ip + ":8081/api/storage/originData/save"
            # 系统信息暂时不用获取
            self.ftpUploadInfo['systemGet'] = "http://" + server_ip + ":8081/api/storage/system/getSystemName"

    def __del__(self):
        # 执行完后自动清除
        print('__del__')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(pathname)s %(funcName)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='[%Y-%m-%d %H:%M:%S]',
        filename='D:/qdaq/Log/temp.log',
        filemode='a'
    )
    try:
        import time
        import os
        from common_info import config_folder, config_file

        t1 = time.perf_counter()
        type_info = "ktz999x_cj"
        config_filename = os.path.join(config_folder,
                                       "_".join([type_info, config_file]))
        print(config_filename)
        if os.path.exists(config_filename):
            param = Parameters(config_filename)
            print("basic info:", param.basicInfo)
            print("folder info:", param.folderInfo)
            print("task info:", param.taskInfo)
            print("speed calc info:", param.speedCalcInfo)
            print("speed recog info:", param.speedRecogInfo)
            print("time domain calc info:", param.timeDomainCalcInfo)
            print("order spectrum calc info:", param.orderSpectrumCalcInfo)
            for i in range(param.speedRecogInfo["test_count_except_dummy"]):
                if param.speedCalcInfo[i]['speedRatio'] != 1:
                    print(
                        f"test section {i}:raw max order:{param.orderSpectrumCalcInfo[i]['order'][-1]}, converted max order:{param.orderSpectrumCalcInfo[i]['convertOrder'][-1]}")
            print("order cutting calc info:", param.orderCutCalcInfo)
            print("1D order indicators calc info:", param.onedOSCalcInfo)
            print("cepstrum calc info:", param.cepstrumCalcInfo)
            print("limit compare info:", param.limitCompareFlag)
            print("data save info:", param.dataSaveFlag)
            print("result send info:", param.sendResultInfo)
            print("simu info:", param.simuInfo)
            print("filter info:", param.filterInfo)
            print("time freq map info:", param.timeFreqMapCalcInfo)
            # print("ssa calc info:", param.ssaCalcInfo)
            print("stat factor calc info:", param.statFactorCalcInfo)
            print("time domain calc info confirm:", param.timeDomainCalcInfo)
        print('cost time: ', time.perf_counter() - t1)
    except Exception:
        traceback.print_exc()
        logging.warning("exec failed, failed msg:" + traceback.format_exc())
        sys.exit()
