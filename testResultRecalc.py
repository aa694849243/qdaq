# -*- coding: utf-8 -*-
"""
create at Thu Aug 19 14:55:55 2021 by Wall

@author: Sergei@Synovate

function: to calculate the test result for the target part of data

update time: 2021-09-24
update content: 1. 截时重计算初始版本（当前版本适用于

return:
1. print out test result for SigMA
"""
import os
import numpy as np
import logging
import traceback
import sys
import getopt
import json
import h5py
import base64
import pickle
from scipy import signal, stats
from scipy.signal import lfilter, butter, filtfilt
from scipy.signal.filter_design import bilinear
from scipy.interpolate import interp1d
from numpy import pi, convolve
import platform
from datetime import datetime

# 读取基本信息
try:
    platform_info = platform.platform()
    if platform_info.lower().startswith('windows'):
        # windows系统性下的日志文件保存路径
        log_folder = 'D:/shengteng-platform/shengteng-platform-storage/pythonScript/Log'
    else:
        # linux的日志文件保存路径
        log_folder = '/shengteng-platform/shengteng-platform-storage/pythonScript/Log'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
except Exception:
    print("script setting error")
    sys.exit()

# 定义日志格式和日志文件名称
logging.basicConfig(
    level=logging.INFO,  # 日志级别，只有日志级别大于等于设置级别的日志才会输出
    format='%(asctime)s %(filename)s %(funcName)s[line:%(lineno)d] %(levelname)s %(message)s',  # 日志输出格式
    datefmt='[%Y-%m-%d %H:%M:%S]',  # 日期表示格式
    filename=os.path.join(log_folder, 'testResultRecalc.log'),  # 输出定向的日志文件路径
    filemode='a'  # 日志写模式，是否尾部添加还是覆盖
)

# 基本信息，包括dB转换信息(包括灵敏度，参考值等信息）等
# 振动信号参考信息
ref_g = 1e-6 / 9.8  # accel unit: m/s^2
# 声压信号参考值
ref_sound_pressure = 20e-6  # unit: Pa
version = 1  # 目前只适用基础版本的qDAQ
xName = "Speed"
xUnit = "rpm"

# 定义配置参数格式(包含文件路径和文件名信息）
config = {"filepath": "",
          "filename": ""}

# 获取外部的输入（调用脚本时的输入信息）
try:
    opts, args = getopt.getopt(sys.argv[1:], '-i:-f:-h-v',
                               ['filepath=', 'filename=', 'help', 'version'])
    for option, value in opts:
        if option in ["-h", "--help"]:
            print("usage: testResultRecalc -i filepath -f filename")
            sys.exit()
        elif option in ['-i']:
            config["filepath"] = value
        elif option in ['-f']:
            config["filename"] = value
        elif option in ["-v", "--version"]:
            print("version: v1.0.0")
            sys.exit()

except Exception:
    print('input command error')
    sys.exit()


# 窗函数
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


# 读取参数配置

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
    功能：更新单位索引信息
    返回：更新后的NI DAQmx数据采集任务配置信息
    """
    # 振动或声音信号的参考值（dB转换）
    task_info['refValue'] = list()
    # 获取用于NVH分析的通道并记录通道名
    task_info['targetChan'] = list()
    # 获取振动或声音通道的原始单位
    task_info['sensorUnit'] = list()
    for channel_index, channel_name in enumerate(task_info['channelNames']):
        if channel_name.startswith('Vib'):
            task_info['refValue'].append(ref_g)
            task_info['targetChan'].append(channel_name)
            task_info['sensorUnit'].append(task_info['units'][channel_index])
        elif channel_name.startswith('Mic'):
            task_info['refValue'].append(ref_sound_pressure)
            task_info['targetChan'].append(channel_name)
            task_info['sensorUnit'].append(task_info['units'][channel_index])
    return task_info


def speed_calc_info_update(speed_calc_info):
    """
    功能：确认转速比信息
    返回：更新后的转速计算信息
    """
    # 确认转速比
    if 'speedRatio' not in speed_calc_info.keys():
        # 若没有设置转速比则默认是1.0
        speed_calc_info['speedRatio'] = 1.0
    if speed_calc_info['speedRatio'] <= 0:
        # 若转速比异常（小于等于0）则认为设置出错，强制设置为1
        speed_calc_info['speedRatio'] = 1.0
    return speed_calc_info


def speed_recog_info_update(speed_recog_info):
    """
    功能：获取最低转速
    输入：
    1. 工况识别参数
    返回：
    1. 更新后的工况识别参数
    """
    min_speed = list()
    # 定义转速范围（用于区分恒速和变速段）
    if 'speedRange' not in speed_recog_info.keys():
        speed_recog_info['speedRange'] = 100
    speed_recog_info['speedPattern'] = list()
    for i in range(len(speed_recog_info['startSpeed'])):
        if not speed_recog_info["testName"][i].lower().startswith("dummy"):
            # dummy段不参与计算
            min_speed.append(
                min(speed_recog_info['startSpeed'][i], speed_recog_info['endSpeed'][i]))
        if abs(speed_recog_info['endSpeed'][i] - speed_recog_info['startSpeed'][i]) <= speed_recog_info['speedRange']:
            # 恒速段
            speed_recog_info['speedPattern'].append(1)
        else:
            # 变速段
            if speed_recog_info['endSpeed'][i] - speed_recog_info['startSpeed'][i] > \
                    speed_recog_info['speedRange']:
                # 升速段
                speed_recog_info['speedPattern'].append(2)
            else:
                # 降速段
                speed_recog_info['speedPattern'].append(3)
    speed_recog_info['overallMinSpeed'] = min(min_speed)
    return speed_recog_info


def time_domain_calc_info_update(time_domian_calc_info, task_info, basic_info):
    """
    功能：更新RMS，Crest，Kurtosis和Skewness，以及SPL或SPLA的单位，具体信息如下：
    1. RMS：单位来源于指定的采集通道
    2. Crest，Kurtosis，Skewness：无单位
    3. SPL：声压级，单位为dB
    4. SPL(A)：A计权声压级，单位为dB(A)
    返回：更新后的时间域指标计算参数
    """
    time_domian_calc_info['indicatorUnit'] = list()
    time_domian_calc_info['refValue'] = task_info['refValue']

    if "Speed" in time_domian_calc_info['indicatorList']:
        time_domian_calc_info['indicatorList'].remove("Speed")

    for unit in task_info["sensorUnit"]:
        temp_unit_list = list()
        for indicator in time_domian_calc_info['indicatorList']:
            if indicator == 'RMS':
                if basic_info['dBFlag']:
                    temp_unit_list.append('dB')
                else:
                    temp_unit_list.append(unit)
            elif indicator == 'SPL(A)':
                temp_unit_list.append('dB(A)')
            elif indicator == 'SPL':
                temp_unit_list.append('dB')
            else:
                temp_unit_list.append('')
        time_domian_calc_info['indicatorUnit'].append(temp_unit_list)
    time_domian_calc_info['xName'] = xName
    time_domian_calc_info['xUnit'] = xUnit
    time_domian_calc_info['calSize'] = int(
        task_info["sampleRate"] / time_domian_calc_info["calRate"])
    return time_domian_calc_info


def order_spectrum_calc_info_update(order_spectrum_calc_info, speed_calc_info, min_speed,
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
                min_speed * 2 * 1.6384)
    else:
        max_order_available = (60 * task_info['sampleRate']) / (
                min_speed * 2)
    if order_spectrum_calc_info['maxOrder'] > max_order_available:
        raise ValueError(
            "max order: {} set error, should less than {}".format(
                order_spectrum_calc_info['maxOrder'],
                max_order_available))
    if order_spectrum_calc_info['overlapRatio'] >= 1:
        order_spectrum_calc_info['overlapRatio'] = 0
    # 定义角度域降采样之前的采样点角度间隔
    order_spectrum_calc_info['dr_bf'] = min_speed / 60 / task_info['sampleRate']
    # 定义角度域降采样之后的采样点角度间隔
    # order_spectrum_calc_info['dr_af'] = 1 / order_spectrum_calc_info[
    #     'maxOrder'] / 2 / 1.6384  # 1.6384=1.28*1.28
    order_spectrum_calc_info['dr_af'] = 1 / order_spectrum_calc_info[
        'maxOrder'] / 2 / 1.28  # 1.6384=1.28*1.28
    order_spectrum_calc_info['nfft'] = int(
        order_spectrum_calc_info['revNum'] / order_spectrum_calc_info['dr_af'])
    order_spectrum_calc_info['nstep'] = int(order_spectrum_calc_info['nfft'] *
                                            (1 - order_spectrum_calc_info[
                                                'overlapRatio']))
    # 构建窗函数
    if order_spectrum_calc_info['window']:
        order_spectrum_calc_info['win'] = list()
        order_spectrum_calc_info['normFactor'] = list()  # 1.414 for rms normalization
        for i, winType in enumerate(order_spectrum_calc_info['window']):
            win_temp = filter_win(order_spectrum_calc_info['nfft'], winType)
            order_spectrum_calc_info['win'].append(win_temp[0])
            # 是否开启修正系数 0是不开启 赋值会被压下来
            if order_spectrum_calc_info['winCorrectFlag'][i]:
                order_spectrum_calc_info['normFactor'].append(
                    1.414 / order_spectrum_calc_info['nfft'] * win_temp[1])
            else:
                order_spectrum_calc_info['normFactor'].append(
                    1.414 / order_spectrum_calc_info['nfft'])
    order_spectrum_calc_info['order'] = (
        (np.fft.rfftfreq(order_spectrum_calc_info['nfft']) *
         order_spectrum_calc_info['orderResolution'] *
         order_spectrum_calc_info['nfft']))
    if order_spectrum_calc_info['maxOrder']:
        order_spectrum_calc_info['order'] = (order_spectrum_calc_info['order'][:(
                order_spectrum_calc_info['revNum'] * order_spectrum_calc_info[
            'maxOrder'])])
    if speed_calc_info['speedRatio'] != 1:
        # 提前计算需要更换的阶次轴（若转速比不为1才需要进行转换）
        order_spectrum_calc_info['convertOrder'] = (
                order_spectrum_calc_info['order'] / speed_calc_info[
            'speedRatio']).tolist()
    order_spectrum_calc_info['ppr'] = speed_calc_info['ppr']
    order_spectrum_calc_info['refValue'] = task_info['refValue']
    order_spectrum_calc_info['xName'] = xName
    order_spectrum_calc_info['xUnit'] = xUnit
    arPoints = round(task_info["sampsPerChan"] / 200)
    order_spectrum_calc_info["arPoints"] = arPoints if arPoints > 10 else 10
    return order_spectrum_calc_info


def order_cut_calc_info_update(order_cut_calc_info, order_spectrum_calc_info):
    """
    功能：提前计算二维阶次切片所需要的参数，主要是限制目标阶次的边界，包括：
    1. 最小阶次（orderMin）：提取阶次切片时的目标阶次应大于该最小阶次，与阶次切片的宽度有关（左右的点数）
    2. 最大阶次（orderMax）：提取阶次切片时的目标阶次应小于该最大阶次，与阶次切片的宽度有关（左右的点数）
    返回：更新后的二维阶次切片计算参数
    """
    # update the order boundary for target order confirm
    min_order_available = order_spectrum_calc_info['orderResolution'] * (
            order_cut_calc_info['pointNum'] // 2)
    max_order_available = \
        order_spectrum_calc_info['maxOrder'] - order_spectrum_calc_info['orderResolution'] * \
        (order_cut_calc_info['pointNum'] // 2 + 1)
    min_order = min(map(min, order_cut_calc_info['orderList']))
    max_order = max(map(max, order_cut_calc_info['orderList']))
    # 校验关注阶次
    if min_order < min_order_available:
        raise ValueError(
            'min order of 2D order slice: {} set is out of range, should bigger than: {}'.format(
                min_order,
                min_order_available))
    if max_order > max_order_available:
        raise ValueError(
            'max order of 2D order slice: {} set is out of range, should smaller than: {}'.format(
                max_order,
                max_order_available))
    order_cut_calc_info['xName'] = xName
    order_cut_calc_info['xUnit'] = xUnit
    return order_cut_calc_info


def oned_os_calc_info_update(oned_os_calc_info, order_spectrum_calc_info):
    """
    功能：提前计算一维阶次切片指标所需要的参数，主要是限制目标阶次的边界，包括：
    1. 最小阶次（orderMin）：提取阶次切片时的目标阶次应大于该最小阶次，与阶次切片的宽度有关（左右的点数）
    2. 最大阶次（orderMax）：提取阶次切片时的目标阶次应小于该最大阶次，与阶次切片的宽度有关（左右的点数）
    返回：更新后的一维阶次切片计算参数
    """
    min_order_available = order_spectrum_calc_info['orderResolution'] * (
            oned_os_calc_info['pointNum'] // 2)
    max_order_available = order_spectrum_calc_info['maxOrder'] - (
            order_spectrum_calc_info['orderResolution'] * (oned_os_calc_info['pointNum'] // 2 + 1))
    min_order = min(map(min, oned_os_calc_info['orderList']))
    max_order = max(map(max, oned_os_calc_info['orderList']))
    # 校验关注阶次
    if min_order < min_order_available:
        raise ValueError(
            'min order of 1D order indicator: {} set is out of range, should bigger than: {}'.format(
                min_order,
                min_order_available))
    if max_order > max_order_available:
        raise ValueError(
            'max order of 1D order indicator: {} set is out of range, should smaller than: {}'.format(
                max_order,
                max_order_available))
    return oned_os_calc_info


def cepstrum_calc_info_update(order_spectrum_calc_info):
    """
    功能：生成倒阶次谱所需要的计算参数，包括：
    1. 圈数（revNum），主要用于形成倒阶次谱的x轴信息，由阶次谱的阶次分辨率得到
    返回：倒阶次谱计算的参数信息
    """
    cepstrum_calc_info = dict()
    cepstrum_calc_info['revNum'] = 1 / order_spectrum_calc_info['orderResolution']
    return cepstrum_calc_info


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


def stat_factor_calc_info_update(stat_factor_calc_info, ssa_calc_info,
                                 order_spectrum_calc_info, task_info, basic_info):
    """
    功能：统计学指标按圈计算参数（指标名称参考时间域指标），包括：
    1. 计算圈数（revNum），即多少圈计算一次。默认为1
    2. 重叠比例（overlapRatio），默认为0.5
    3. 计算的点数（pointsNum），根据转一圈需要的点数依次对应不同的轴
    4. 步进长度（stepPoints），每次步进的点数，由重叠比例决定，每根轴对应不同的值
    输入：
    1. 时间域指标参数信息
    2. ssa分析参数信息
    3. 阶次谱计算参数信息
    4. 数据采集参数信息
    返回：按圈计算的参数信息
    """
    # 更新统计学指标按圈计算参数
    if 'revNum' not in stat_factor_calc_info.keys():
        # 每次计算的圈数未设置则默认为1
        stat_factor_calc_info['revNum'] = 1
    if 'overlapRev' not in stat_factor_calc_info.keys():
        # 重叠比例未设置则默认为0.5
        stat_factor_calc_info['overlapRev'] = 0
    stat_factor_calc_info['overlapRatio'] = stat_factor_calc_info['overlapRev'] / \
                                            stat_factor_calc_info['revNum']
    # 确定重叠比率是否设置合理
    if stat_factor_calc_info['overlapRatio'] >= 1:
        # 若重叠比率超过1则强制归零
        stat_factor_calc_info['overlapRatio'] = 0

    # 计算的圈数定义在stat_factor_calc_info的revNum中，转速来源轴定义在ssa_calc_info的onInputShaft中
    # 基于转速来源轴和齿轮副信息得到每个轴转过固定圈数所需要的数据点数
    stat_factor_calc_info['pointsNum'] = list()
    stat_factor_calc_info['stepPoints'] = list()
    # 基于角度域重采样后的振动信号
    stat_factor_calc_info['sampleRate'] = round(1 / order_spectrum_calc_info['dr_af'])
    temp_num = int(stat_factor_calc_info['revNum'] / order_spectrum_calc_info['dr_af'])
    # print(temp_num)
    if ssa_calc_info['gearRatio']:
        stat_factor_calc_info['revNums'] = list()
        stat_factor_calc_info['stepNums'] = list()
        for i in range(len(ssa_calc_info['gearRatio'])):
            stat_factor_calc_info['pointsNum'].append(
                int(round(temp_num * np.prod(ssa_calc_info['gearRatio'][:i + 1]) / np.prod(
                    ssa_calc_info['gearRatio'][:ssa_calc_info['shaftIndex'] + 1]))))
            stat_factor_calc_info['stepPoints'].append(
                int(stat_factor_calc_info['pointsNum'][i] * (
                        1 - stat_factor_calc_info['overlapRatio'])))

            # 其它轴每转过revNum，转速来源轴转了多少圈
            stat_factor_calc_info['revNums'].append(
                stat_factor_calc_info['revNum'] * np.prod(
                    ssa_calc_info['gearRatio'][:i + 1]) / np.prod(
                    ssa_calc_info['gearRatio'][:ssa_calc_info['shaftIndex'] + 1]))
            stat_factor_calc_info['stepNums'].append(stat_factor_calc_info['revNums'][i] * (
                    1 - stat_factor_calc_info['overlapRatio']))
    stat_factor_calc_info['indicatorUnit'] = list()
    stat_factor_calc_info['refValue'] = task_info['refValue']

    if "Speed" in stat_factor_calc_info["indicatorList"]:
        stat_factor_calc_info["indicatorList"].remove("Speed")

    for unit in task_info["sensorUnit"]:
        temp_unit_list = list()
        for indicator in stat_factor_calc_info['indicatorList']:
            if indicator == 'RMS':
                if basic_info['dBFlag']:
                    temp_unit_list.append('dB')
                else:
                    temp_unit_list.append(unit)
            elif indicator == 'SPL(A)':
                temp_unit_list.append('dB(A)')
            elif indicator == 'SPL':
                temp_unit_list.append('dB')
            else:
                temp_unit_list.append('')
        stat_factor_calc_info['indicatorUnit'].append(temp_unit_list)
    stat_factor_calc_info['gearName'] = ssa_calc_info['gearName']
    stat_factor_calc_info['indicatorNum'] = len(stat_factor_calc_info['indicatorList'])
    stat_factor_calc_info['xName'] = xName
    stat_factor_calc_info['xUnit'] = xUnit
    return stat_factor_calc_info


def sensor_confirm(task_info, basic_info):
    """
    功能：确认传感器信息设置是否正确（如果设置的传感器数量和采集的不一致则报错）
    输入：
    1. 数据采集参数信息， 里面包含每个通道的名称
    2. 基础信息，包含传感器id信息
    返回：记录信息错误并报错
    """
    if len(task_info["sensorUnit"]) != len(basic_info["sensorId"]):
        raise ValueError("sensor id and data acquisition info not matched")


def create_empty_final_result(basic_info, recalc_info):
    """
    功能：初始化一份结果数据，用于数据分析完之后进行更新
    输入：
    1. 时间戳信息，即测试开始的时间
    2. 基本信息，包括检测系统名称，产品类型，序列号，传感器等信息
    3. 测试段信息
    返回：一份数值为空的结果数据
    """
    final_result = dict()
    final_result['systemNo'] = basic_info['systemNo']
    final_result['type'] = basic_info['type']
    final_result['serialNo'] = recalc_info['serialNo']
    # put some judgement info into final result
    final_result['artificialJudgment'] = None
    final_result['artificialDefectDescription'] = None
    final_result['judgedBy'] = None
    final_result['intelligenceStatus'] = None
    final_result['intelligenceDefectDescription'] = None
    # 这里的时间为当地时间，并不是UTC时间
    final_result['time'] = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    final_result['overallResult'] = -1
    final_result['qdaqDefectDescription'] = ""
    final_result['resultData'] = list()
    for i, sensorId in enumerate(basic_info['sensorId']):
        final_result['resultData'].append(dict())
        final_result['resultData'][i]['sensorId'] = sensorId
        final_result['resultData'][i]['resultBySensor'] = -1
        # limit version的信息是与传感器一起的
        final_result['resultData'][i]['limitVersion'] = "-1"
        data_section = list()
        # 由于只有一个测试段则不用再遍历创建
        data_section.append(dict())
        j = 0
        data_section[j]['testName'] = recalc_info['testName']
        data_section[j]['testResult'] = -1
        data_section[j]['testSensorState'] = 'ok'
        data_section[j]['startTime'] = recalc_info['startTime']
        data_section[j]['endTime'] = recalc_info['endTime']
        data_section[j]['results'] = dict()
        data_section[j]['results']['onedData'] = -1
        data_section[j]['results']['twodTD'] = -1
        data_section[j]['results']['twodOC'] = -1
        data_section[j]['results']['twodOS'] = -1
        data_section[j]['results']['twodCeps'] = -1
        data_section[j]['onedData'] = None
        data_section[j]['twodTD'] = None
        data_section[j]['twodOC'] = None
        data_section[j]['twodOS'] = None
        data_section[j]['twodCeps'] = None
        final_result['resultData'][i]['dataSection'] = data_section
    return final_result


# dB转换函数
def db_convertion(data, ref_value):
    """
    功能：转换为dB值
    输入：
    1. 结果数据，可以是一维或者二维数据
    2. 参考值，用于db的计算
    返回：转换后的结果
    """
    return 20 * np.log10(np.abs(data) / ref_value)


# 原始数据读取函数
def raw_data_read(h5filename, channel_names):
    # 读取传感器数据（振动或声音）
    sensor_data = dict()
    with h5py.File(h5filename, 'r') as h5pyFile:
        data_group = h5pyFile['AIData']
        for channel_name in channel_names:
            sensor_data[channel_name] = np.array(data_group[channel_name], dtype='float')
    return sensor_data


# 读取转速曲线
def read_speed_curve(filename):
    # 读取转速曲线
    with h5py.File(filename, 'r') as h5pyFile:
        speed_loc = np.array(h5pyFile['speedData']['speedLoc'], dtype='float')
        speed_value = np.array(h5pyFile['speedData']['speedValue'], dtype='float')
    return speed_loc, speed_value


# 读取trigger
def speed_trigger_read(h5filename):
    with h5py.File(h5filename, 'r') as h5pyFile:
        trigger_location = np.array(h5pyFile['triggerData']['Trigger'], dtype='int')
    return trigger_location


def read_json(filename, flag=0):
    """
    功能：读取json格式的数据（里面是json字符串），可指定按那种编码格式服务（需根据写入时的编码格式来定）
    输入：
    1. 文件名（包含完整路径）
    2. 编码格式包括：
        2.1 flag=1：b64decode
        2.2 flag=2：b32decode
        2.3 flag=3：b16decode
        2.4 flag=4：pickle进行反序列化
        2.5 其他，直接读取json字符串
    返回：字典类型的json格式数据
    function: read out the data saved in the target JSON file
    :param
    filename(str): read the target JSON file into the data
    :return
    data(dict): the data read from the JSON file
    """
    if flag == 1:
        # b64解码
        with open(filename, 'rb') as f:
            data = json.loads(base64.b64decode(f.read()).decode('utf-8'))
    elif flag == 2:
        # b32解码
        with open(filename, 'rb') as f:
            data = json.loads(base64.b32decode(f.read()).decode('utf-8'))
    elif flag == 3:
        # b16解码
        with open(filename, 'rb') as f:
            data = json.loads(base64.b16decode(f.read()).decode('utf-8'))
    elif flag == 4:
        # pickle进行反序列化
        data = pickle.loads(open(filename, "rb").read())
    else:
        # 直接读取
        with open(filename, 'r') as f:
            data = json.load(f)
    return data


def create_empty_twodtd(time_domain_calc_info, sensor_index, indicator_diagnostic=-1):
    """
    功能：初始化二维时间域指标，用于实时更新
    输入：
    1. 时间域指标计算配置参数信息
    2. 传感器索引，用于不同传感器信号对应的单位
    3. 初始评判结果，可以指定的值进行初始化
    返回：不包含具体数值的二维时间域结果
    """
    twodtd_result = list()
    for i, indicator in enumerate(time_domain_calc_info['indicatorList']):
        twodtd_result.append({
            'xName': time_domain_calc_info['xName'],
            'xUnit': time_domain_calc_info['xUnit'],
            'xValue': list(),
            'yName': indicator,
            'yUnit': time_domain_calc_info['indicatorUnit'][sensor_index][i],
            'yValue': list(),
            "indicatorDiagnostic": indicator_diagnostic
        })
    return twodtd_result


def create_empty_temptd():
    """
    功能：创建数据结构用于保存时间域计算中的临时数据（最终用于一维时间域指标计算），每计算一段信号就会有一个对应的值
    返回：字典型数据包括：
    1. 平方和值xi2
    2. 最大值xmax
    3. 均值xmean
    4. 三次方和值xi3
    5. 四次方和值xi4
    6. A计权的平方值xi2_A
    """
    temptd_result = dict()
    temptd_result['xi2'] = list()
    temptd_result['xmax'] = list()
    temptd_result['xmean'] = list()
    temptd_result['xi3'] = list()
    temptd_result['xi4'] = list()
    temptd_result['xi2_A'] = list()
    return temptd_result


def butter_filter(signal, wn, fs, order=3, btype='lowpass'):
    """
    功能：利用巴特沃斯滤波器对信号进行滤波
    输入：
    1. 待滤波信号
    2. 滤波截止频率
    3. fs采样率
    4. 滤波器阶次，默认为3
    5. 滤波器类型，包括低通，高通，带通，带阻
    返回：滤波后的信号
    function: signal filter user butter
    :param
    signal(list): input signal(before filtering)
    Wn(list): normalized cutoff frequency （one for high/low pass filter，two for band pass/stop filter)
    sampleRate(int): sample rate of input signal
    order(int)：filter order（generally the order higher， transition band narrower）default as 5
    btyte(str): filter type（low pass：'low'；high pass：'high'；band pass：'bandpass'；band stop：'bandstop'）
    default as low
    analog: digital or analog filter，cutoff of analog filter is angular frequency，for digital is relative
    frequency. default is digital filter
    :return
        signal after filter
    """
    nyq = fs / 2  # Nyquist's Law
    if btype == 'lowpass' or btype == 'highpass':
        cutoff = wn[0] / nyq
        b, a = butter(order, cutoff, btype, analog=False)
    else:
        lowcut = wn[0] / nyq
        highcut = wn[1] / nyq
        b, a = butter(order, [lowcut, highcut], btype, analog=False)
    return filtfilt(b, a, signal)


def A_weighting(Fs):
    """
    定义A计权函数，用于生成A计权滤波器
    输入：采样率
    返回：滤波器系数
    Design of an A-weighting filter.

    B, A = A_weighting(Fs) designs a digital A-weighting filter for
    sampling frequency Fs. Usage: y = lfilter(B, A, x).
    Warning: Fs should normally be higher than 20 kHz. For example,
    Fs = 48000 yields a class 1-compliant filter.

    Originally a MATLAB script. Also included ASPEC, CDSGN, CSPEC.

    Author: Christophe Couvreur, Faculte Polytechnique de Mons (Belgium)
            couvreur@thor.fpms.ac.be
    Last modification: Aug. 20, 1997, 10:00am.

    http://www.mathworks.com/matlabcentral/fileexchange/69
    http://replaygain.hydrogenaudio.org/mfiles/adsgn.m
    Translated from adsgn.m to PyLab 2009-07-14 endolith@gmail.com

    References:
       [1] IEC/CD 1672: Electroacoustics-Sound Level Meters, Nov. 1996.

    Fs： sample rate
    return: filter parameters
    """
    # Definition of analog A-weighting filter according to IEC/CD 1672.
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997
    m = 'full'

    NUMs = [(2 * pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
    DENs = convolve([1, +4 * pi * f4, (2 * pi * f4) ** 2],
                    [1, +4 * pi * f1, (2 * pi * f1) ** 2], mode=m)
    DENs = convolve(convolve(DENs, [1, 2 * pi * f3], mode=m),
                    [1, 2 * pi * f2], mode=m)

    # Use the bilinear transformation to get the digital filter.
    # (Octave, MATLAB, and PyLab disagree about Fs vs 1/Fs)
    return bilinear(NUMs, DENs, Fs)


def rms(data):
    """
    功能：计算rms
    输入：要计算的数据
    返回：rms结果
    function: calculate the RPM of an array/list(vibration data)
    :param
    data(list/array): input data need to do RMS
    :return
    result(float): only one value after RMS of data
    """
    return np.sqrt(sum(np.power(data, 2)) / len(data))


def kurtosis(xi4, xi3, xi2, mean_value, size):
    """
    功能：根据二维kurtosis的中间结果计算一维kurtosis值
    输入：
    1. 四次方和值xi4
    2. 三次方和值xi3
    3. 平方和值xi2
    4. 平均值mean_value
    5. 每段数据的长度size
    返回：峰度值（一个数）
    function: calculate kurtosis value when parameters ready (1D result with temp values)
    :param
    xi4(float): biquadratic value
    xi3(float): cube value
    xi2(float): quadratic value
    mean_value(float): average value
    size(int): length of 1 frame data
    :return:
    result(float): 1D kurtosis value
    """
    return ((xi4 - 4 * mean_value * xi3 + 6 * np.power(mean_value,
                                                       2) * xi2 - 3 * size * np.power(
        mean_value, 4)) * size / np.power(xi2 - size * np.power(mean_value, 2), 2))


def skewness(xi3, xi2, mean_value, size):
    """
    功能：根据二维skewness的中间结果计算一维skewness值
    输入：
    1. 三次方值xi3
    2. 平方值xi2
    3. 平均值mean_value
    4. 每段数据的长度size
    返回：偏度值（一个数）
    function: calculate kurtosis value when parameters ready
    :param
    xi3: cube value
    xi2: quadratic value
    mean_value: average value
    size: length of 1 frame data
    :return:
    result(float): 1D skewness value
    """
    return ((xi3 - 3 * mean_value * xi2 + 2 * size * np.power(mean_value,
                                                              3)) * np.sqrt(
        size)
            / np.power(xi2 - size * np.power(mean_value, 2), 1.5))


def twodtd_rms(data, size):
    """
    功能：计算二维rms结果
    输入：
    1. 待计算rms的数据序列
    2. 数据长度
    返回：除了rms值还包括中间量
    1. 目标段数据的rms值
    2. 平方和值xi2
    function: calculate the RPM of an array/list(vibration data)
    :param
    data(list/array): input data need to do RMS
    size(int): length of data
    :return
    result(float): only one value after RMS of data
    xi2(float): temp sum quadratic value of 1 frame data
    """
    xi2 = np.sum(np.power(data, 2))
    rms_value = np.sqrt(xi2 / size)
    return rms_value, xi2


def twodtd_crest(data, size):
    """
    功能：计算二维峰值因子Crest
    输入：
    1. 待计算crest的数据序列
    2. 数据长度
    返回：除了峰值因子还包括中间量
    1. 目标段数据的crest值
    2. 目标段数据的最大值
    function: calculate the Crest of an array/list(vibration data)
    :param
    data(list/array): input data for crest
    size(int): length of data
    :return
    crest(float): one value for the input data
    max_value(float): temp max value of 1 frame data
    """
    max_value = np.max(data)
    xi2 = np.sum(np.power(data, 2))
    crest = max_value / np.sqrt(xi2 / size)
    return crest, max_value, xi2


def twodtd_kurtosis(data, size):
    """
    功能：计算二维峰度指标kurtosis
    输入：
    1. 待计算kurtosis的数据序列
    2. 数据长度
    返回：除了峰度值还包括中间量
    1. 目标段数据的kurtosis值
    2. 均值mean_value
    3. 四次方和值xi4
    4. 三次方和值xi3
    5. 二次方和值xi2
    function: calculate the kurtosis of an array/list(vibration data)
    :param
    data(list/array): input data for kurtosis
    size(int): length of data
    :return
    kur(float): kurtosis of 1 frame data
    mean_value(float): average of the input data
    xi4(float): temp sum biquadratic value of 1 frame data
    xi3(float): temp sum cube value of 1 frame data
    """
    mean_value = np.mean(data)
    xi4 = np.sum(np.power(data, 4))
    xi3 = np.sum(np.power(data, 3))
    xi2 = np.sum(np.power(data, 2))
    kur = kurtosis(xi4, xi3, xi2, mean_value, size)
    return kur, mean_value, xi4, xi3, xi2


def twodtd_skewness(data, size):
    """
    功能：计算二维偏度指标skewness
    输入：
    1. 待计算skewness的数据序列
    2. 数据长度
    返回：除了偏度值还包括中间量
    1. 目标段数据的skewness值
    2. 均值mean_value
    3. 三次方和值xi3
    4. 二次方和值xi2
    function: calculate the kurtosis of an array/list(vibration data)
    :param
    data(list/array): input data for kurtosis
    size(int): length of data
    :return
    skew(float): skewness value of the input data
    mean_value(float): average of the input data
    xi3(float): temp sum cube value of 1 frame data
    xi2(float): temp sum quadratic value of 1 frame data
    """
    mean_value = np.mean(data)
    xi3 = np.sum(np.power(data, 3))
    xi2 = np.sum(np.power(data, 2))
    skew = skewness(xi3, xi2, mean_value, size)
    return skew, mean_value, xi3, xi2


def twod_spl(data, size):
    """
    功能：计算二维声压级（随时间或转速变化），参考声压级为2*10-5Pa，该算法是基于信号的rms计算声压级的
    输入：
    1. 待计算声压级的数据序列
    2. 数据长度
    返回：包括声压级和中间值
    1. 声压级
    2. 平方和值xi2
    """
    rms_value, xi2 = twodtd_rms(data, size)
    return 20 * np.log10(rms_value / (2 * 10 ** -5)), xi2


def oned_rms(temptd, calc_size):
    """
    功能：基于中间量计算一维rms值
    输入：
    1. 中间变量（这里是平方和值xi2）
    2. 二维rms计算时的数据长度
    返回：一维rms结果
    """
    return np.sqrt(np.sum(temptd['xi2']) / (calc_size * len(temptd['xi2'])))


def oned_crest(temptd, calc_size):
    """
    功能：基于中间量计算一维crest值
    输入：
    1. 中间变量（这里是max值）
    2. 二维crest计算时的数据长度
    返回：一维crest结果
    """
    return np.max(temptd['xmax']) / oned_rms(temptd, calc_size)


def oned_kurtosis(temptd, calc_size):
    """
    功能：基于中间量计算一维kurtosis值
    输入：
    1. 中间变量（这里是xi4，xi3，xi2和xmean）
    2. 二维kurtosis计算时的数据长度
    返回：一维kurtosis结果
    """
    xi4 = np.sum(temptd['xi4'])
    xi3 = np.sum(temptd['xi3'])
    xi2 = np.sum(temptd['xi2'])
    mean_value = np.mean(temptd['xmean'])
    size = len(temptd['xi4']) * calc_size
    return kurtosis(xi4, xi3, xi2, mean_value, size)


def oned_skewness(temptd, calc_size):
    """
    功能：基于中间量计算一维skewness值
    输入：
    1. 中间变量（这里是xi3，xi2和xmean）
    2. 二维skewness计算时的数据长度
    返回：一维skewness结果
    """
    xi3 = np.sum(temptd['xi3'])
    xi2 = np.sum(temptd['xi2'])
    mean_value = np.mean(temptd['xmean'])
    size = len(temptd['xi3']) * calc_size
    return skewness(xi3, xi2, mean_value, size)


def oned_spl(temptd, calc_size):
    """
    功能：基于中间量计算一维SPL值（基于得到的rms计算声压级）
    输入：
    1. 中间变量（这里是平方和值xi2）
    2. 二维SPL计算时的数据长度
    返回：一维SPL结果
    """
    return 20 * np.log10(oned_rms(temptd, calc_size) / (2 * 10 ** -5))


def oned_a_spl(temptd, calc_size):
    """
    功能：基于中间量计算一维A计权SPL值
    输入：
    1. 中间变量（这里是A计权后的平方和值xi2_A）
    2. 二维SPL(A)计算时的数据长度
    返回：一维SPL(A)结果
    """
    if calc_size > 500:
        # 由于滤波导致边缘部分的数据误差较大，故而在计算时需要去掉，这里设置为500，所以要求每次参与计算的数据长度大于500（一般都是至少5000个点）
        return 20 * np.log10(np.sqrt(np.sum(temptd['xi2_A']) / (
                (calc_size - 500) * len(temptd['xi2_A']))) / (2 * 10 ** -5))
    else:
        # 纳入边缘的点计算得到的A计权声压级可能会不太准确
        return 20 * np.log10(np.sqrt(
            np.sum(temptd['xi2_A']) / (calc_size * len(temptd['xi2_A']))) / (
                                     2 * 10 ** -5))


def oned_time_domain(temptd, time_domain_calc_info, sensor_index,
                     db_flag=0):
    """
    功能：计算一维时间域指标（基于二维时间域指标计算的中间量计算得到）
    输入：
    1. 二维时域指标计算的中间值
    2. 二维时域指标计算时每段数据的长度
    3. 时域指标计算参数信息（主要需要指标列表）
    4. 传感器索引，用于获取指定单位信息
    5. 是否转换db，默认否（0表示否，1表示是）
    返回：带有数据的一维时间域指标结果列表
    """
    onedtd_result = list()
    calc_size = time_domain_calc_info['calSize']
    for i, indicator in enumerate(time_domain_calc_info['indicatorList']):
        # 根据指标名称进行计算
        if indicator == 'RMS':
            # 是否换算为dB
            if db_flag:
                temp_oned_rms = db_convertion(oned_rms(temptd, calc_size),
                                              time_domain_calc_info['refValue'][
                                                  sensor_index])
                target_unit = 'dB'
            else:
                temp_oned_rms = oned_rms(temptd, calc_size)
                target_unit = \
                    time_domain_calc_info['indicatorUnit'][sensor_index][i]
            onedtd_result.append({
                'name': 'RMS',
                'unit': target_unit,
                'value': temp_oned_rms,
                "indicatorDiagnostic": -1
            })
        elif indicator == 'Crest':
            onedtd_result.append({
                'name': 'Crest',
                'unit': time_domain_calc_info['indicatorUnit'][sensor_index][i],
                'value': oned_crest(temptd, calc_size),
                "indicatorDiagnostic": -1
            })
        elif indicator == 'Kurtosis':
            onedtd_result.append({
                'name': 'Kurtosis',
                'unit': time_domain_calc_info['indicatorUnit'][sensor_index][i],
                'value': oned_kurtosis(temptd, calc_size),
                "indicatorDiagnostic": -1
            })
        elif indicator == 'Skewness':
            onedtd_result.append({
                'name': 'Skewness',
                'unit': time_domain_calc_info['indicatorUnit'][sensor_index][i],
                'value': oned_skewness(temptd, calc_size),
                "indicatorDiagnostic": -1
            })
        elif indicator == 'SPL':
            onedtd_result.append({
                'name': 'SPL',
                'unit': 'dB',
                'value': oned_spl(temptd, calc_size),
                "indicatorDiagnostic": -1
            })
        elif indicator == 'SPL(A)':
            onedtd_result.append({
                'name': 'SPL(A)',
                'unit': 'dB(A)',
                'value': oned_a_spl(temptd, calc_size),
                "indicatorDiagnostic": -1
            })
        elif indicator == 'speed':
            pass
        else:
            logging.info(
                "error, no this indicator calculation for now, please check the indicator name first")
    return onedtd_result


def twod_time_domain(twodtd, temptd, counter_td, cum_vib, fs, start_time,
                     time_domain_calc_info, sensor_index):
    """
    功能：计算二维时间域指标（通过计算频率，如10Hz表示每0.1秒的原始数据计算一次）
    输入：
    1. 实时更新的二维时间域指标结果集，第一次为初始化的结果集（即空结果集）
    2. 实时更新的二维时间域指标计算中间结果
    3. 计数器，记录进行了多少次二维时间域计算以方便提取目标信号
    4. 累积信号，每次计算只需要提取其中的一部分
    5. 采样率，主要用于转换时间
    6. 该测试段起始时间
    7. 每段信号长度（用于计算的）
    8. 时间域指标计算参数（主要需要指标列表和单位）
    9. 传感器索引，用于获取db转换时的参考值
    返回：
    1. 实时更新的二维时间域指标结果集
    2. 实时更新的二维时间域指标计算中间结果
    3. 计数器
    """
    calc_size = time_domain_calc_info['calSize']
    while calc_size * (counter_td + 1) <= len(cum_vib):
        # cut out the data to calculate the indicators（根据计算频率截取要计算的信号）
        calc_vib = cum_vib[calc_size * counter_td: calc_size * (counter_td + 1)]
        # 生成待计算信号所对应的时间（开始点时间和结束点时间）
        # x_value = [counter_td * calc_size / fs + start_time,
        #            ((counter_td + 1) * calc_size - 1) / fs + start_time]
        # 下面是记录中间点时间的代码
        x_value = (counter_td + 0.5) * calc_size / fs + start_time
        # 初始化要返回的中间量
        xi2 = 0
        xi3 = 0
        xi4 = 0
        max_value = 0
        mean_value = 0
        # 滤波之后的x^2
        xi2_A = 0
        for j, indicator in enumerate(time_domain_calc_info['indicatorList']):
            # 计算二维时间域指标（基于指标列表）
            if indicator == 'RMS':
                rms_value, xi2 = twodtd_rms(calc_vib, calc_size)
                # 判断是否需要转换dB
                if twodtd[j]['yUnit'] == 'dB':
                    twodtd[j]['yValue'].append(
                        db_convertion(rms_value,
                                      time_domain_calc_info['refValue'][
                                          sensor_index]))
                else:
                    twodtd[j]['yValue'].append(rms_value)
                twodtd[j]['xValue'].append(x_value)
            elif indicator == 'Crest':
                crest_value, max_value, xi2 = twodtd_crest(calc_vib, calc_size)
                twodtd[j]['yValue'].append(crest_value)
                twodtd[j]['xValue'].append(x_value)
            elif indicator == 'Kurtosis':
                kur_value, mean_value, xi4, xi3, xi2 = twodtd_kurtosis(calc_vib,
                                                                       calc_size)
                twodtd[j]['yValue'].append(kur_value)
                twodtd[j]['xValue'].append(x_value)
            elif indicator == 'Skewness':
                skew_value, mean_value, xi3, xi2 = twodtd_skewness(calc_vib,
                                                                   calc_size)
                twodtd[j]['yValue'].append(skew_value)
                twodtd[j]['xValue'].append(x_value)
            elif indicator == "SPL":
                spl_value, xi2 = twod_spl(calc_vib, calc_size)
                twodtd[j]['yValue'].append(spl_value)
                twodtd[j]['xValue'].append(x_value)
            elif indicator == "SPL(A)":
                # 生成A计权滤波器
                B, A = A_weighting(fs)
                if calc_size > 500:
                    a_weighting_calc_vib = lfilter(B, A, calc_vib)[500:]
                else:
                    a_weighting_calc_vib = lfilter(B, A, calc_vib)
                spl_value, xi2_A = twod_spl(a_weighting_calc_vib, calc_size)
                twodtd[j]['yValue'].append(spl_value)
                twodtd[j]['xValue'].append(x_value)
            else:
                logging.info(
                    "error, no this indicator calculation for now, please check the indicator name!")

        # record the temp value of twodTD，记录二维时间域指标计算的中间量
        if [x for x in time_domain_calc_info['indicatorList'] if
            x in ['RMS', 'Crest', 'Kurtosis', 'Skewness', 'SPL']]:
            # 确认需要记录平方和值xi2，而且需要避免重复记录
            temptd['xi2'].append(xi2)
        if [x for x in time_domain_calc_info['indicatorList'] if
            x in ['Crest']]:
            # 确认需要记录最大值值xmax，而且需要避免重复记录
            temptd['xmax'].append(max_value)
        if [x for x in time_domain_calc_info['indicatorList'] if
            x in ['Kurtosis', 'Skewness']]:
            # 确认需要记录三次方和值xi3和平均值xmean，而且需要避免重复记录
            temptd['xi3'].append(xi3)
            temptd['xmean'].append(mean_value)
        if [x for x in time_domain_calc_info['indicatorList'] if
            x in ['Kurtosis']]:
            # 确认需要记录四次方和值xi4，而且需要避免重复记录
            temptd['xi4'].append(xi4)
        if [x for x in time_domain_calc_info['indicatorList'] if x == 'SPL(A)']:
            # 确认需要记录A计权平方和值xi2_A，而且需要避免重复记录
            temptd['xi2_A'].append(xi2_A)
        # 计算完一段数据则记一次，方便下一次计算提取数据
        counter_td += 1
    return twodtd, temptd, counter_td


def angular_resampling(rev, rev_time, target_vib, target_time, points, dr_bf, dr_af):
    """
    功能：进行角度域重采样，得到等角度间隔的振动或声音信号，整段重采样（不考虑分帧）
    输入：
    1. 累积圈数序列
    2. 累积圈数序列对应的时间
    3. 累积振动序列
    4. 累积振动序列对应的时间
    5. 计数器，记录进行过多少次角度域重采样
    6. 需要切割的点数，防止边缘效应
    7. 抗混叠滤波前的角度域重采样间隔，根据该测试段最小转速来定（最小转速一维这每圈实际采集到的时域点数最多）
    8. 目标采样间隔，根据最大关注阶次确定
    返回：
    1. 角度域重采样后的振动或声音序列
    2. 角度域重采样后的振动或声音序列对应的时间
    """
    # 记录开始点结束点的时间，一遍最后切割出目标段
    end_time_flag = target_time[-points]

    # 对累积圈数进行插值，使其与振动信号序列等长
    rev_interp = interp1d(rev_time, rev, kind='linear', fill_value='extrapolate', assume_sorted=True)
    target_rev = rev_interp(target_time)

    # 根据抗混叠滤波前的等角度采样间隔生成采样序列，dr_bf=min_speed / 60 / Fs，即该测试段最高采样率
    # dr for the points of 1 rotation, high dr to make more points,
    # interval is dr_bf = min_speed / 60 / Fs
    # 每次都用该测试段已有的所有数据进行插值
    rev_rsp = np.arange(target_rev[0] // dr_bf, target_rev[-1] // dr_bf) * dr_bf + dr_bf
    # 根据rev_rsp得到角度域重采样后的振动信号序列
    # create an array of vib to match the rev_rsp
    vib_rsp_bf = interp1d(target_rev, target_vib, kind='cubic', assume_sorted=True)(rev_rsp)

    # 抗混叠滤波，避免在降采样时出现干扰信息dr_af = 1/ max_order / 2 / 1.6384，预留一定的空间来避免过渡带
    # Anti-aliasing
    # bf是间隔，bf<af 用af得到的采样点少于bf
    # 从[int(1 / dr_af) / 2] 截止频率 默认低通 int(1 / dr_bf)是vib_rsp_bf的采样率、
    # vib_rsp_bf的长度和vib_rsp_af的长度相同
    vib_rsp_af = butter_filter(vib_rsp_bf, [int(1 / dr_af) / 2], int(1 / dr_bf))

    # 角度域降采样，降至dr_af，根据最大关注阶次来定，重采样率高了浪费计算力，低了取不到用户关注的阶次
    # down resampling, interval dr_af = 1/ max_order / 2 / 1.6384
    # 生成角度域重采样的等角度间隔序列（目标采样率）
    rev_rsp_af = np.arange(target_rev[0] // dr_af, target_rev[-1] // dr_af) * dr_af + dr_af
    # 基于rev_rsp_af插值得到振动信号的重采样序列（三次方比较符合振动信号的变化规律），fill_value指的是边缘多出来的部分继续插值得到
    vib_rsp = interp1d(rev_rsp, vib_rsp_af, kind='cubic', fill_value='extrapolate', assume_sorted=True)(
        rev_rsp_af)
    # 基于rev_rsp_af插值得到振动信号对应的时间重采样序列（时间是线性变化的）
    vib_rsp_time = interp1d(target_rev, target_time, kind='linear', fill_value='extrapolate',
                            assume_sorted=True)(rev_rsp_af)
    # 根据之前记录的时间标志切割并获取目标段振动信号和对应的时间序列，但需要区别第一帧（不需要截取头部）
    target_part = (vib_rsp_time <= end_time_flag)
    return vib_rsp[target_part], vib_rsp_time[target_part]


def create_empty_threedos(order_spectrum_calc_info, task_info, sensor_index, db_flag=0,
                          indicator_diagnostic=-1):
    """
    功能：初始化三维阶次谱结果，用于实时阶次谱结果记录更新
    输入：
    1. 阶次谱计算信息
    2. 数据采集信息，用于提供单位
    3. 传感器索引，用于获取指定单位
    4. 单位是否为dB，默认不是（0表示不是，1表示是）
    5. 初始化的评判结果，默认是为-1（表示缺失）
    返回：不包含具体数值的三维阶次谱结果
    """
    if db_flag:
        target_unit = 'dB'
    else:
        target_unit = task_info['sensorUnit'][sensor_index]
    threedos_result = {
        'xName': order_spectrum_calc_info['xName'],
        'xUnit': order_spectrum_calc_info['xUnit'],
        'xValue': list(),
        'yName': 'Order',
        'yUnit': '',
        'yValue': list(order_spectrum_calc_info['order']),
        'zName': 'Order Spectrum',
        'zUnit': target_unit,
        'zValue': list(),
        "indicatorDiagnostic": indicator_diagnostic,
        'tempXi2': np.zeros(len(order_spectrum_calc_info['order']))
    }
    # 其中tempXi2是用于二维阶次谱计算
    return threedos_result


def create_empty_twodoc(order_cut_calc_info, task_info, sensor_index, db_flag=0,
                        indicator_diagnostic=-1):
    """
    功能：初始化二维阶次切片结果，用户实时计算中进行更新
    输入：
    1. 阶次切片计算参数，主要包含关注阶次和阶次宽度
    2. 数据采集信息，主要提供单位
    3. 传感器索引，用于获取指定的单位
    4. 单位是否为dB， 默认不是（0表示不是，1表示是）
    返回：不包含具体数值的二维阶次切片结果
    """
    twodoc_result = list()
    if db_flag:
        target_unit = 'dB'
    else:
        target_unit = task_info['sensorUnit'][sensor_index]
    for indicator in order_cut_calc_info['orderName']:
        twodoc_result.append({
            'xName': order_cut_calc_info['xName'],
            'xUnit': order_cut_calc_info['xUnit'],
            'xValue': list(),
            'yName': indicator,
            # follow the unit of NI Unit, g or Pa, or dB
            'yUnit': target_unit,
            'yValue': list(),
            "indicatorDiagnostic": indicator_diagnostic
        })
    return twodoc_result


def order_spectrum(threed_os, twod_oc, counter_or, vibration_rsp,
                   vibration_rsp_time,
                   order_spectrum_calc_info, order_cut_calc_info, sensor_index,
                   db_flag=0):
    """
    功能：计算三维阶次谱和二维阶次切片结果
    输入：
    1. 实时更新的三维结果集，第一次为空结果集
    2. 实时更新的二维阶次切片结果集，第一次为空结果集
    3. 计数器，记录进行了多少次阶次谱计算
    4. 角度域重采样后的振动信号序列
    5. 角度域重采样后的振动信号序列对应的时间序列
    6. 阶次谱计算参数信息，包括最大关注阶次，阶次分辨率等信息
    7. 阶次切片计算参数信息，包括关注阶次列表和阶次宽度
    8. 传感器索引，主要为了获取单位和db转换的参考值，以及窗函数
    9. 是否进行db转换
    返回：
    1. 三维阶次谱结果
    2. 二维阶次切片结果
    3. 计算器
    function: apply FFT(Fast Fourier Transform) to angular resampled vibration signal and get order spectrum
    (3D OS amd 2D OC)
    :param
    threed_os(dict): include the speed or time list for X, order list for Y, and order spectrum for Z
    twod_oc(dict): include the speed or time list for X, order spectrum for Y
    counter_or(int): used for cut off the vib data each time, initial value is zero
    vibration_rsp(list): overall angular resampled vibration signal
    vibration_rsp_xaxis(list): corresponding time list for resampled vibration data(time or speed due to angular
    resampling)
    order_spectrum_calc_info(dict): the parameters for order spectrum calculation
        revNum(int): revolution num to do fft one time
        overlapRatio(float): the overlap between two calculation, with revNum will know the step of order spectrum
        calculation
        dr (float): revolution interval between samples of acceleration signal
        ord_max (float): maximum order of the order spectrum, default is None and will return all the orders
        window (string): define the window applied for vib data before fft, default is 'hanning'
    ordercutCal(dict): the parameters for order cut calculation
    :returns
    threed_os(dict): update the 3D order spectrum result that include the info below:
        idx (numpy.ndarray): revolution index of the spectrum(time or speed)
        ord (numpy.ndarray): order of the spectrum
        mag (numpy.ndarray): magnitude of the spectrum
    twod_oc(dict): update the 2D order spectrum that include the time/speed, order amplitude
    counter_or(int): update and feedback to next frame
    """
    while order_spectrum_calc_info['nfft'] + order_spectrum_calc_info[
        'nstep'] * counter_or <= len(vibration_rsp):
        # 第一部分：计算三维阶次谱
        # 记录目标端数据对应的时间（开始和结束时间）
        # x_value = [
        #     vibration_rsp_time[order_spectrum_calc_info['nstep'] * counter_or],
        #     vibration_rsp_time[
        #         order_spectrum_calc_info['nstep'] * counter_or +
        #         order_spectrum_calc_info['nfft'] - 1]]

        # 下面这句是记录中间点时间的代码（其实可以直接获取中间点的速度值）
        x_value = vibration_rsp_time[order_spectrum_calc_info['nstep'] * counter_or + order_spectrum_calc_info['nfft']
                                     // 2]

        # 提取指定圈数的振动或声音信号
        vib = vibration_rsp[order_spectrum_calc_info['nstep'] * counter_or:
                            order_spectrum_calc_info['nfft'] +
                            order_spectrum_calc_info['nstep'] * counter_or]

        # 对目标段信号加窗
        wvib = np.array(vib) * order_spectrum_calc_info['win'][sensor_index]

        # 傅里叶变换得到阶次谱
        fvib = np.abs(np.fft.rfft(wvib, order_spectrum_calc_info['nfft'])) * \
               order_spectrum_calc_info['normFactor'][
                   sensor_index]
        # 直流信号修正
        fvib[0] = fvib[0] / 2
        # 提取最大阶次范围内的阶次谱
        if order_spectrum_calc_info['maxOrder']:
            fvib = fvib[:len(order_spectrum_calc_info['order'])]
        # 更新计算结果到三维阶次谱结果中
        threed_os["xValue"].append(x_value)
        # 记录中间值用于计算二维平均阶次谱
        threed_os["tempXi2"] += fvib * fvib
        if db_flag:
            threed_os["zValue"].append(db_convertion(fvib,
                                                     order_spectrum_calc_info[
                                                         'refValue'][
                                                         sensor_index]).tolist())
        else:
            threed_os["zValue"].append(list(fvib))

        # 第二部分：计算二维阶次切片结果
        if twod_oc is not None:
            # 确保已创建过二维阶次切片结果集
            # calculate the 2D order cutting(consider that order also should include sub order)
            # 下面的关注阶次切片计算方法与一维阶次切片指标一致，不再做解释
            for i, order in enumerate(order_cut_calc_info['orderList']):
                temp_result = list()
                for suborder in order:
                    # the target order should be in the range(right and left side all have enough points)
                    target_order_index = round(
                        suborder / order_spectrum_calc_info[
                            'orderResolution'])
                    temp_result.append(
                        np.sqrt(sum(np.power((fvib[target_order_index -
                                                   order_cut_calc_info[
                                                       'pointNum'] // 2:
                                                   target_order_index +
                                                   order_cut_calc_info[
                                                       'pointNum'] // 2 + 1]),
                                             2))))
                if db_flag:
                    twod_oc[i]['yValue'].append(
                        db_convertion(rms(temp_result),
                                      order_spectrum_calc_info['refValue'][
                                          sensor_index]))
                else:
                    twod_oc[i]['yValue'].append(rms(temp_result))
                twod_oc[i]['xValue'].append(x_value)

        # 更新计数器
        counter_or = counter_or + 1
    return threed_os, twod_oc, counter_or


def twod_order_spectrum(threed_os, task_info, sensor_index, indicator_diagnostic=-1):
    """
    功能：基于三维阶次谱结果计算二维平均阶次谱, 不能提前转换db，因为还需要用于计算一维阶次切片和倒阶次谱
    输入：
    1. 三维阶次谱结果
    2. 数据采集信息，提供单位
    3. 传感器索引，用于获取指定单位信息
    返回：二维平均阶次谱
    function: calculate the 2D order spectrum based on the 3D order spectrum
    :param
    threed_os(dict): 3D order spectrum result(time/speed, order, amplitude)
    task_info(dict): to update the unit of indicators
    :return
    twodos_result(list): 2D order spectrum result(order, amplitude)
    """
    twodos_result = list()
    if len(threed_os['xValue']):
        # threed_os['xValue']是array,因此不能写if threed_os['xValue']
        # 用threed_os['xValue'].any()或者threed_os['xValue'].all()判断
        # 如果存在阶次谱结果才进行计算，不然直接返回空
        twodos_result.append({
            'xName': 'Order',
            'xUnit': '',
            'xValue': threed_os['yValue'],
            'yName': 'Order Spectrum',
            'yUnit': task_info['sensorUnit'][sensor_index],
            'yValue': np.sqrt(threed_os['tempXi2'] / len(threed_os['xValue'])),
            "indicatorDiagnostic": indicator_diagnostic
        })
    return twodos_result


def oned_order_spectrum(twod_os, oned_os_calc_info, task_info,
                        modulationDepthCalcInfo, sensor_index, db_flag=0, indicator_diagnostic=-1):
    """
    功能：计算一维关注阶次指标
    输入：
    1. 二维平均阶次谱
    2. 一维关注阶次参数信息，主要包括关注阶次列表和阶次宽度
    3. 数据采集参数信息，提供单位
    4. 传感器索引，用于获取指定单位信息
    5. 是否转换db，默认否（0表示否，1表示是）
    返回：一维关注阶次指标结果列表
    """
    onedos_result = list()
    if twod_os:
        # 如果存在阶次谱则进行计算
        order_resolution = twod_os[0]['xValue'][1] - twod_os[0]['xValue'][0]
        yValue_array = twod_os[0]['yValue']
        for i, order in enumerate(oned_os_calc_info['orderList']):
            index_array = np.around(
                [[x // order_resolution + k for k in range(-(oned_os_calc_info[
                                                                 'pointNum'] // 2),
                                                           oned_os_calc_info[
                                                               'pointNum'] // 2 + 1)]
                 for x in order]).astype('int')
            temp_result = np.sqrt(
                np.sum(np.power(yValue_array[index_array], 2), axis=1))

            # 记录每个一维关注阶次指标，如果存在多个关注阶次，则求rms值
            if db_flag:
                target_value = db_convertion(rms(temp_result),
                                             task_info['refValue'][
                                                 sensor_index])
                target_unit = 'dB'
            else:
                target_value = rms(temp_result)
                target_unit = task_info['sensorUnit'][sensor_index]
            onedos_result.append({
                'name': oned_os_calc_info['orderName'][i],
                'unit': target_unit,
                'value': target_value,
                'indicatorDiagnostic': indicator_diagnostic
            })

        modulationDepthCalcInfolist = modulationDepthCalcInfo[
            'modulationDepthCalcList']
        # 由于前端不方便传一个空的list,约定若list的第一个元素的调制阶次为0，则不需要计算
        if modulationDepthCalcInfolist[0]["modulationOrder"] != 0:
            # 调制深度指标计算
            for calinfo in modulationDepthCalcInfolist:
                # 调制阶次
                for modulatedOrder in calinfo['modulatedOrder']:
                    numerator_list = [[x + order_resolution * k for k in
                                       range(-(modulationDepthCalcInfo[
                                                   'pointNum'] // 2),
                                             modulationDepthCalcInfo[
                                                 'pointNum'] // 2 + 1)]
                                      for x in [calinfo[
                                                    "modulationOrder"] * i + modulatedOrder * j
                                                for i in
                                                range(1, calinfo[
                                                    "harmonicCount"] + 1) for j
                                                in
                                                list(range(-calinfo[
                                                    'sideFrequencyCount'],
                                                           0)) + list(
                                                    range(1, calinfo[
                                                        'sideFrequencyCount'] + 1))]]

                    numerator_array = np.array(numerator_list)

                    denominator_list = [[x + order_resolution * k for k in
                                         range(-(modulationDepthCalcInfo[
                                                     'pointNum'] // 2),
                                               modulationDepthCalcInfo[
                                                   'pointNum'] // 2 + 1)]
                                        for x in [calinfo[
                                                      "modulationOrder"] * i + modulatedOrder * j
                                                  for i in
                                                  range(1, calinfo[
                                                      "harmonicCount"] + 1) for
                                                  j in list([0])]]

                    # # axis=0 计算每一列的和 axis=1计算每一行的和 np.sum计算每一行的和之后得到一维array
                    # np.sum(np.sqrt(np.sum(np.power(numerator_array,2),axis=1)))

                    denominator_array = np.array(denominator_list)

                    yValue_numerator = yValue_array[
                        np.around(numerator_array // order_resolution).astype(
                            "int")]
                    yValue_denominator = yValue_array[
                        np.around(denominator_array // order_resolution).astype(
                            "int")]

                    # 调制深度文档上的第一种方法
                    onedos_result.append({
                        'name': "MD-" + str(
                            calinfo['modulationOrder']) + '-' + str(
                            modulatedOrder),
                        'unit': None,
                        'value': np.sum(np.sqrt(
                            np.sum(np.power(yValue_numerator, 2),
                                   axis=1))) / np.sum(np.sqrt(
                            np.sum(np.power(yValue_denominator, 2),
                                   axis=1))),
                        'indicatorDiagnostic': indicator_diagnostic
                    })
                    # 调制深度文档上的第二种方法（暂时不用）
                    # onedos_result.append({
                    #     'name': "MD-" + str(calinfo['modulationOrder']) + '-' + str(modulatedOrder) + '-method2',
                    #     'unit': None,
                    #     'value': np.sqrt(np.sum(
                    #         np.sum(np.power(yValue_numerator, 2),
                    #                axis=1))) / np.sqrt(np.sum(
                    #         np.sum(np.power(yValue_denominator, 2),
                    #                axis=1))),
                    #     'indicatorDiagnostic': -1
                    # })
    return onedos_result


def cepstrum(twod_os, cepstrum_calc_info, task_info, sensor_index, db_flag=0, indicator_diagnostic=-1):
    """
    功能：计算倒阶次谱（基于阶次谱进行倒频谱运算）
    输入：
    1. 二维平均阶次谱
    2.倒阶次谱计算参数信息，主要是构造倒阶次谱x轴的圈数信息
    3. 数据采集信息，主要提供单位信息
    4. 传感器索引，用于获取指定单位信息
    返回：倒阶次谱结果
    :param
    twod_os(list): 2D order spectrum result
    cepstrum_calc_info(dict): parameters for cepstrum calculation, include 'orderResolution','overlapRatio'
    task_info(dict): provide the unit info
    :return
    result(list): cepstrum of order spectrum
    """
    ceps_result = list()
    if len(twod_os) > 0:
        # 存在阶次谱才能计算，不然返回空
        # 计算倒阶次谱的值（即y轴信息）
        if db_flag:
            target_value = db_convertion(
                np.fft.ifft(2 * np.log(twod_os[0]['yValue']))[
                :len(twod_os[0]['yValue']) // 2],
                task_info['refValue'][sensor_index]).tolist()
            target_unit = 'dB'
        else:
            target_value = np.abs(np.fft.ifft(2 * np.log(twod_os[0]['yValue']))[
                                  :len(twod_os[0]['yValue']) // 2]).tolist()
            target_unit = task_info['sensorUnit'][sensor_index]
            # 强制归零（直流分量），不然过大影响倒阶次谱结果的查看，这里需要主要的是转换dB时要避开
        target_value[0] = 0  # filter out the DC signal
        ceps_result.append({
            'xName': 'CumulativeRotation ',
            'xUnit': 'r',
            # 构造倒阶次谱的x轴
            'xValue': list(np.linspace(0, cepstrum_calc_info['revNum'] // 2,
                                       len(twod_os[0]['yValue']) // 2,
                                       endpoint=False)),
            'yName': 'Cepstrum',
            'yUnit': target_unit,
            'yValue': target_value,
            "indicatorDiagnostic": indicator_diagnostic
        })
    return ceps_result


def create_empty_twodsf(stat_factor_calc_info, sensor_index, indicator_diagnostic=-1):
    """
    功能：初始化按圈计算的统计学指标结果，用于实时更新
    输入：
    1. 按圈计算参数信息
    2. 传感器索引（用于指定单位）
    3. 初始化的评判结果，默认是为-1（表示缺失）
    返回：包含结构但不包含具体数值的按圈计算结果
    """
    twodsf_result = list()
    if not stat_factor_calc_info['indicatorList']:
        return twodsf_result
    for gearName in stat_factor_calc_info['gearName']:
        for i, indicator in enumerate(stat_factor_calc_info['indicatorList']):
            twodsf_result.append({
                'xName': stat_factor_calc_info['xName'],
                'xUnit': stat_factor_calc_info['xUnit'],
                'xValue': list(),
                'yName': '-'.join([gearName, indicator]),
                'yUnit': stat_factor_calc_info['indicatorUnit'][sensor_index][i],
                'yValue': list(),
                "indicatorDiagnostic": indicator_diagnostic
            })
    return twodsf_result


def create_empty_tempsf(stat_factor_calc_info):
    """
    功能：初始化按圈计算的统计学指标临时结果，用于最后计算一维指标
    输入：计算参数信息
    返回：包含结构但不包含具体数值的临时结果，包括：
    1. 平方和值xi2
    2. 最大值xmax
    3. 均值xmean
    4. 三次方和值xi3
    5. 四次方和值xi4
    每个轴都有相应的结果，所以是按gear name保存的
    """
    tempsf_result = dict()
    if not stat_factor_calc_info['indicatorList']:
        return tempsf_result
    for gearName in stat_factor_calc_info['gearName']:
        tempsf_result[gearName] = dict()
        tempsf_result[gearName]['xi2'] = list()
        tempsf_result[gearName]['xmax'] = list()
        tempsf_result[gearName]['xmean'] = list()
        tempsf_result[gearName]['xi3'] = list()
        tempsf_result[gearName]['xi4'] = list()
        tempsf_result[gearName]['xi2_A'] = list()
        tempsf_result[gearName]['counter'] = 0
    return tempsf_result


def twod_stat_factor(twodsf, tempsf, vibration_rsp, vibration_rsp_time,
                     stat_factor_calc_info, average_flag=1):
    """
    功能：基于角度域重采样的振动数据按圈计算统计学指标
    输入：
    1. 实时更新的二维统计学按圈计算指标结果集，第一次为空
    2. 实时更新的按圈计算统计学指标中间结果，用于计算一维结果
    3. 累积振动数据（角度域重采样后）
    4. 累积振动数据对应的时间序列（角度域重采样后）
    5. 按圈计算参数信息
    6. 是否直接对二维结果作平均得到一维结果，默认为1（表示是），0表示否
    返回：
    1. 实时更新的二维统计学按圈计算指标结果集
    2. 实时更新的按圈计算统计学指标中间结果
    """
    indicatorSet = set(stat_factor_calc_info['indicatorList'])
    if not indicatorSet:
        # 如果指标列表为空则不进行按圈计算
        return twodsf, tempsf
    for i, pointsNum in enumerate(stat_factor_calc_info['pointsNum']):
        # 依次计算各个转轴的结果
        temp_counter = tempsf[stat_factor_calc_info['gearName'][i]]['counter']
        temp_gear_name = stat_factor_calc_info['gearName'][i]
        while pointsNum * (temp_counter + 1) <= len(vibration_rsp):
            # 获取当前转轴的振动数据以及对应的时间
            # 首尾点
            # x_value = [vibration_rsp_time[pointsNum * temp_counter],
            #            vibration_rsp_time[pointsNum * (temp_counter + 1) - 1]]
            # 中间点
            x_value = vibration_rsp_time[round(pointsNum * (temp_counter+0.5))]
            vib = vibration_rsp[
                  pointsNum * temp_counter: pointsNum * (temp_counter + 1)]
            # 计算相关指标(基于指标列表）
            # 初始化临时值
            xi2 = 0
            xi3 = 0
            xi4 = 0
            max_value = 0
            mean_value = 0
            # 根据指标列表分别进行计算
            for j, indicator in enumerate(
                    stat_factor_calc_info['indicatorList']):
                if indicator == 'RMS':
                    rms_value, xi2 = twodtd_rms(vib, pointsNum)
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'yValue'].append(rms_value)
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'xValue'].append(x_value)
                elif indicator == 'Crest':
                    crest_value, max_value, xi2 = twodtd_crest(vib, pointsNum)
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'yValue'].append(crest_value)
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'xValue'].append(x_value)
                elif indicator == 'Kurtosis':
                    kur_value, mean_value, xi4, xi3, xi2 = twodtd_kurtosis(vib,
                                                                           pointsNum)
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'yValue'].append(kur_value)
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'xValue'].append(x_value)
                elif indicator == 'Skewness':
                    skew_value, mean_value, xi3, xi2 = twodtd_skewness(vib,
                                                                       pointsNum)
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'yValue'].append(skew_value)
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'xValue'].append(x_value)
                elif indicator == "SPL":
                    spl_value, xi2 = twod_spl(vib, pointsNum)
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'yValue'].append(spl_value)
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'xValue'].append(x_value)
                elif indicator == "SPL(A)":
                    # 生成A计权滤波器
                    B, A = A_weighting(stat_factor_calc_info['sampleRate'])
                    if pointsNum > 500:
                        a_weighting_calc_vib = lfilter(B, A, vib)[500:]
                    else:
                        a_weighting_calc_vib = lfilter(B, A, vib)
                    spl_value, xi2_A = twod_spl(a_weighting_calc_vib, pointsNum)
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'yValue'].append(spl_value)
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'xValue'].append(x_value)
                else:
                    logging.info(
                        "error, no this indicator calculation for now, please check the indicator name!")
            if not average_flag:
                # 记录统计学指标所需要的的中间值
                if [x for x in stat_factor_calc_info['indicatorList'] if
                    x in ['RMS', 'Crest', 'Kurtosis', 'Skewness',
                          'SPL']]:
                    tempsf[temp_gear_name]['xi2'].append(xi2)
                if [x for x in stat_factor_calc_info['indicatorList'] if
                    x in ['Crest']]:
                    tempsf[temp_gear_name]['xmax'].append(max_value)
                if [x for x in stat_factor_calc_info['indicatorList'] if
                    x in ['Kurtosis', 'Skewness']]:
                    tempsf[temp_gear_name]['xi3'].append(xi3)
                    tempsf[temp_gear_name]['xmean'].append(mean_value)
                if [x for x in stat_factor_calc_info['indicatorList'] if
                    x in ['Kurtosis']]:
                    tempsf[temp_gear_name]['xi4'].append(xi4)
                if [x for x in stat_factor_calc_info['indicatorList'] if
                    x == 'SPL(A)']:
                    # 确认需要记录A计权平方和值xi2_A，而且需要避免重复记录
                    tempsf[temp_gear_name]['xi2_A'].append(xi2_A)
            temp_counter += 1
            tempsf[temp_gear_name]['counter'] += 1
    return twodsf, tempsf


def average_oned_stat_factor(twodsf, stat_factor_calc_info, sensor_index):
    """
    功能：直接基于二维按圈计算结果求平均值得到一维按圈计算结果
    输入：二维按圈计算结果
    返回：一维按圈计算结果
    """
    onedsf_result = list()
    for i, data in enumerate(twodsf):
        # 确认是否进行db转换
        if data['yUnit'] == 'dB':
            # 计算一维结果
            target_value = db_convertion(np.mean(data['yValue']),
                                         stat_factor_calc_info['refValue'][
                                             sensor_index])
            target_unit = 'dB'
            # 更新二维结果
            twodsf[i]['yValue'] = db_convertion(data['yValue'],
                                                stat_factor_calc_info[
                                                    'refValue'][
                                                    sensor_index]).tolist()
            twodsf[i]['yUnit'] = 'dB'
        else:
            target_value = np.mean(data['yValue'])
            target_unit = data['yUnit']
        onedsf_result.append({
            'name': data['yName'],
            'unit': target_unit,
            'value': target_value,
            "indicatorDiagnostic": -1
        })
    return twodsf, onedsf_result


def convert_time_speed(twod_data, curve_x, curve_y, speed_pattern, speed_ratio,
                       threed_os=None, indicator_num=None):
    """
    功能：将X轴由时间转速度（只针对单个传感器单个测试段），同时根据转速比调整速度值
    输入：
    1. 二维数据（包括二维时间域指标，按圈计算结果，二维阶次切片（可同时转换三维阶次彩图））
    2. 转速曲线，内含X轴和Y轴数据
    3. 转速类型，主要是为了区分变速和恒速，1为恒速，2为升速，3为降速
    4. 转速比， 主要用于转速换算
    5. 阶次彩图数据，对于二维时间域和按圈计算结果来说他是None
    6. 按圈计算的指标列表，用于按轴进行
    """
    if speed_pattern > 1:
        # 变速段
        if indicator_num:
            # 按圈计算指标列表存在表示需要按按圈计算指标进行转换，每根轴转换一次
            for i in range(len(twod_data)):
                if i % indicator_num == 0:
                    if speed_ratio != 1:
                        if len(twod_data[i]['xValue']) > 0:
                            # 避免空数据
                            temp_x = (np.interp(twod_data[i]['xValue'],
                                curve_x,
                                curve_y) * speed_ratio).tolist()
                        else:
                            temp_x = list()
                    else:
                        if len(twod_data[i]['xValue']) > 0:
                            # 避免空数据
                            temp_x = np.interp(
                                twod_data[i]['xValue'],
                                curve_x, curve_y).tolist()
                        else:
                            temp_x = list()
                twod_data[i]['xValue'] = temp_x
            return twod_data
        else:
            # 二维时间域或者二维阶次切片
            if speed_ratio != 1:
                if len(twod_data[0]['xValue']) > 0:
                    # 避免空数据
                    temp_x = (np.interp(twod_data[0]['xValue'],
                                        curve_x,
                                        curve_y) * speed_ratio).tolist()
                else:
                    temp_x = list()
            else:
                if len(twod_data[0]['xValue']) > 0:
                    # 避免空数据
                    temp_x = np.interp(twod_data[0]['xValue'],
                                       curve_x, curve_y).tolist()
                else:
                    temp_x = list()
            for i in range(len(twod_data)):
                twod_data[i]['xValue'] = temp_x
            if threed_os:
                # 需要改变三维结果数据的X轴（与阶次切片的X轴相同）
                threed_os['xValue'] = temp_x
                return twod_data, threed_os
            else:
                return twod_data
    else:
        # 恒速段(只需要切换X轴名称即可）
        for i in range(len(twod_data)):
            twod_data[i]['xName'] = 'Time'
            twod_data[i]['xUnit'] = 's'
        if threed_os:
            # 需要改变三维结果数据的X轴（与阶次切片的X轴相同）
            threed_os['xName'] = "Time"
            threed_os['xUnit'] = "s"
            return twod_data, threed_os
        else:
            return twod_data


if __name__ == '__main__':
    # 主程序入口
    try:
        # just for internal test
        # import matplotlib.pyplot as plt
        # import time
        # t1 = time.time()
        config = {'filepath': r'D:\qdaq\test\210925-1',
                  "filename": 'TZ220XS004M20210001M021512002_210518025321_210819054719.h5'}

        # step1: read the input
        try:
            raw_config_filename = os.path.join(config['filepath'], 'configForPython.json')
            recalc_config_filename = os.path.join(config['filepath'], 'configForTestRecalc.json')
            raw_data_filename = os.path.join(config['filepath'], config['filename'])
            raw_config = read_json(raw_config_filename)
            recalc_config = read_json(recalc_config_filename)
        except Exception:
            print("config file not found")
            traceback.print_exc()
            logging.warning("config read exec failed, failed msg:" + traceback.format_exc())
            sys.exit()

        # step2: read the config in folder
        try:
            basicInfo = basic_info_update(raw_config['basicInfo'])
            taskInfo = task_info_update(raw_config['taskInfo'])
            sensor_confirm(taskInfo, basicInfo)
            speedCalcInfo = speed_calc_info_update(raw_config['speedCalcInfo'])
            speedRecogInfo = speed_recog_info_update(raw_config['speedRecogInfo'])
            timeDomainCalcInfo = time_domain_calc_info_update(raw_config["timeDomainCalcInfo"],
                                                              taskInfo, basicInfo)
            orderSpectrumCalcInfo = order_spectrum_calc_info_update(
                raw_config["orderSpectrumCalcInfo"], speedCalcInfo, speedRecogInfo['overallMinSpeed'], taskInfo)
            orderCutCalcInfo = order_cut_calc_info_update(raw_config["orderCutCalcInfo"],
                                                          orderSpectrumCalcInfo)
            onedOSCalcInfo = oned_os_calc_info_update(raw_config["onedOSCalcInfo"],
                                                      orderSpectrumCalcInfo)
            modulationDepthCalcInfo = raw_config["modulationDepthCalcInfo"]
            cepstrumCalcInfo = cepstrum_calc_info_update(orderSpectrumCalcInfo)
            ssaCalcInfo = ssa_calc_info_update(raw_config["ssaCalcInfo"], speedCalcInfo)
            statFactorCalcInfo = stat_factor_calc_info_update(raw_config['indicatorByRevCalcInfo'],
                                                              ssaCalcInfo, orderSpectrumCalcInfo,
                                                              taskInfo, basicInfo)
            sampleRate = taskInfo['sampleRate']
            sampsPerChan = taskInfo['sampsPerChan']
        except Exception:
            print("config info error")
            traceback.print_exc()
            logging.warning("config info change failed, failed msg:" + traceback.format_exc())
            sys.exit()
        try:
            # 读取传感器数据
            sensor_data = raw_data_read(raw_data_filename, taskInfo['targetChan'])
            # 读取转速曲线
            rpmSpeedLocation, rpmSpeedValue = read_speed_curve(raw_data_filename)
            # 读取trigger信息
            triggerLocation = speed_trigger_read(raw_data_filename)
        except Exception:
            print("data reading error")
            traceback.print_exc()
            logging.warning("config info change failed, failed msg:" + traceback.format_exc())
            sys.exit()

        try:
            test_result = create_empty_final_result(basicInfo, recalc_config)

            # 提取目标段的转速曲线
            speed_condition = (rpmSpeedLocation >= recalc_config['startTime']) & (
                    rpmSpeedLocation <= recalc_config['endTime'])
            # 开始点结束点的索引
            startIndex, endIndex = int(recalc_config['startTime'] * sampleRate), int(
                recalc_config['endTime'] * sampleRate)

            # 提取目标段的trigger信息
            target_trigger = triggerLocation[
                            (triggerLocation >= startIndex) & (triggerLocation <= endIndex)]

            startIndex_fix = target_trigger[0]
            startTime_fix = startIndex_fix / sampleRate

            target_trigger_fix = target_trigger - startIndex_fix

            # 开始计算
            for sensor_i, chan_name in enumerate(taskInfo['targetChan']):
                # cut the vibration
                target_data = sensor_data[chan_name][startIndex_fix:endIndex]
                twod_td = create_empty_twodtd(timeDomainCalcInfo, sensor_i)
                temp_td = create_empty_temptd()
                counter_td = 0
                # 二维时域指标计算
                twod_td, temp_td, counter_td = twod_time_domain(twod_td,
                                                                temp_td,
                                                                counter_td, target_data,
                                                                sampleRate,
                                                                startTime_fix,
                                                                timeDomainCalcInfo,
                                                                sensor_i)
                # 一维时间域指标
                oned_td = oned_time_domain(temp_td, timeDomainCalcInfo, sensor_i, db_flag=basicInfo['dBFlag'])


                # 角度域重采样
                # trigger对应的圈数
                rev = (np.arange(0, len(target_trigger_fix))) / speedCalcInfo['ppr']
                # trigger对应的时刻点，以测试段开始点为0时刻
                rev_time = target_trigger_fix / sampleRate
                vib_rsp, vib_rsp_time = angular_resampling(rev,
                                                           rev_time,
                                                           target_data, np.arange(0, len(target_data)) / sampleRate,
                                                           orderSpectrumCalcInfo[
                                                               'arPoints'],
                                                           speedRecogInfo['overallMinSpeed'] / 60 / sampleRate,
                                                           orderSpectrumCalcInfo[
                                                               'dr_af'])
                vib_rsp_time = vib_rsp_time + startTime_fix
                # 三维阶次谱和二维阶次切片
                threed_os = create_empty_threedos(orderSpectrumCalcInfo,
                                                  taskInfo, sensor_i,
                                                  db_flag=basicInfo['dBFlag'])
                twod_oc = create_empty_twodoc(orderCutCalcInfo,
                                              taskInfo, sensor_i,
                                              db_flag=basicInfo['dBFlag'])
                counter_or = 0
                threed_os, twod_oc, counter_or = order_spectrum(threed_os,
                                                                twod_oc,
                                                                counter_or,
                                                                vib_rsp, vib_rsp_time,
                                                                orderSpectrumCalcInfo,
                                                                orderCutCalcInfo,
                                                                sensor_i,
                                                                db_flag=basicInfo[
                                                                    'dBFlag'])
                # 二维平均阶次谱(先不进行db转换，一维阶次指标和倒阶次谱计算完之后再转换）
                twod_os = twod_order_spectrum(threed_os, taskInfo, sensor_i)

                # 一维关注阶次指标
                oned_os = oned_order_spectrum(twod_os, onedOSCalcInfo,
                                              taskInfo, modulationDepthCalcInfo, sensor_i,
                                              db_flag=basicInfo['dBFlag'])

                # 二维倒阶次谱
                twod_ceps = cepstrum(twod_os, cepstrumCalcInfo, taskInfo, sensor_i,
                                     db_flag=basicInfo['dBFlag'])

                # 更新二维阶次谱结果(db和阶次切换）
                if basicInfo['dBFlag']:
                    twod_os[0]['yUnit'] = 'dB'
                    twod_os[0]['yValue'] = db_convertion(twod_os[0]['yValue'],
                                                         taskInfo['refValue'][
                                                             sensor_i]).tolist()
                else:
                    twod_os[0]['yValue'] = list(twod_os[0]['yValue'])
                if speedCalcInfo['speedRatio'] != 1:
                    twod_os[0]['xValue'] = orderSpectrumCalcInfo['convertOrder']

                # 按圈计算
                twod_sf = create_empty_twodsf(statFactorCalcInfo, sensor_i)
                temp_sf = create_empty_tempsf(statFactorCalcInfo)
                # 按圈计算
                twod_sf, temp_sf = twod_stat_factor(twod_sf, temp_sf, vib_rsp,
                                                    vib_rsp_time, statFactorCalcInfo)
                twod_sf, oned_sf_average = average_oned_stat_factor(twod_sf,
                                                                    statFactorCalcInfo,
                                                                    sensor_i)

                # 时间转速切换(需要先确定是否为恒速即获取speedPattern)
                test_index = speedRecogInfo['testName'].index(recalc_config['testName'])
                twod_td = convert_time_speed(twod_td, rpmSpeedLocation[speed_condition],
                                    rpmSpeedValue[speed_condition], speedRecogInfo['speedPattern'][test_index], speedCalcInfo['speedRatio'])
                twod_oc, threed_os = convert_time_speed(twod_oc,
                                                        rpmSpeedLocation[speed_condition],
                                                        rpmSpeedValue[speed_condition],
                                                        speedRecogInfo['speedPattern'][test_index],
                                                        speedCalcInfo['speedRatio'],
                                                        threed_os=threed_os)
                twod_sf = convert_time_speed(twod_sf, rpmSpeedLocation[speed_condition],
                                    rpmSpeedValue[speed_condition], speedRecogInfo['speedPattern'][test_index], speedCalcInfo['speedRatio'], indicator_num=statFactorCalcInfo['indicatorNum'])

                # 更新结果数据
                test_result['resultData'][sensor_i]['dataSection'][0]['startTime'] = startTime_fix
                test_result['resultData'][sensor_i]['dataSection'][0]['twodTD'] = twod_td
                test_result['resultData'][sensor_i]['dataSection'][0]['onedData'] = oned_td
                test_result['resultData'][sensor_i]['dataSection'][0]['twodTD'].extend(twod_sf)
                test_result['resultData'][sensor_i]['dataSection'][0]['onedData'].extend(oned_sf_average)
                test_result['resultData'][sensor_i]['dataSection'][0]['onedData'].extend(oned_os)
                test_result['resultData'][sensor_i]['dataSection'][0]['twodOC'] = twod_oc
                test_result['resultData'][sensor_i]['dataSection'][0]['twodOS'] = twod_os
                test_result['resultData'][sensor_i]['dataSection'][0]['twodCeps'] = twod_ceps

        except Exception:
            print("test result recalc error")
            traceback.print_exc()
            logging.warning("config info change failed, failed msg:" + traceback.format_exc())
            sys.exit()

        # print(time.time() - t1)
        # step6: return test result
        try:
            # print(json.dumps(test_result))
            with open(os.path.join(config['filepath'], 'recalcTestResult.json'), 'w') as f:
                json.dump(test_result, f)
            print("Done")
        except Exception:
            print("test result package error")
            traceback.print_exc()
            logging.warning("test result recalc failed, failed msg:" + traceback.format_exc())
            sys.exit()
        # print(time.time() - t1)
    except Exception:
        print("other error")
        traceback.print_exc()
        logging.warning("test result recalc script exec failed, failed msg:" + traceback.format_exc())
        sys.exit()
