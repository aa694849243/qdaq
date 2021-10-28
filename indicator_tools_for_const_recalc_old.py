# -*- coding: utf-8 -*-
"""
Created on Wed Mar 4

@author: Wall@Synovate

function: define the function do calculate the indicators(onedData, twodTD, twodOC, twodOS, SSA, Cepstrum)
功能：该模块主要包含了指标计算的函数，包括时间域指标，阶次切片指标，阶次谱指标，时频谱，ssa等指标的计算方法
update: 2021/06/18 18:36

"""
from scipy import signal
from scipy.signal import butter, filtfilt, lfilter
from scipy.signal import resample
from scipy.interpolate import interp1d
import math
import numpy as np
import traceback
import os
import sys
import time
from numpy import pi, convolve
from scipy.signal.filter_design import bilinear


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


def db_convertion(data, ref_value):
    """
    功能：转换为dB值
    输入：
    1. 结果数据，可以是一维或者二维数据
    2. 参考值，用于db的计算
    返回：转换后的结果
    """
    return 20 * np.log10(np.abs(data) / ref_value)


def oned_time_domain(temptd, calc_size, time_domain_calc_info, sensor_index,
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
    function: update the 1D indicators value based on the twodData(RMS of the all the data of this test)
    :param
    temptd(dict): temp value from 2D time domain indicators calculation
    calc_size(int): the length of data(frame) in 2D time domain indicators calculation
    time_domain_calc_info(dict): parameters for time domain indicators calculations, include the info below:
        Cal_list(list): time domain calculation list, ["RMS","Crest"]
        Unit_list(list): corresponding unit list for indicator in Cal_list
    :return
    result(list): the indicator result value(twodTD calculation already finished)
    """
    onedtd_result = list()
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
        elif indicator == 'Speed':
            pass
    return onedtd_result



def twod_time_domain_for_share(vib, last_calc_index, right_index, calc_size,
                               time_domain_calc_info, sensor_index, twodtd,
                               temptd, counter_td, fs, index_twod_td
                               ):
    """
    功能：二维时间域指标实时计算函数（共享内存模式）
    """
    # 当前累积到的数据可以进行几次计算
    count = (right_index - last_calc_index) // calc_size
    for i in range(count):
        calc_vib = vib[last_calc_index:last_calc_index + calc_size]

        xi2 = 0
        xi3 = 0
        xi4 = 0
        max_value = 0
        mean_value = 0
        # 滤波之后的x^2
        xi2_A = 0
        # 本计算帧的中间时刻点
        xValue = (last_calc_index + calc_size / 2) / fs

        for j, indicator in enumerate(time_domain_calc_info['indicatorList']):
            # 计算二维时间域指标（基于指标列表）
            if indicator == 'RMS':
                rms_value, xi2 = twodtd_rms(calc_vib, calc_size)
                # 判断是否需要转换dB
                if twodtd[j]['yUnit'] == 'dB':
                    twodtd[j]['yValue'][index_twod_td] = \
                        db_convertion(rms_value,
                                      time_domain_calc_info['refValue'][
                                          sensor_index])
                else:
                    twodtd[j]['yValue'][index_twod_td] = rms_value
                twodtd[j]['xValue'][index_twod_td] = xValue
            elif indicator == 'Crest':
                crest_value, max_value, xi2 = twodtd_crest(calc_vib, calc_size)
                twodtd[j]['yValue'][index_twod_td] = crest_value
                twodtd[j]['xValue'][index_twod_td] = xValue
            elif indicator == 'Kurtosis':
                kur_value, mean_value, xi4, xi3, xi2 = twodtd_kurtosis(calc_vib,
                                                                       calc_size)
                twodtd[j]['yValue'][index_twod_td] = kur_value
                twodtd[j]['xValue'][index_twod_td] = xValue
            elif indicator == 'Skewness':
                skew_value, mean_value, xi3, xi2 = twodtd_skewness(calc_vib,
                                                                   calc_size)
                twodtd[j]['yValue'][index_twod_td] = skew_value
                twodtd[j]['xValue'][index_twod_td] = xValue
            elif indicator == "SPL":
                spl_value, xi2 = twod_spl(calc_vib, calc_size)
                twodtd[j]['yValue'][index_twod_td] = spl_value
                twodtd[j]['xValue'][index_twod_td] = xValue
            elif indicator == "SPL(A)":
                # 生成A计权滤波器
                B, A = A_weighting(fs)
                if calc_size > 500:
                    a_weighting_calc_vib = lfilter(B, A, calc_vib)[500:]
                else:
                    a_weighting_calc_vib = lfilter(B, A, calc_vib)
                spl_value, xi2_A = twod_spl(a_weighting_calc_vib, calc_size)
                twodtd[j]['yValue'][index_twod_td] = spl_value
                twodtd[j]['xValue'][index_twod_td] = xValue
            elif indicator=="Speed":
                pass

        # record the temp value of twodTD，记录二维时间域指标计算的中间量
        indicatorSet = set(time_domain_calc_info['indicatorList'])
        if {'RMS', 'Crest', 'Kurtosis', 'Skewness', 'SPL'} & indicatorSet:
            # 确认需要记录平方和值xi2，而且需要避免重复记录
            temptd['xi2'][index_twod_td] = xi2
        if {'Crest'} & indicatorSet:
            # 确认需要记录最大值值xmax，而且需要避免重复记录
            temptd['xmax'][index_twod_td] = max_value

        if {'Kurtosis', 'Skewness'} & indicatorSet:
            # 确认需要记录三次方和值xi3和平均值xmean，而且需要避免重复记录
            temptd['xi3'][index_twod_td] = xi3
            temptd['xmean'][index_twod_td] = mean_value
        if {'Kurtosis'} & indicatorSet:
            # 确认需要记录四次方和值xi4，而且需要避免重复记录
            temptd['xi4'][index_twod_td] = xi4
        if {'SPL(A)'} & indicatorSet:
            # 确认需要记录A计权平方和值xi2_A，而且需要避免重复记录
            temptd['xi2_A'][index_twod_td] = xi2_A

        # 最后计算点增加
        last_calc_index += calc_size
        # twod_td和temp_td下一次要放入的索引增加
        index_twod_td += 1

        # 计算完一段数据则记一次，方便下一次计算提取数据
        counter_td += 1
    return twodtd, temptd, counter_td, index_twod_td, last_calc_index


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
            'yUnit': task_info['units'][task_info["indicatorsUnitChanIndex"][sensor_index]],
            'yValue': np.sqrt(threed_os['tempXi2'] / len(threed_os['xValue'])),
            "indicatorDiagnostic": indicator_diagnostic
        })
    return twodos_result


def oned_order_spectrum(twod_os, oned_os_calc_info, task_info,
                        modulationDepthCalcInfo, sensor_index, db_flag=0):
    """
    功能：计算一维关注阶次指标
    输入：
    1. 二维平均阶次谱
    2. 一维关注阶次参数信息，主要包括关注阶次列表和阶次宽度
    3. 数据采集参数信息，提供单位
    4. 传感器索引，用于获取指定单位信息
    5. 是否转换db，默认否（0表示否，1表示是）
    返回：一维关注阶次指标结果列表
    function: to calculate the 1D order spectrum(must after 2D order spectrum finished)
    :param
    twod_os(list): include the overall 2D order spectrum data
    oned_os_calc_info(dict): include the info below:
        orderList(list): the order that user focus on, if this order is not exact the order inside, find the nearest one
        orderName(list): corresponding name for the orderList
        pointNum(int): how many point to include in from in side of the target order, e.gi:if it is 2,
                       means 5 points needed
        maxOrder(float): maximum order
        orderResolution(float): order Resolution
    task_info(dict): to update the unit of indicators
    :return
    onedos_result(list): 1D order spectrum data into result
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
                target_unit = task_info['units'][
                    task_info["indicatorsUnitChanIndex"][sensor_index]]
            onedos_result.append({
                'name': oned_os_calc_info['orderName'][i],
                'unit': target_unit,
                'value': target_value,
                'indicatorDiagnostic': -1
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
                        'indicatorDiagnostic': -1
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


def angular_resampling_old(overall_rev, overall_rev_time, overall_vib, overall_time,
                           counter, samps_per_chan, points,
                           dr_bf, dr_af):
    """
    功能：进行角度域重采样，得到等角度间隔的振动或声音信号，进而计算阶次谱
    输入：
    1. 累积圈数序列
    2. 累积圈数序列对应的时间
    3. 累积振动序列
    4. 累积振动序列对应的时间
    5. 计数器，记录进行过多少次角度域重采样
    6. 每帧的长度，重采样时是按帧处理的
    7. 需要切割的点数，防止边缘效应
    8. 抗混叠滤波前的角度域重采样间隔，根据该测试段最小转速来定（最小转速一维这每圈实际采集到的时域点数最多）
    9. 目标采样间隔，根据最大关注阶次确定
    返回：
    1. 角度域重采样后的振动或声音序列
    2. 角度域重采样后的振动或声音序列对应的时间
    function: angular resampling of vibration (time to angular)
    :param
    overall_rev(list): overall accumulative revolutions
    overall_rev_time(list): corresponding time list of overall accumulative revolutions
    overall_vib(list): overall accumulative vibration
    overall_time(list): corresponding time list of overall vibration
    counter(int): counter of angular resampling to index target data
    samps_per_chan(int): length of 1 frame data
    points(int): overlapped point to avoid boundary effect
    dr_bf(float): revolution interval between samples of resultant vibration signal(before filter)
    dr_af(float): revolution interval between samples of resultant vibration signal(after filter)
    :return
    vib_rsp(list): resampled vibration data
    vib_rsp_time(list): corresponding time of resampled vibration data
    """
    if counter > 0:
        # 不是第一帧需要往前一帧多取一些点（滤波带来的边缘效应），然后由下一次计算的结果补上
        s_index = counter * samps_per_chan - points * 2
    else:
        # 第一帧无法往前取点
        s_index = 0
    # 确定结束点的索引
    e_index = (counter + 1) * samps_per_chan

    # 记录开始点结束点的时间，一遍最后切割出目标段
    start_time_flag = overall_time[s_index + points]
    end_time_flag = overall_time[e_index - points]

    # 根据开始点结束点索引切割目标段振动或者声音信号
    target_vib = overall_vib[s_index:e_index]
    target_time = overall_time[s_index:e_index]

    # 对累积圈数进行插值，使其与振动信号序列等长
    rev_interp = interp1d(overall_rev_time, overall_rev, kind='linear',
                          fill_value='extrapolate', assume_sorted=True)
    target_rev = rev_interp(target_time)

    # 根据抗混叠滤波前的等角度采样间隔生成采样序列，dr_bf=min_speed / 60 / Fs，即该测试段最高采样率
    # dr for the points of 1 rotation, high dr to make more points,
    # interval is dr_bf = min_speed / 60 / Fs
    rev_rsp = np.arange(target_rev[0] // dr_bf,
                        target_rev[-1] // dr_bf) * dr_bf + dr_bf
    # 根据rev_rsp得到角度域重采样后的振动信号序列
    # create an array of vib to match the rev_rsp
    vib_rsp_bf = interp1d(target_rev, target_vib, kind='cubic',
                          assume_sorted=True)(rev_rsp)

    # 抗混叠滤波，避免在降采样时出现干扰信息dr_af = 1/ max_order / 2 / 1.6384，预留一定的空间来避免过渡带
    # Anti-aliasing
    # bf是间隔，bf<af 用af得到的采样点少于bf
    # 从[int(1 / dr_af) / 2] 截止频率 默认低通 int(1 / dr_bf)是vib_rsp_bf的采样率、
    # vib_rsp_bf的长度和vib_rsp_af的长度相同
    vib_rsp_af = butter_filter(vib_rsp_bf, [int(1 / dr_af) / 2], int(1 / dr_bf))

    # 角度域降采样，降至dr_af，根据最大关注阶次来定，重采样率高了浪费计算力，低了取不到用户关注的阶次
    # down resampling, interval dr_af = 1/ max_order / 2 / 1.6384
    # 生成角度域重采样的等角度间隔序列（目标采样率）
    rev_rsp_af = np.arange(target_rev[0] // dr_af,
                           target_rev[-1] // dr_af) * dr_af + dr_af
    # 基于rev_rsp_af插值得到振动信号的重采样序列（三次方比较符合振动信号的变化规律），fill_value指的是边缘多出来的部分继续插值得到
    vib_rsp = interp1d(rev_rsp, vib_rsp_af, kind='cubic',
                       fill_value='extrapolate', assume_sorted=True)(rev_rsp_af)
    # 基于rev_rsp_af插值得到振动信号对应的时间重采样序列（时间是线性变化的）
    vib_rsp_time = interp1d(target_rev, target_time, kind='linear',
                            fill_value='extrapolate', assume_sorted=True)(
        rev_rsp_af)
    # 根据之前记录的时间标志切割并获取目标段振动信号和对应的时间序列，但需要区别第一帧（不需要截取头部）
    if counter > 0:
        target_part = (start_time_flag < vib_rsp_time) & (
                vib_rsp_time <= end_time_flag)
        return vib_rsp[target_part], vib_rsp_time[target_part]
    else:
        target_part = (vib_rsp_time <= end_time_flag)
        return vib_rsp[target_part], vib_rsp_time[target_part]


def angular_resampling(rev, rev_time, target_vib, target_time, counter, points, dr_bf, dr_af):
    """
    功能：进行角度域重采样，得到等角度间隔的振动或声音信号，进而计算阶次谱（共享内存）
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
    function: angular resampling of vibration (time to angular)
    :param
    overall_rev(list): overall accumulative revolutions
    overall_rev_time(list): corresponding time list of overall accumulative revolutions
    overall_vib(list): overall accumulative vibration
    overall_time(list): corresponding time list of overall vibration
    counter(int): counter of angular resampling to index target data
    samps_per_chan(int): length of 1 frame data
    points(int): overlapped point to avoid boundary effect
    dr_bf(float): revolution interval between samples of resultant vibration signal(before filter)
    dr_af(float): revolution interval between samples of resultant vibration signal(after filter)
    :return
    vib_rsp(list): resampled vibration data
    vib_rsp_time(list): corresponding time of resampled vibration data
    """
    # 记录开始点结束点的时间，一遍最后切割出目标段
    start_time_flag = target_time[points]
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
    # vib_rsp_af = butter_filter(vib_rsp_bf, [int(1 / dr_af) / 2], int(1 / dr_bf))
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
    if counter > 0:
        target_part = (start_time_flag < vib_rsp_time) & (
                vib_rsp_time <= end_time_flag)
        return vib_rsp[target_part], vib_rsp_time[target_part]
    else:
        target_part = (vib_rsp_time <= end_time_flag)
        return vib_rsp[target_part], vib_rsp_time[target_part]



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
        x_value = [
            vibration_rsp_time[order_spectrum_calc_info['nstep'] * counter_or],
            vibration_rsp_time[
                order_spectrum_calc_info['nstep'] * counter_or +
                order_spectrum_calc_info['nfft'] - 1]]

        # 下面这句是记录中间点时间的代码（其实可以直接获取中间点的速度值）
        # x_value = vibration_rsp_time[order_spectrum_calc_info['nstep'] * counter_or + order_spectrum_calc_info['nfft']
        #                              // 2]

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


def order_spectrum_for_share(threed_os, twod_oc, counter_or, vibration_rsp,
                             vibration_rsp_time,
                             order_spectrum_calc_info, order_cut_calc_info,
                             sensor_index,
                             db_flag=0):
    order_resolution = order_spectrum_calc_info['orderResolution']
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
        x_value = vibration_rsp_time[order_spectrum_calc_info['nstep'] * counter_or +
                                     order_spectrum_calc_info['nfft'] // 2]

        # 提取指定圈数的振动或声音信号
        vib = vibration_rsp[order_spectrum_calc_info['nstep'] * counter_or:
                            order_spectrum_calc_info['nfft'] +
                            order_spectrum_calc_info['nstep'] * counter_or]

        # 对目标段信号加窗
        wvib = vib * order_spectrum_calc_info['win'][sensor_index]

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
        # threed_os["xValue"].append(x_value)
        threed_os["xValue"][counter_or] = x_value
        # 记录中间值用于计算二维平均阶次谱
        threed_os["tempXi2"] += fvib * fvib

        if db_flag:
            threed_os["zValue"][counter_or][:] = db_convertion(fvib,
                                                            order_spectrum_calc_info[
                                                                'refValue'][
                                                                sensor_index]).tolist()
        else:
            # threed_os["zValue"].append(list(fvib))
            threed_os["zValue"][counter_or][:] = fvib

        # 第二部分：计算二维阶次切片结果
        if twod_oc is not None:
            # 确保已创建过二维阶次切片结果集
            # calculate the 2D order cutting(consider that order also should include sub order)
            # 下面的关注阶次切片计算方法与一维阶次切片指标一致，不再做解释
            for i, order in enumerate(order_cut_calc_info['orderList']):

                index_array = np.around([[x // order_resolution + k for k in
                                          range(-(order_cut_calc_info[
                                                      'pointNum'] // 2),
                                                order_cut_calc_info[
                                                    'pointNum'] // 2 + 1)] for x in
                                         order]).astype('i')
                temp_result = np.sqrt(
                    np.sum(np.power(fvib[index_array], 2), axis=1))

                if db_flag:
                    twod_oc[i]['yValue'][counter_or] = \
                        db_convertion(rms(temp_result),
                                      order_spectrum_calc_info['refValue'][
                                          sensor_index])
                else:
                    twod_oc[i]['yValue'][counter_or] = rms(temp_result)
                twod_oc[i]['xValue'][counter_or] = x_value

        # 更新计数器
        counter_or = counter_or + 1
    return threed_os, twod_oc, counter_or


def order_spectrum_for_const(threed_os, twod_oc, counter_or, vibration_rsp,
                             order_spectrum_calc_info, order_cut_calc_info,
                             sensor_index,
                             vib_start_index,
                             sampleRate,
                             db_flag=0):
    order_resolution = order_spectrum_calc_info['orderResolution']
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
        # x_value = vibration_rsp_time[order_spectrum_calc_info['nstep'] * counter_or +
        #     order_spectrum_calc_info['nfft']// 2]

        # 提取指定圈数的振动或声音信号
        vib = vibration_rsp[order_spectrum_calc_info['nstep'] * counter_or:
                            order_spectrum_calc_info['nfft'] +
                            order_spectrum_calc_info['nstep'] * counter_or]

        # 得到的时间点是以测试开始时为0时刻点
        x_value = ((order_spectrum_calc_info['nstep'] * counter_or +
                    order_spectrum_calc_info['nfft'] // 2) + vib_start_index) / sampleRate

        # 对目标段信号加窗
        wvib = vib * order_spectrum_calc_info['win'][sensor_index]

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
        # threed_os["xValue"].append(x_value)
        threed_os["xValue"][counter_or] = x_value
        # 记录中间值用于计算二维平均阶次谱
        threed_os["tempXi2"] += fvib * fvib

        if db_flag:
            threed_os["zValue"][counter_or] = db_convertion(fvib,
                                                            order_spectrum_calc_info[
                                                                'refValue'][
                                                                sensor_index]).tolist()
        else:
            # threed_os["zValue"].append(list(fvib))
            threed_os["zValue"][counter_or][:] = fvib

        # 第二部分：计算二维阶次切片结果
        if twod_oc is not None:
            # 确保已创建过二维阶次切片结果集
            # calculate the 2D order cutting(consider that order also should include sub order)
            # 下面的关注阶次切片计算方法与一维阶次切片指标一致，不再做解释
            for i, order in enumerate(order_cut_calc_info['orderList']):

                index_array = np.around([[x // order_resolution + k for k in
                                          range(-(order_cut_calc_info[
                                                      'pointNum'] // 2),
                                                order_cut_calc_info[
                                                    'pointNum'] // 2 + 1)] for x in
                                         order]).astype('i')
                temp_result = np.sqrt(
                    np.sum(np.power(fvib[index_array], 2), axis=1))

                if db_flag:
                    twod_oc[i]['yValue'][counter_or] = \
                        db_convertion(rms(temp_result),
                                      order_spectrum_calc_info['refValue'][
                                          sensor_index])
                else:
                    twod_oc[i]['yValue'][counter_or] = rms(temp_result)
                twod_oc[i]['xValue'][counter_or] = x_value

        # 更新计数器
        counter_or = counter_or + 1
    return threed_os, twod_oc, counter_or


def order_spectrum_for_const_fluctuation(threed_os, twod_oc, counter_or, vibration_rsp,
                                         order_spectrum_calc_info, order_cut_calc_info,
                                         sensor_index,
                                         vib_start_index,
                                         sampleRate,
                                         speed,
                                         db_flag=0):
    """
    阶次谱计算，转速存在波动，每一帧转速不同
    Args:
        threed_os:
        twod_oc:
        counter_or:
        vibration_rsp:
        order_spectrum_calc_info:
        order_cut_calc_info:
        sensor_index:
        vib_start_index:
        sampleRate:
        db_flag:

    Returns:

    """

    # 计算阶次谱
    if len(vibration_rsp)==0:
        return threed_os, twod_oc, counter_or
    order_resolution = order_spectrum_calc_info['orderResolution']
    fvib = get_order_spectrum_frame(vibration_rsp, speed / 60 / sampleRate,
                                    order_spectrum_calc_info["revNum"],
                                    order_spectrum_calc_info["overlapRatio"],
                                    order_spectrum_calc_info["window"][sensor_index],
                                    order_spectrum_calc_info['normFactor'][
                                        sensor_index], order_spectrum_calc_info["maxOrder"], )
    if len(fvib)==0:
        return threed_os, twod_oc, counter_or
    x_value = (vib_start_index + len(vibration_rsp) / 2) / sampleRate
    threed_os["xValue"][counter_or] = x_value
    # 记录中间值用于计算二维平均阶次谱
    threed_os["tempXi2"] += fvib * fvib

    if db_flag:
        threed_os["zValue"][counter_or][:] = db_convertion(fvib,
                                                           order_spectrum_calc_info[
                                                               'refValue'][
                                                               sensor_index]).tolist()
    else:
        threed_os["zValue"][counter_or][:] = fvib

    # 第二部分：计算二维阶次切片结果
    if twod_oc is not None:
        # 确保已创建过二维阶次切片结果集
        # calculate the 2D order cutting(consider that order also should include sub order)
        # 下面的关注阶次切片计算方法与一维阶次切片指标一致，不再做解释
        for i, order in enumerate(order_cut_calc_info['orderList']):

            index_array = np.around([[x // order_resolution + k for k in
                                      range(-(order_cut_calc_info[
                                                  'pointNum'] // 2),
                                            order_cut_calc_info[
                                                'pointNum'] // 2 + 1)] for x in
                                     order]).astype('i')
            temp_result = np.sqrt(
                np.sum(np.power(fvib[index_array], 2), axis=1))

            if db_flag:
                twod_oc[i]['yValue'][counter_or] = \
                    db_convertion(rms(temp_result),
                                  order_spectrum_calc_info['refValue'][
                                      sensor_index])
            else:
                twod_oc[i]['yValue'][counter_or] = rms(temp_result)
            twod_oc[i]['xValue'][counter_or] = x_value

    # 更新计数器
    counter_or = counter_or + 1

    return threed_os, twod_oc, counter_or


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
    if twod_os:
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
            target_unit = task_info['units'][
                task_info["indicatorsUnitChanIndex"][sensor_index]]
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


def freq(fs, nfft, freq_max):
    """
    功能：提前计算好频谱的x轴，主要用于时频谱，避免每次都需要构建x轴
    输入：
    1. 采样率
    2. 傅里叶变换后的点数
    3. 最大关注频率
    返回： 频率序列
    """
    # to create the frequency list of spectrum
    f = np.fft.rfftfreq(nfft, 1 / fs)
    if freq_max:
        # pick out the value with f <= freqMax
        f = f[f <= freq_max]
    return f


def time_frequency_analysis(threed_tfm, data, counter_tfm, fs, frequency,
                            start_time, time_freq_map_calc_info,
                            task_info, sensor_index, db_flag=0):
    """
    功能：时频分析得到时频彩图
    输入：
    1. 实时更新的三维时频谱结果集，第一次为空结果集
    2. 累积数据，比如该测试段的振动或者声音信号
    3. 计数器
    4. 采样率
    5. 频谱x轴（频率序列）
    6. 测试段开始时间
    7. 时频分析参数信息
    8. 数据采集信息，主要提供db转换参考值
    9. 传感器索引，用于得到指定的额db转换参考值
    10. 是否进行db转换，默认为0（0表示否，1表示转换）
    返回：
    1. 时频彩图结果
    2. 计数器
    function: to calculate the time frequency analysis threed_tfm(3D time/speed frequency colormap)
    :param
    threed_tfm(dict): the data set to record the time frequency map threed_tfm, will be updated after one calculation, 
    init as empty
    data(list): overall vib or sound data
    counter_tfm(int): to count the number of frequency spectrum, mainly used to create the time flag for each frequency 
    spectrum, intial is zero
    fs(float): fs of the raw signal
    frequency(list): frequency of time-fre-colormap
    start_time(float): the start time of target test
    time_freq_map_calc_info(dict):
        nperseg(int): length of the segment to do fft
        freqMax(float): the max frequency need to show up
        noverlop(int): number of points to overlap between segments, if None, means noverlap=nperseg//2
        window(string): efine the window applied for vib data before fft, default is 'hanning'
        nfft(int): length of the FFT used, default as None, means nfft=nperseg
    :returns
        threed_tfm(dict): updated threed_tfm
        rest_vib(list): updated rest data
        counter_tfm(int): update the counter_tfm
    """
    while time_freq_map_calc_info['nperseg'] + time_freq_map_calc_info[
        'nstep'] * counter_tfm <= len(data):
        # 提取目标段信号
        vib = data[counter_tfm * time_freq_map_calc_info['nstep']:
                   counter_tfm * time_freq_map_calc_info['nstep'] +
                   time_freq_map_calc_info['nperseg']]
        # 对目标段信号加窗
        wvib = np.array(vib) * time_freq_map_calc_info['win']
        # 傅里叶变换得到频谱
        fvib = np.fft.rfft(wvib, time_freq_map_calc_info['nfft']) * \
               time_freq_map_calc_info['normFactor']
        # 根据最大关注频率截取频谱
        if time_freq_map_calc_info['freqMax']:
            fvib = fvib[:frequency.shape[0]]
        # 更新三维时频彩图结果
        threed_tfm['yValue'].append((time_freq_map_calc_info['nperseg'] +
                                     time_freq_map_calc_info['nstep'] *
                                     counter_tfm) / fs + start_time)
        if db_flag:
            threed_tfm['zValue'].append(db_convertion(fvib,
                                                      task_info['refValue'][
                                                          sensor_index]).tolist())
        else:
            threed_tfm['zValue'].append(np.abs(fvib).tolist())
        # 更新计数器
        counter_tfm += 1

    return threed_tfm, counter_tfm


def frequency_spectrum(vib, fs, task_info, sensor_index, freq_max=None,
                       db_flag=0):
    """
    功能：计算频谱
    输入：
    1. 振动或声音信号
    2. 采样率
    3. 数据采集参数信息，主要需要用到其中的单位信息
    4. 传感器索引，用于获取指定单位信息
    5. 最大关注频率，默认为空
    6. 是否进行db转换，默认为0（0表示否，1表示转换）
    返回：频谱
    function: calculate the frequency spectrum
    :param
    vib(list): vibration data
    fs(float): sample rate of raw vibration data
    freqMax(float): max frequency to show up
    :return
    result(dict): with frequency and spectrum amplitude
    """
    fvib = np.fft.rfft(vib) / len(vib) * 1.414
    f = np.fft.rfftfreq(len(vib), 1 / fs)
    if freq_max:
        f = f[f <= freq_max]
        fvib = fvib[:f.shape[0]]
    if db_flag:
        target_value = db_convertion(fvib, task_info['refValue'][
            sensor_index]).tolist()
        target_unit = 'dB'
    else:
        target_value = np.abs(fvib).tolist()
        target_unit = task_info['units'][
            task_info["indicatorsUnitChanIndex"][sensor_index]]
    twod_freq_spec_result = {
        'xName': 'Freq',
        'xUnit': 'Hz',
        'xValue': f.tolist(),
        'yName': 'Frequency Spectrum',
        'yUnit': target_unit,
        'yValue': target_value
    }
    return twod_freq_spec_result


def real_time_ssa_calc(ssa_result, data, trigger_location, base_sample,
                       ssa_calc_info):
    """
    功能：ssa分析
    输入:
    1. 实时更新的ssa结果
    2. 时域振动或声音信号
    3. 脉冲触发位置信息（即记录了有多少个脉冲）
    4. 根据该测试段设置信息里的最小转速得到的一圈点数
    5. ssa计算参数信息
    返回：实时更新的ssa结果
    function: Time-synchronous signal average/analysis
    :param
    ssa_result(dict): tsa result of all shaft(set), include the counters for each shaft
    data(list): cum vib or sound signal data of target test
    base_sample(int): resample num
    ssa_calc_info(dict): the parameters used to calculate the tsa, include the info below:
        gearRatio(list): the teeth number of each shaft(1st is the output shaft)
        gearName(list): the name for each shaft
        revNum(int): the revolution number of output shaft
        ppr(int): ppr of output shaft
    :return
    ssa_result(list): updated ssa data
    """

    def get_int_float(number):
        # 获取整数部分和小数部分
        float_value, int_value = math.modf(number)
        return int(int_value), float_value

    for i in range(0, len(ssa_calc_info['pprNum'])):
        # 确定需要扩展成的点数
        sample_num = round(base_sample * ssa_calc_info['factors'][i])
        while ssa_calc_info['pprNum'][i] * (
                ssa_result['counters'][i] + 1) + 1 <= len(trigger_location):
            # 计算开始点索引
            ppr_int_value, ppr_float_value = get_int_float(
                ssa_calc_info['pprNum'][i] * ssa_result['counters'][i])
            float_part_num = round(
                (trigger_location[ppr_int_value + 1] - trigger_location[
                    ppr_int_value]) * ppr_float_value)
            start_index = trigger_location[ppr_int_value] + float_part_num
            # 计算结束点索引
            ppr_int_value, ppr_float_value = get_int_float(
                ssa_calc_info['pprNum'][i] * (ssa_result['counters'][i] + 1))
            float_part_num = round(
                (trigger_location[ppr_int_value + 1] - trigger_location[
                    ppr_int_value]) * ppr_float_value)
            end_index = trigger_location[ppr_int_value] + float_part_num
            # 提取振动信号并叠加
            ssa_result['resultData'][i]['yValue'] += resample(
                data[start_index: end_index], sample_num)
            # 更新计数器
            ssa_result['counters'][i] += 1
    return ssa_result


def ssa_update(ssa_result):
    """
    功能：基于实时更新的ssa结果生成最终结果（也就是基于之间累积的结果做平均）
    输入：ssa结果
    返回：整理后的ssa结果
    """
    for i in range(len(ssa_result['resultData'])):
        # Y值求平均
        ssa_result['resultData'][i]['yValue'] = list(
            ssa_result['resultData'][i]['yValue'] / ssa_result['counters'][i])
        # 生成X值
        ssa_result['resultData'][i]['xValue'] = list(
            np.linspace(0, 360, len(ssa_result['resultData'][i]['yValue']),
                        endpoint=False))
    # 返回结果数据（计数器可以省去）
    return ssa_result['resultData']








def twod_stat_factor_for_const_fluctuation(twodsf, tempsf, rev, rev_index, vibration_rsp,
                                           twod_sf_counter, vib_start_index, sampleRate,
                                           stat_factor_calc_info,
                                           average_flag=1):
    """
    功能：基于角度域重采样的振动数据按圈计算统计学指标,恒速电机版
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
        return twodsf, tempsf
    if rev_index==0:
        return twodsf,tempsf
    for i, temp_rev_num in enumerate(stat_factor_calc_info['revNums']):
        temp_counter = tempsf[stat_factor_calc_info['gearName'][i]]['counter']
        temp_gear_name = stat_factor_calc_info['gearName'][i]
        while temp_rev_num * (temp_counter + 1) < rev[rev_index - 1]:
            temp_last_rev_index = tempsf[stat_factor_calc_info['gearName'][i]]["lastRevIndex"]
            # 可以计算下一次
            temp_vib_right_index = np.searchsorted(rev[temp_last_rev_index:rev_index],
                                              temp_rev_num * (temp_counter + 1), side="right")
            vib=vibration_rsp[temp_last_rev_index:temp_last_rev_index+temp_vib_right_index]

            # qDAQ_logger.debug("rev[temp_last_rev_index+temp_vib_right_index]:{}".format(rev[temp_last_rev_index+temp_vib_right_index]))
            # qDAQ_logger.debug("rev[temp_last_rev_index+temp_vib_right_index]:{}".format(temp_last_rev_index+temp_vib_right_index))
            x_value=(vib_start_index+(temp_last_rev_index+temp_last_rev_index+temp_vib_right_index)/2)/sampleRate
            # qDAQ_logger.debug("x_value:{}".format(x_value))
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
                    rms_value, xi2 = twodtd_rms(vib, len(vib))
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'yValue'][temp_counter] = rms_value
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'xValue'][temp_counter] = x_value
                elif indicator == 'Crest':
                    try:
                        crest_value, max_value, xi2 = twodtd_crest(vib, len(vib))
                    except Exception:
                        time.sleep(0.1)
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'yValue'][temp_counter] = crest_value
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'xValue'][temp_counter] = x_value
                elif indicator == 'Kurtosis':
                    kur_value, mean_value, xi4, xi3, xi2 = twodtd_kurtosis(vib,
                                                                           len(vib))
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'yValue'][temp_counter] = kur_value
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'xValue'][temp_counter] = x_value
                elif indicator == 'Skewness':
                    skew_value, mean_value, xi3, xi2 = twodtd_skewness(vib,
                                                                       len(vib))
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'yValue'][temp_counter] = skew_value
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'xValue'][temp_counter] = x_value
                elif indicator == "SPL":
                    spl_value, xi2 = twod_spl(vib, len(vib))
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'yValue'][temp_counter] = spl_value
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'xValue'][temp_counter] = x_value
                elif indicator == "SPL(A)":
                    # 生成A计权滤波器
                    B, A = A_weighting(stat_factor_calc_info['sampleRate'])
                    if len(vib) > 500:
                        a_weighting_calc_vib = lfilter(B, A, vib)[500:]
                    else:
                        a_weighting_calc_vib = lfilter(B, A, vib)
                    spl_value, xi2_A = twod_spl(a_weighting_calc_vib, len(vib))
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'yValue'][temp_counter] = spl_value
                    twodsf[i * stat_factor_calc_info['indicatorNum'] + j][
                        'xValue'][temp_counter] = x_value
                elif indicator=="Speed":
                    # 恒速电机会有Speed指标
                    pass
            if not average_flag:
                # 记录统计学指标所需要的的中间值
                if {'RMS', 'Crest', 'Kurtosis', 'Skewness',
                    'SPL'} & indicatorSet:
                    tempsf[temp_gear_name]['xi2'][temp_counter] = xi2
                if {'Crest'} & indicatorSet:
                    tempsf[temp_gear_name]['xmax'][temp_counter] = max_value
                if {'Kurtosis', 'Skewness'} & indicatorSet:
                    tempsf[temp_gear_name]['xi3'][temp_counter] = xi3
                    tempsf[temp_gear_name]['xmean'][temp_counter] = mean_value
                if {'Kurtosis'} & indicatorSet:
                    tempsf[temp_gear_name]['xi4'][temp_counter] = xi4
                if {'SPL(A)'} & indicatorSet:
                    # 确认需要记录A计权平方和值xi2_A，而且需要避免重复记录
                    tempsf[temp_gear_name]['xi2_A'][temp_counter] = xi2_A

            temp_counter += 1
            tempsf[temp_gear_name]['counter'] += 1
            twod_sf_counter[temp_gear_name] = temp_counter
            # 更新最后一次计算的temp_vib_right_index
            tempsf[stat_factor_calc_info['gearName'][i]]["lastRevIndex"]=temp_last_rev_index+temp_vib_right_index
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
                            temp_x = (np.interp(
                                np.mean(twod_data[i]['xValue'], axis=1),
                                curve_x,
                                curve_y) * speed_ratio).tolist()
                        else:
                            temp_x = list()
                    else:
                        if len(twod_data[i]['xValue']) > 0:
                            # 避免空数据
                            temp_x = np.interp(
                                np.mean(twod_data[i]['xValue'], axis=1),
                                curve_x, curve_y).tolist()
                        else:
                            temp_x = list()
                twod_data[i]['xValue'] = temp_x
        else:
            # 二维时间域或者二维阶次切片
            if speed_ratio != 1:
                if len(twod_data[0]['xValue']) > 0:
                    # 避免空数据
                    temp_x = (np.interp(np.mean(twod_data[0]['xValue'], axis=1),
                                        curve_x,
                                        curve_y) * speed_ratio).tolist()
                else:
                    temp_x = list()
            else:
                if len(twod_data[0]['xValue']) > 0:
                    # 避免空数据
                    temp_x = np.interp(np.mean(twod_data[0]['xValue'], axis=1),
                                       curve_x, curve_y).tolist()
                else:
                    temp_x = list()
            for i in range(len(twod_data)):
                twod_data[i]['xValue'] = temp_x
    else:
        # 恒速段
        if indicator_num:
            for i in range(len(twod_data)):
                if i % indicator_num == 0:
                    if len(twod_data[i]['xValue']) > 0:
                        temp_x = np.mean(twod_data[i]['xValue'],
                                         axis=1).tolist()
                    else:
                        temp_x = list()
                twod_data[i]['xValue'] = temp_x
                twod_data[i]['xName'] = 'Time'
                twod_data[i]['xUnit'] = 's'
        else:
            if len(twod_data[0]['xValue']) > 0:
                temp_x = np.mean(twod_data[0]['xValue'], axis=1).tolist()
            else:
                temp_x = list()
            for i in range(len(twod_data)):
                twod_data[i]['xValue'] = temp_x
                twod_data[i]['xName'] = 'Time'
                twod_data[i]['xUnit'] = 's'
    if threed_os:
        # 需要改变三维结果数据的X轴（与阶次切片的X轴相同）
        threed_os['xValue'] = temp_x
        return twod_data, threed_os
    else:
        # 不用返回三维结果
        return twod_data


def convert_time_speed_for_share(twod_data, curve_x, curve_y, speed_pattern,
                                 speed_ratio,
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
                            temp_x = (np.interp(twod_data[i]['xValue'], curve_x,
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
        else:
            if len(twod_data):
                # 二维时间域或者二维阶次切片
                if speed_ratio != 1:
                    if len(twod_data[0]['xValue']) > 0:
                        # 避免空数据
                        temp_x = (np.interp(twod_data[0]['xValue'], curve_x,
                                            curve_y) * speed_ratio).tolist()
                    else:
                        temp_x = list()
                else:
                    if len(twod_data[0]['xValue']) > 0:
                        # 避免空数据
                        temp_x = np.interp(twod_data[0]['xValue'], curve_x, curve_y).tolist()
                    else:
                        temp_x = list()
                for i in range(len(twod_data)):
                    twod_data[i]['xValue'] = temp_x
    else:
        # 恒速段
        if indicator_num:
            for i in range(len(twod_data)):
                if i % indicator_num == 0:
                    if len(twod_data[i]['xValue']) > 0:
                        temp_x = twod_data[i]['xValue'].tolist()
                    else:
                        temp_x = list()
                twod_data[i]['xValue'] = temp_x
                twod_data[i]['xName'] = 'Time'
                twod_data[i]['xUnit'] = 's'
        else:
            if len(twod_data):
                if len(twod_data[0]['xValue']) > 0:
                    temp_x = twod_data[0]['xValue'].tolist()
                else:
                    temp_x = list()
                for i in range(len(twod_data)):
                    twod_data[i]['xValue'] = temp_x
                    twod_data[i]['xName'] = 'Time'
                    twod_data[i]['xUnit'] = 's'
    if threed_os:
        # 需要改变三维结果数据的X轴（与阶次切片的X轴相同）
        threed_os['xValue'] = temp_x
        return twod_data, threed_os
    else:
        # 不用返回三维结果
        return twod_data


def convert_time_speed_for_const(twod_data, threed_os=None, indicator_num=None):
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

    if indicator_num:
        for i in range(len(twod_data)):
            if i % indicator_num == 0:
                if len(twod_data[i]['xValue']) > 0:
                    temp_x = twod_data[i]['xValue'].tolist()
                else:
                    temp_x = list()
            twod_data[i]['xValue'] = temp_x
            twod_data[i]['xName'] = 'Time'
            twod_data[i]['xUnit'] = 's'
    else:
        if len(twod_data):
            if len(twod_data[0]['xValue']) > 0:
                temp_x = twod_data[0]['xValue'].tolist()
            else:
                temp_x = list()
            for i in range(len(twod_data)):
                twod_data[i]['xValue'] = temp_x
                twod_data[i]['xName'] = 'Time'
                twod_data[i]['xUnit'] = 's'
    if threed_os:
        # 需要改变三维结果数据的X轴（与阶次切片的X轴相同）
        threed_os['xValue'] = temp_x
        return twod_data, threed_os
    else:
        # 不用返回三维结果
        return twod_data


def direct_convert_time_speed(x, curve_x, curve_y, speed_ratio):
    """
    功能：根据转速曲线的x和y得到指定x的y值
    输入：
    1. 指定的x序列（序列内直接就是float值）
    2. 转速曲线的x轴
    3. 转速曲线的y轴
    返回：指定x序列对应的y序列
    """
    if speed_ratio != 1:
        y = (np.interp(x, curve_x, curve_y) * speed_ratio).tolist()
    else:
        y = np.interp(x, curve_x, curve_y).tolist()
    return y


def convert_time_speed_npinterp(test_result, threed_os, speed_curve,
                                sensor_index, test_name_index, speed_pattern):
    """
    功能：时间速度转换
    输入：
    1. 结果数据
    2. 三维阶次谱
    3. 转速曲线
    4. 传感器索引
    5. 测试段索引
    6. 转速类型（升速，降速，恒速）
    返回：进行速度和时间转换的
    1. 结果数据
    2. 三维阶次谱
    function: to convert the X-axis in testResult from time to speed or from speed to time
    :param
    test_result(dict): the overall result of the whole test
    threed_os(dict): 3D order time colormap
    speed_curve(dict): the overall speed curve
    sensor_index(int): index of sensor id
    test_name_index(int): index of test
    speed_pattern(int): flag of speed
    indicator_list(int): indicator list of the stat indicators by shaft
    :return
    test_result(dict): the overall result with another X-axis
    threed_os(dict): 3D order speed colormap
    """
    dataSection = test_result['resultData'][sensor_index]['dataSection'][
        test_name_index]
    if speed_pattern > 1:
        # 变速段
        if dataSection['twodTD']:
            # 直接根据目标测试段转速曲线插值得到转速值，这里记录的开始点和结束点，若只记录了中间点也不影响，mean后仍然是那个数
            temp_td = np.interp(
                [np.mean(x) for x in dataSection['twodTD'][0]['xValue']],
                speed_curve['time'],
                speed_curve['speed']).tolist()
            for twodtd in dataSection['twodTD']:
                twodtd['xValue'] = temp_td
        if dataSection['twodOC']:
            # 直接根据目标测试段转速曲线插值得到转速值（twodTD和twodOC对应时间不一样，所以要分别获取），该结果记录了开始点和结束点
            temp_oc = np.interp(
                [np.mean(x) for x in dataSection['twodOC'][0]['xValue']],
                speed_curve['time'],
                speed_curve['speed']).tolist()
            threed_os['xValue'] = temp_oc
            for twodoc in dataSection['twodOC']:
                twodoc['xValue'] = temp_oc
    else:
        # 恒速段
        if dataSection['twodTD']:
            # 记录了开始点和结束点，若只记录了中间点则mean后不发生变化
            temp_td = [np.mean(x) for x in dataSection['twodTD'][0]['xValue']]
            for twodtd in dataSection['twodTD']:
                twodtd['xValue'] = temp_td
                twodtd['xName'] = 'Time'
                twodtd['xUnit'] = 's'
        if dataSection['twodOC']:
            temp_oc = [np.mean(x) for x in dataSection['twodOC'][0]['xValue']]
            threed_os['xValue'] = temp_oc
            for twodoc in dataSection['twodOC']:
                twodoc['xValue'] = temp_oc
                twodoc['xName'] = 'Time'
                twodoc['xUnit'] = 's'
    return test_result, threed_os


def get_order_spectrum_frame(vib, d_revolution, n_revolution, overlap, wtype, winCorrectFlag,
                             max_order=200,
                             normalize=False):
    '''
    draw order spectrum from vibration data for constant speed
    ***************************************************
    # 不够算一次则补零
    parameters
             vib: vibration data
    d_revolution: revolutions between two points
    n_revolution: revolutions to do fft
       max_order: required max order
       normalize: normalize rms

    returns
          o[roi]: x-axis for order spectrum
           specy: y-axis for order spectrum
    '''

    # nfft
    n = int(n_revolution / d_revolution)

    # x-axis for order spectrum
    # o = np.fft.rfftfreq(n, d=d_revolution)
    # roi = o < max_order
    # o=o[roi]

    step = int(n * (1 - overlap))

    # result varibale
    yy = []

    if len(vib) <= n:
        count = 1
    else:
        count = (len(vib) - n) // step + 1

    for i in range(count):
        frame = vib[i * step:i * step + n]
        if len(frame)<n:

            return []
        if wtype == "hanning":
            fy = abs(np.fft.rfft(frame * np.hanning(len(frame)), n)) /len(frame)*2*0.707*np.sqrt(len(frame))/np.sqrt(n)
            if winCorrectFlag:
                fy *= 1.633
        elif wtype == "hamming":
            fy = abs(np.fft.rfft(frame * np.hamming(len(frame)), n)) / len(frame)*2*0.707*np.sqrt(len(frame))/np.sqrt(n)
            if winCorrectFlag:
                fy *= 1.586
        elif wtype == 'kaiser':
            fy = abs(np.fft.rfft(frame * np.kaiser(len(frame), 9), n)) / len(frame)*2*0.707*np.sqrt(len(frame))/np.sqrt(n)
            if winCorrectFlag:
                fy *= 1.81
        elif wtype == 'blackman':
            fy = abs(np.fft.rfft(frame * np.blackman(len(frame)), n)) / len(frame)*2*0.707*np.sqrt(len(frame))/np.sqrt(n)
            if winCorrectFlag:
                fy *= 1.812
        elif wtype == 'bartlett':
            fy = abs(np.fft.rfft(frame * np.bartlett(len(frame)), n)) / len(frame)*2*0.707*np.sqrt(len(frame))/np.sqrt(n)
            if winCorrectFlag:
                fy *= 1.732
        elif wtype == 'flattop':
            fy = abs(np.fft.rfft(frame * signal.flattop(len(frame)), n)) / len(frame)*2*0.707*np.sqrt(len(frame))/np.sqrt(n)
            if winCorrectFlag:
                fy *= 1.069
        else:
            # 加矩形窗
            fy = abs(np.fft.rfft(frame * np.ones(len(frame)), n)) / len(frame)*2*0.707*np.sqrt(len(frame))/np.sqrt(n)

        # 直流分量修正
        fy[0] /= 2
        yy.append(fy[:max_order * n_revolution])

    if normalize == True:
        specy = np.mean(yy, axis=0) / max(np.mean(yy, axis=0))
    else:
        specy = np.mean(yy, axis=0)

    return specy

