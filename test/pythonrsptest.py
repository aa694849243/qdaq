import logging
import traceback

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

from nptdms import TdmsFile
from scipy import stats, fftpack, signal
from scipy.interpolate import interp1d


def speed_calibration_with_orderlist(vib, min_speed, max_speed, order, fs):
    '''
    speed calibration for constant speed
    ************************************************
    parameters
           vib: vibration data
     speed_ref: speed reference
         order: main order for machine
            fs: sampling rate
    return
    speed_cali: speed after calibration
    '''

    frame = vib
    if len(vib) > 8192:
        frame = vib[:8192]
    samplePerChan = len(vib)
    ref_order_array = np.sort(order)
    frame_fft_abs = np.abs(np.fft.rfft(frame)) / len(frame) * 2
    frame_freq = np.fft.rfftfreq(len(frame), d=1 / fs)

    # 多个阶次的话为最大阶次的频率寻找范围，单个阶次时为该阶次的寻找范围
    max_freq = max_speed / 60 * ref_order_array[-1]
    min_freq = min_speed / 60 * ref_order_array[-1]

    if len(ref_order_array) == 1:
        idx = (frame_freq >= min_freq) & (frame_freq <= max_freq)
        # find target frequency
        target = np.argmax(np.abs(frame_fft_abs[idx]))
        # target_min = np.argmin(vib_fft[idx])
        speed_cali = frame_freq[idx][target] / ref_order_array[-1] * 60
        return speed_cali

    fft_abs_cut = frame_fft_abs[frame_freq <= max_freq]
    freq_cut = frame_freq[frame_freq <= max_freq]
    peaks_index, _ = signal.find_peaks(fft_abs_cut)

    max_n = get_nlargest(fft_abs_cut[peaks_index], 100)
    max_n = np.sort(max_n)
    max_n_freq = freq_cut[peaks_index][max_n]
    max_n_vib = fft_abs_cut[peaks_index][max_n]

    peak_to_compare_list = list()
    last_peak_to_compare_index = 0
    min_diff_frequency = 0.5 * min_speed / 60

    for i in range(0, len(max_n_freq)):
        if max_n_freq[i] < min_diff_frequency:
            continue
        if max_n_freq[i] - max_n_freq[last_peak_to_compare_index] < min_diff_frequency:
            if max_n_vib[i] > max_n_vib[last_peak_to_compare_index]:
                last_peak_to_compare_index = i
        else:
            if max_n_freq[last_peak_to_compare_index] > min_diff_frequency:
                peak_to_compare_list.append(last_peak_to_compare_index)

            last_peak_to_compare_index = i

    peak_to_compare_list.append(last_peak_to_compare_index)
    peak_to_compare_array = np.array(peak_to_compare_list)
    freq_to_compare = max_n_freq[peak_to_compare_array]
    al_to_compare = max_n_vib[peak_to_compare_array]

    freq_revolution = fs / len(frame)
    # 每一个点作为第一个参考阶次的评分
    score = list()
    vib_to_compare = max_n_vib[peak_to_compare_array]
    arg_vib_max = np.argsort(vib_to_compare)
    rank = np.zeros(len(vib_to_compare))
    rank[arg_vib_max] = range(1, len(vib_to_compare) + 1)
    score_without_al = list()
    score_al = list()
    for i in range(len(freq_to_compare) - 1, -1, -1):
        if freq_to_compare[i] < min_freq:
            break
        speed_temp = freq_to_compare[i] / ref_order_array[-1]
        freq_to_find = speed_temp * ref_order_array
        argmin_index = list(map(find_argmin_in(freq_to_compare), freq_to_find))
        # if i==7:
        #     time.sleep(0.1)

        # if len(np.where(np.abs(freq_to_find - freq_to_compare[argmin_index])>3*freq_revolution)[0])!=0:
        #     score.append(0)
        #     score_without_al.append(0)
        #     score_al.append(0)
        #     continue
        # score.append(np.sum(1 - np.abs(freq_to_find - freq_to_compare[argmin_index]) / freq_to_find) + np.sum(freq_revolution  * rank[argmin_index]))
        # score_without_al.append(np.sum(1 - np.abs(freq_to_find - freq_to_compare[argmin_index]) / freq_to_find))
        # score_al.append(np.sum(freq_revolution  * rank[argmin_index]))

        freq_right_index = \
            np.where(np.abs(freq_to_find - freq_to_compare[argmin_index]) <= 3 * freq_revolution)[0]
        score.append(np.sum(al_to_compare[argmin_index][freq_right_index]))

        # score.append(np.sum(1-np.abs(speed_temp_array-freq_to_compare[l])/speed_temp_array))
        # a=freq_to_compare-speed_temp_array.reshape((len(ref_order_array)-1,1))
        # min_index=np.argmin(np.abs(a), axis=1)
        # if np.max(a[min])>min_diff_frequency:
        #     continue
        # else:
        #     rpm_list.append(speed_temp*60)
        #     break
    index_first_order = np.argmax(score)
    speed_cali = freq_to_compare[len(freq_to_compare) - 1 - index_first_order] / ref_order_array[-1] * 60
    return speed_cali


def find_argmin_in(a):
    def find_argmin(y):
        return np.argmin(np.abs(a - y))

    return find_argmin


def get_nlargest(array, n=3):
    '''
    get n largest number in array
    ********************************************
    parameters
         array: array to be processed
    return
           arg: index of n largest numbers
    '''
    arr = np.array(array)
    arg = arr.argsort()[-n:]
    return arg


def read_raw_data(filename, channel_names, file_type='tdms'):
    """
    功能：读取原始数据（全部通道）
    输入：
    1. 原始数据文件名（全路径）
    2. 通道名列表
    3. 原始文件类型
    返回：原始数据（按通道名存储）
    """
    raw_data = dict()
    if file_type == 'tdms':
        with TdmsFile.open(filename) as tdms_file:
            for channel_name in channel_names:
                raw_data[channel_name] = list(tdms_file['AIData'][channel_name][:])
    else:
        with h5py.File(filename, 'r') as h5pyFile:
            data_group = h5pyFile['AIData']
            for channel_name in channel_names:
                raw_data[channel_name] = list(
                    # np.array(data_group[channel_name][11000000:19000000], dtype='float'))
                    np.array(data_group[channel_name], dtype='float'))
    return raw_data


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
    if len(vibration_rsp) == 0:
        return threed_os, twod_oc, counter_or
    order_resolution = order_spectrum_calc_info['orderResolution']
    fvib = get_order_spectrum_frame(vibration_rsp, speed / 60 / sampleRate,
                                    order_spectrum_calc_info["revNum"],
                                    order_spectrum_calc_info["overlapRatio"],
                                    order_spectrum_calc_info["window"][sensor_index],
                                    order_spectrum_calc_info['winCorrectFlag'][sensor_index],
                                    order_spectrum_calc_info["maxOrder"], )
    if len(fvib) == 0:
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
        if len(frame) < n:
            logging.warning("参与傅式变换的点不足32圈")
            return []
        if wtype == "hanning":
            fy = abs(np.fft.rfft(frame * np.hanning(len(frame)), n)) / len(frame) * 2 * 0.707 * np.sqrt(
                len(frame)) / np.sqrt(n)
            if winCorrectFlag:
                fy *= 1.633
        elif wtype == "hamming":
            fy = abs(np.fft.rfft(frame * np.hamming(len(frame)), n)) / len(frame) * 2 * 0.707 * np.sqrt(
                len(frame)) / np.sqrt(n)
            if winCorrectFlag:
                fy *= 1.586
        elif wtype == 'kaiser':
            fy = abs(np.fft.rfft(frame * np.kaiser(len(frame), 9), n)) / len(
                frame) * 2 * 0.707 * np.sqrt(len(frame)) / np.sqrt(n)
            if winCorrectFlag:
                fy *= 1.81
        elif wtype == 'blackman':
            fy = abs(np.fft.rfft(frame * np.blackman(len(frame)), n)) / len(frame) * 2 * 0.707 * np.sqrt(
                len(frame)) / np.sqrt(n)
            if winCorrectFlag:
                fy *= 1.812
        elif wtype == 'bartlett':
            fy = abs(np.fft.rfft(frame * np.bartlett(len(frame)), n)) / len(frame) * 2 * 0.707 * np.sqrt(
                len(frame)) / np.sqrt(n)
            if winCorrectFlag:
                fy *= 1.732
        elif wtype == 'flattop':
            fy = abs(np.fft.rfft(frame * signal.flattop(len(frame)), n)) / len(
                frame) * 2 * 0.707 * np.sqrt(len(frame)) / np.sqrt(n)
            if winCorrectFlag:
                fy *= 1.069
        else:
            # 加矩形窗
            fy = abs(np.fft.rfft(frame * np.ones(len(frame)), n)) / len(frame) * 2 * 0.707 * np.sqrt(
                len(frame)) / np.sqrt(n)

        # 直流分量修正
        fy[0] /= 2
        yy.append(fy[:max_order * n_revolution])

    if normalize == True:
        specy = np.mean(yy, axis=0) / max(np.mean(yy, axis=0))
    else:
        specy = np.mean(yy, axis=0)

    return specy


def db_convertion(data, ref_value):
    """
    功能：转换为dB值
    输入：
    1. 结果数据，可以是一维或者二维数据
    2. 参考值，用于db的计算
    返回：转换后的结果
    """
    return 20 * np.log10(np.abs(data) / ref_value)


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
            'yUnit': "g",
            'yValue': np.sqrt(threed_os['tempXi2'] / len(threed_os['xValue'])),
            "indicatorDiagnostic": indicator_diagnostic
        })
    return twodos_result


def create_empty_threedos_for_share(order_spectrum_calc_info, os_max_len,
                                    db_flag=0, indicator_diagnostic=-1):
    """
    功能：初始化三维阶次谱结果，用于实时阶次谱结果记录更新
    输入：
    1. 阶次谱计算信息
    2. 数据采集信息，用于提供单位
    3. 传感器索引，用于获取指定单位
    4. 单位是否为dB，默认不是（0表示不是，1表示是）
    5. 初始化的评判结果，默认是为-1（表示缺失）
    返回：不包含具体数值的三维阶次谱结果
    function: create the empty 3D order spectrum use it in following indicators calculation
    :param
    order_spectrum_calc_info(dict): the parameters for order spectrum calculation, include the xAxis and xUnit
    task_info(dict): update the unit of indicators
    :return:
    threedos_result(dict): a list include all 3D order spectrum, used for real time update
    """

    threedos_result = {
        'xName': "order",
        'xUnit': "",
        'xValue': np.zeros(os_max_len),
        'yName': 'Order',
        'yUnit': '',
        'yValue': np.arange(order_spectrum_calc_info["maxOrder"]*32)*0.03125,
        'zName': 'Order Spectrum',
        'zUnit': "pa",
        'zValue': np.zeros(shape=(os_max_len, int(order_spectrum_calc_info["maxOrder"] / 0.03125))),
        "indicatorDiagnostic": indicator_diagnostic,
        'tempXi2': np.zeros(int(order_spectrum_calc_info["maxOrder"] / 0.03125))
    }
    # 其中tempXi2是用于二维阶次谱计算
    return threedos_result


def create_empty_twodoc_for_share(order_cut_calc_info, os_max_len, db_flag=0,
                                  indicator_diagnostic=-1):
    """
    功能：初始化二维阶次切片结果，用户实时计算中进行更新（共享内存）
    输入：
    1. 阶次切片计算参数，主要包含关注阶次和阶次宽度
    2. 数据采集信息，主要提供单位
    3. 传感器索引，用于获取指定的单位
    4. 最多计算多少次阶次切片
    5. 单位是否为dB， 默认不是（0表示不是，1表示是）
    返回：不包含具体数值的二维阶次切片结果
    function: create the empty twodoc_result and use it in following indicators calculation
    :param
    order_cut_calc_info(dict): the parameters for order cut calculation, include the cutted order list,
    e.g. ['1st','2nd']
    :return:
    twodoc_result(list): a list include all 2D order cutting indicator twodoc_result
    """
    twodoc_result = list()
    if db_flag:
        target_unit = 'dB'
    else:
        target_unit = "pa"
    for indicator in order_cut_calc_info['orderName']:
        twodoc_result.append({
            'xName': "order",
            'xUnit': "",
            'xValue': np.zeros(os_max_len),
            'yName': indicator,
            # follow the unit of NI Unit, g or Pa, or dB
            'yUnit': target_unit,
            'yValue': [None] * os_max_len,
            "indicatorDiagnostic": indicator_diagnostic
        })
    return twodoc_result


if __name__ == "__main__":
    abs_filename = r"E:\RawData\xiaodianji\21082609"
    vib_ndarray = \
        read_raw_data(os.path.join(abs_filename, "65NG1-1_210826093439.tdms"), ["Mic1"], file_type="tdms")["Mic1"]
    startSpeed = 60000
    endSpeed = 70000
    sampleRate = 102400
    sampsPerChan = int(32 / (startSpeed / 60) * sampleRate)

    vib_start_index = 0
    counter_or = 0
    order = [13]
    rpm = list()
    rpml = list()
    maxT = 20
    rev = np.zeros(int(maxT * sampleRate) + 1)
    rev_index = 0

    # 阶次谱计算
    orderSpectrumCalcInfo = dict()
    orderSpectrumCalcInfo["orderResolution"] = 0.03125
    orderSpectrumCalcInfo["revNum"] = 32
    orderSpectrumCalcInfo["overlapRatio"] = 0.75
    orderSpectrumCalcInfo["window"] = ["kaiser", "kaiser"]
    orderSpectrumCalcInfo["winCorrectFlag"] = [0, 0]
    orderSpectrumCalcInfo["overlapRatio"] = 0.75
    orderSpectrumCalcInfo["maxOrder"] = 28
    maxOrder=orderSpectrumCalcInfo["maxOrder"]
    orderCutCalcInfo = dict()
    orderCutCalcInfo["orderList"] = [[13]]
    orderCutCalcInfo["orderName"] = ["13order"]
    orderCutCalcInfo["pointNum"] = 17
    osMaxLen = int(maxT * sampleRate // sampsPerChan) + 2
    threed_os = create_empty_threedos_for_share(orderSpectrumCalcInfo,
                                                osMaxLen,
                                                db_flag=0,
                                                indicator_diagnostic=-1)
    twod_oc = create_empty_twodoc_for_share(orderCutCalcInfo,
                                            osMaxLen,
                                            db_flag=0,
                                            indicator_diagnostic=-1)

    while vib_start_index + (counter_or + 1) * sampsPerChan < len(vib_ndarray):
        speed = speed_calibration_with_orderlist(vib_ndarray[vib_start_index + counter_or * sampsPerChan:
                                                             vib_start_index + (counter_or + 1) * sampsPerChan],
                                                 startSpeed,
                                                 endSpeed,
                                                 order, sampleRate
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

        threed_os, twod_oc, counter_or = order_spectrum_for_const_fluctuation(threed_os,
                                                                              twod_oc,
                                                                              counter_or,
                                                                              vib_ndarray[
                                                                              vib_start_index + counter_or * sampsPerChan:
                                                                              vib_start_index + (
                                                                                      counter_or + 1) * sampsPerChan],
                                                                              orderSpectrumCalcInfo,
                                                                              orderCutCalcInfo,
                                                                              0,
                                                                              vib_start_index + counter_or * sampsPerChan,
                                                                              sampleRate,
                                                                              speed,
                                                                              db_flag=
                                                                              False)


    threed_os['xValue'] = threed_os['xValue'][:counter_or]
    threed_os['zValue'] = threed_os['zValue'][:counter_or]

    for result in twod_oc:
        result['xValue'] = result['xValue'][:counter_or]
        result['yValue'] = result['yValue'][:counter_or]

    twod_os = twod_order_spectrum(threed_os, None, None)

    rev=rev[:rev_index]
    rev_diff=1/orderSpectrumCalcInfo["maxOrder"]/2
    rsp=interp1d(rev,vib_ndarray[:len(rev)],kind='cubic',assume_sorted=True)(np.arange(0,rev[-1],rev_diff))

    fft_len=int(32/(rev_diff))-1
    fft_counter=0
    order_list=np.arange(maxOrder*32)*0.03125
    os_list=list()
    while (fft_counter+1) *fft_len<len(rsp):
        os_list.append(np.fft.rfft(rsp[fft_counter*fft_len:(fft_counter+1)*fft_len]*np.kaiser(fft_len, 9))/fft_len*2*0.707)

        fft_counter+=1


    rsp_yValue=np.sqrt(np.sum(np.square(os_list),axis=0)/len(os_list))
    plt.plot(twod_os[0]["xValue"],twod_os[0]["yValue"],c="r",label="norsp")
    plt.plot(order_list,rsp_yValue[:],c="b",label="rsp")
    plt.legend()

    plt.show()
    print(1)
