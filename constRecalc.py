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
import base64
import json
import os
import pickle
import signal

import numpy as np
import logging
import traceback
import sys
import getopt
import h5py
import platform
from datetime import datetime

# 读取基本信息
from parameters_for_const_recalc import basic_info_update, task_info_update, speed_calc_info_update, \
    speed_recog_info_update, time_domain_calc_info_update, order_spectrum_calc_info_update, \
    order_cut_calc_info_update, oned_os_calc_info_update, cepstrum_calc_info_update, \
    ssa_calc_info_update, stat_factor_calc_info_update
from indicator_tools_for_const_recalc import twod_time_domain_for_share, \
    order_spectrum_for_const_fluctuation, \
    twod_stat_factor_for_const_fluctuation, oned_time_domain, \
    convert_time_speed_for_const, twod_order_spectrum, cepstrum, oned_order_spectrum, db_convertion, \
    oned_stat_factor_mean_for_const
from initial import create_empty_twodtd_for_share, create_empty_temptd_for_share, \
    create_empty_twodsf_for_const, create_empty_tempsf_for_const, create_empty_threedos_for_share, \
    create_empty_twodoc_for_share, update_nvh_data_for_thread

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
version = 4  # 目前只适用基础版本的qDAQ

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
            f.close()
    return data


def write_json(filename, data, flag=0, indent=4):
    """
    功能：json格式数据写入文件，其中包括其他编码格式（为了防止用户在本地修改文件）。默认不进行其他方式编码
    输入：
    1. filename：文件名，包含完整的路径信息
    2. data：json格式的数据
    3. flag：编码标志，默认为0，表示直接写入json字符串，标志及其意义如下：
        3.1 flag=1：b64encode
        3.2 flag=2：b32encode
        3.3 flag=3：b16encode
        3.4 flag=4：pickle进行序列化
        3.5 其他，直接写入json字符串
    返回：写入数据到文件中
    function: write the data into json file
    :param
    filename: the full path of target JSON file
    data(dict): the data need to write into the JSON file
    :return
    existed json file
    """
    if flag == 1:
        # b64编码
        with open(filename, 'wb') as f:
            f.write(base64.b64encode(json.dumps(data).encode('utf-8')))
    elif flag == 2:
        # b32编码
        with open(filename, 'wb') as f:
            f.write(base64.b32encode(json.dumps(data).encode('utf-8')))
    elif flag == 3:
        # b16编码
        with open(filename, 'wb') as f:
            f.write(base64.b16encode(json.dumps(data).encode('utf-8')))
    elif flag == 4:
        # pickle进行序列化
        open(filename, "wb").write(pickle.dumps(data))
    else:
        # 直接写入
        with open(filename, 'w') as f:
            json.dump(data, f, indent=indent)


def find_argmin_in(a):
    def find_argmin(y):
        return np.argmin(np.abs(a - y))

    return find_argmin


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


def ramp_quality_for_const(speed_curve_x, speed_curve_y, speed_recog_info, recog_index):
    """
    功能：计算转速的爬坡质量
    输入：
    1. 转速曲线的x轴，必须是array
    2. 转速曲线的y轴，必须是array
    3. 转速识别信息，里面包含了目标转速区间和期望时长
    4. 目标工况的索引，即第几个测试段
    返回：爬坡质量
    """
    if len(speed_curve_x) > 0:
        # 恒速段
        target_speed = np.mean(
            [speed_recog_info['startSpeed'][recog_index], speed_recog_info['endSpeed'][recog_index]])
        quality = 1 - np.sqrt(np.sum(np.power(speed_curve_y / target_speed - 1, 2)) / len(speed_curve_y))
    else:
        quality = 0
    return {"name": "RampQuality", "unit": "", "value": quality, "indicatorDiagnostic": 1}


# 原始数据读取函数
def raw_data_read(h5filename, channel_names):
    # 读取传感器数据（振动或声音）
    sensor_data = dict()
    with h5py.File(h5filename, 'r') as h5pyFile:
        data_group = h5pyFile['AIData']
        for channel_name in channel_names:
            sensor_data[channel_name] = np.array(data_group[channel_name], dtype='float')
    return sensor_data


def sensor_confirm(task_info, basic_info):
    """
    功能：确认传感器信息设置是否正确（如果设置的传感器数量和采集的不一致则报错）
    输入：
    1. 数据采集参数信息， 里面包含每个通道的名称
    2. 基础信息，包含传感器id信息
    返回：记录信息错误并报错
    """
    if len(task_info["units"]) != len(basic_info["sensorId"]):
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


if __name__ == '__main__':
    # 主程序入口
    # config = dict()
    # config["filepath"] = r"D:\qdaq\对比\截时重计算"
    # config["filename"] = "byd-alltests-1mic.h5"
    try:
        # just for internal test
        # import matplotlib.pyplot as plt
        # import time
        # t1 = time.time()
        # config = {'filepath': r'D:\qdaq\test\210925-1',
        #           "filename": 'TZ220XS004M20210001M021512002_210518025321_210819054719.h5'}

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
                raw_config["orderSpectrumCalcInfo"], speedCalcInfo, speedRecogInfo['overallMinSpeed'],
                taskInfo)
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
        except Exception:
            print("data reading error")
            traceback.print_exc()
            logging.warning("config info change failed, failed msg:" + traceback.format_exc())
            sys.exit()

        try:
            test_result = create_empty_final_result(basicInfo, recalc_config)

            for i, testName in enumerate(speedRecogInfo["testName"]):
                if testName == recalc_config["testName"]:
                    testNameIndex = i

            # 开始点结束点的索引
            startIndex, endIndex = int(recalc_config['startTime'] * sampleRate), int(
                recalc_config['endTime'] * sampleRate)

            # 开始计算
            for sensor_index, chan_name in enumerate(taskInfo['targetChan']):
                # cut the vibration
                # vib_ndarray = sensor_data[chan_name][startIndex:endIndex]
                # 恒速电机重计算的逻辑和nvh中的逻辑一样，qdaq可以一次性做完所有角度重采样，但恒速电机没有角度重采样，
                # 阶次谱计算帧长是变化的
                vib_ndarray = sensor_data[chan_name]
                counter_data_queue = 0
                while startIndex + counter_data_queue * sampsPerChan <= endIndex:

                    calSize = timeDomainCalcInfo["calSize"]
                    if counter_data_queue == 0:
                        # 测试段的第一帧数据
                        # 左索引
                        vib_start_index = startIndex
                        # 帧内左索引
                        vib_left_index = vib_start_index
                        # 更新右索引，右端索引处没有值
                        vib_right_index = min(startIndex + (counter_data_queue + 1) * sampsPerChan,
                                              endIndex)

                        # maxT是测试段的最长时间
                        maxT = recalc_config["endTime"] - recalc_config["startTime"]

                        # endSpeed是测试段内的最大速度
                        maxSpeed = speedRecogInfo["endSpeed"][testNameIndex]
                        minSpeed = speedRecogInfo["startSpeed"][testNameIndex]

                        # nvh计算的帧长,该值与参数配置/speed进程中的值不同，该值在不同的测试段不同
                        # # 每一帧大约计算两次阶次谱
                        # sampsPerChan = int(param.orderSpectrumCalcInfo["revNum"] * (
                        #         2 - param.orderSpectrumCalcInfo["overlapRatio"]) / (minSpeed / 60) * sampleRate)
                        # 每一帧大约计算一次阶次谱
                        sampsPerFrame_os = int(
                            orderSpectrumCalcInfo["revNum"] / (minSpeed / 60) * sampleRate)

                        # maxLen是该测试段时域计算时按帧长calSize最多计算多少次
                        maxLen = int(maxT * sampleRate // calSize) + 2
                        # 创建空的twodtd和空的temp_td，创建的容量保证能放下测试段内的数据，
                        # 将来测试段结束后再进行截取，然后再进行计算
                        twod_td = create_empty_twodtd_for_share(timeDomainCalcInfo, sensor_index,
                                                                maxLen,
                                                                indicator_diagnostic=
                                                                speedRecogInfo[
                                                                    'initial_indicator_diagnostic'][
                                                                    testNameIndex])
                        temp_td = create_empty_temptd_for_share(maxLen)
                        # 一个测试段内计算了多少个twod_td计算了多少次 ，新测试段开始时赋值为0
                        index_twod_td = 0

                        # 该测试段
                        max_circle = maxT * maxSpeed / 60
                        # 记录按圈计算过程中每一个轴计算过的次数
                        twod_sf_counter = dict()
                        twod_sf = create_empty_twodsf_for_const(statFactorCalcInfo, sensor_index,
                                                                max_circle,
                                                                indicator_diagnostic=
                                                                speedRecogInfo[
                                                                    'initial_indicator_diagnostic'][
                                                                    testNameIndex])
                        temp_sf = create_empty_tempsf_for_const(statFactorCalcInfo, sensor_index,
                                                                max_circle)

                        # 每32圈计算一次，overlapRatio为0.75 即步进长度为8
                        # osMaxLen = int(maxT * maxSpeed / 60 // (param.orderSpectrumCalcInfo['revNum'] * (
                        #         1 - param.orderSpectrumCalcInfo['overlapRatio']))) + 1
                        osMaxLen = int(maxT * sampleRate // sampsPerFrame_os) + 2
                        threed_os = create_empty_threedos_for_share(orderSpectrumCalcInfo,
                                                                    taskInfo,
                                                                    sensor_index, osMaxLen,
                                                                    db_flag=basicInfo['dBFlag'],
                                                                    indicator_diagnostic=
                                                                    speedRecogInfo[
                                                                        'initial_indicator_diagnostic'][
                                                                        testNameIndex])
                        twod_oc = create_empty_twodoc_for_share(orderCutCalcInfo,
                                                                taskInfo,
                                                                sensor_index, osMaxLen,
                                                                db_flag=basicInfo['dBFlag'],
                                                                indicator_diagnostic=
                                                                speedRecogInfo[
                                                                    'initial_indicator_diagnostic'][
                                                                    testNameIndex])
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
                        vib_right_index = min(startIndex + (counter_data_queue + 1) * sampsPerChan,
                                              endIndex)
                        len_vib_frame = vib_right_index - vib_left_index

                    # 测试段内的第几个data
                    counter_data_queue += 1

                    # 进行测试段内数据的计算
                    # 以传来的第一个值作为测试段的0时刻点，作为0圈数点
                    # 角度重采样，一测试段内传进来的第一个vib点为0时刻点，0圈数点

                    # 测试段内vib的长度
                    len_vib = vib_right_index - vib_start_index

                    # 分帧计算时域信息
                    # time domain indicators
                    try:
                        twod_td, temp_td, counter_td, index_twod_td, last_calc_index = twod_time_domain_for_share(
                            vib_ndarray,
                            last_calc_index,
                            vib_right_index,
                            calSize,
                            timeDomainCalcInfo["indicatorNestedList"][sensor_index],
                            timeDomainCalcInfo["refValue"][sensor_index],
                            twod_td,
                            temp_td,
                            counter_td,
                            sampleRate,
                            index_twod_td)
                    except Exception:
                        logging.error("二维时间域计算失败")
                        logging.error(traceback.format_exc())

                    try:
                        # 转速计算
                        while vib_start_index + (counter_or + 1) * sampsPerFrame_os <= vib_right_index:
                            speed = speed_calibration_with_orderlist(
                                vib_ndarray[vib_start_index + counter_or * sampsPerFrame_os:
                                            vib_start_index + (counter_or + 1) * sampsPerFrame_os],
                                speedRecogInfo["startSpeed"][testNameIndex],
                                speedRecogInfo["endSpeed"][testNameIndex],
                                speedRecogInfo["order"][testNameIndex], sampleRate
                            )

                            rpm.append(speed)
                            rpml.append(
                                ((counter_or + 0.5) * sampsPerFrame_os + vib_start_index) / sampleRate)

                            if counter_or == 0:

                                rev[rev_index:(rev_index + sampsPerFrame_os)] = np.arange(
                                    sampsPerFrame_os) * (speed / 60 / sampleRate)
                                rev_index += sampsPerFrame_os
                            else:
                                rev[rev_index:(rev_index + sampsPerFrame_os)] = rev[
                                                                                    rev_index - 1] + np.arange(
                                    1, sampsPerFrame_os + 1) * (speed / 60 / sampleRate)
                                rev_index += sampsPerFrame_os

                            # 阶次谱计算
                            threed_os, twod_oc, counter_or = order_spectrum_for_const_fluctuation(
                                threed_os,
                                twod_oc,
                                counter_or,
                                vib_ndarray[
                                vib_start_index + counter_or * sampsPerFrame_os:
                                vib_start_index + (counter_or + 1) * sampsPerFrame_os],
                                orderSpectrumCalcInfo,
                                orderCutCalcInfo,
                                sensor_index,
                                vib_left_index,
                                sampleRate,
                                speed,
                                db_flag=
                                basicInfo[
                                    'dBFlag'])

                    except Exception:
                        print("转速计算失败")
                        logging.error(traceback.format_exc())

                        break
                    try:
                        # 按圈计算
                        twod_sf, temp_sf = twod_stat_factor_for_const_fluctuation(twod_sf, temp_sf, rev,
                                                                                  rev_index,
                                                                                  vib_ndarray[
                                                                                  vib_start_index:vib_right_index],
                                                                                  twod_sf_counter,
                                                                                  vib_start_index,
                                                                                  sampleRate,
                                                                                  sensor_index,
                                                                                  statFactorCalcInfo)
                    except Exception:
                        logging.error("按圈计算出错")
                        logging.error(traceback.format_exc())

                # 该传感器分帧计算完成
                twod_td, temp_td, twod_sf, temp_sf, threed_os, twod_oc = update_nvh_data_for_thread(
                    twod_td, temp_td, index_twod_td,
                    twod_sf, temp_sf, twod_sf_counter,
                    threed_os, twod_oc, counter_or)

                # prepare test finished

                # update 1D time domain indicators into final test result
                try:
                    if test_result['resultData'][sensor_index]['dataSection'][0]['onedData'] is None:
                        test_result['resultData'][sensor_index]['dataSection'][0]['onedData'] = list()
                    test_result['resultData'][sensor_index]['dataSection'][0]['onedData'].extend(
                        oned_time_domain(temp_td, calSize, timeDomainCalcInfo, sensor_index,
                                         db_flag=basicInfo['dBFlag']))
                except:
                    logging.error("一维时间域计算出错")
                    logging.error(traceback.format_exc())

                try:
                    rampQuality = ramp_quality_for_const(rpml, rpm, speedRecogInfo, testNameIndex)
                    test_result['resultData'][sensor_index]['dataSection'][0][
                        'onedData'].append(rampQuality)
                    test_result['resultData'][sensor_index]['dataSection'][0][
                        'onedData'].append(
                        {"name": "Speed", "unit": "", "value": np.mean(rpm), "indicatorDiagnostic": 1})
                except:
                    logging.error("转速指标添加出错")
                    logging.error(traceback.format_exc())
                try:
                    # update 1D indicators by revolution into final test result
                    if test_result['resultData'][sensor_index]['dataSection'][0]['onedData'] is None:
                        test_result['resultData'][sensor_index]['dataSection'][0]['onedData'] = list()
                    # 计算一维按圈计算指标并更新二维结果
                    oned_sf, twod_sf = oned_stat_factor_mean_for_const(temp_sf, twod_sf, statFactorCalcInfo,
                                                                       sensor_index)
                    test_result['resultData'][sensor_index]['dataSection'][0]['onedData'].extend(oned_sf)
                except:
                    logging.error("一维按圈计算出错")
                    logging.error(traceback.format_exc())

                # 更新二维时间域结果的x轴
                twod_td = convert_time_speed_for_const(twod_td)
                test_result['resultData'][sensor_index]['dataSection'][0]['twodTD'] = twod_td

                # 更新二维按圈计算指标的x轴
                twod_sf = convert_time_speed_for_const(twod_sf,indicator_num=statFactorCalcInfo['indicatorNum'][sensor_index])
                # 添加二维按圈计算指标到二维时间域指标中
                test_result['resultData'][sensor_index]['dataSection'][0]['twodTD'].extend(twod_sf)

                # 向二维结果数据中保存转速曲线
                test_result['resultData'][sensor_index]['dataSection'][0][
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
                test_result['resultData'][sensor_index]['dataSection'][0][
                    'startTime'] = vib_start_index / sampleRate
                test_result['resultData'][sensor_index]['dataSection'][0][
                    'endTime'] = vib_right_index / sampleRate
                try:
                    # update 2D order spectrum into final test result
                    twod_os = twod_order_spectrum(threed_os, taskInfo, sensor_index)
                except:
                    logging.error("二维阶次谱计算出错")
                    logging.error(traceback.format_exc())

                try:
                    test_result['resultData'][sensor_index]['dataSection'][0]['twodCeps'] = \
                        cepstrum(twod_os, cepstrumCalcInfo, taskInfo,sensor_index, db_flag=basicInfo['dBFlag'])
                except:
                    logging.error("倒阶次谱计算出错")
                    logging.error(traceback.format_exc())

                # update 1D order domain indicators into final test result
                try:
                    test_result['resultData'][sensor_index]['dataSection'][
                        0]['onedData'] += oned_order_spectrum(twod_os,
                                                                          onedOSCalcInfo,
                                                                          taskInfo,
                                                                          modulationDepthCalcInfo,
                                                                          sensor_index,
                                                                          db_flag=
                                                                          basicInfo[
                                                                              'dBFlag'])
                except:
                    logging.error("阶次谱计算出错")
                    logging.error(traceback.format_exc())

                # 更新二维阶次谱和三维阶次谱的阶次信息（一定要在阶次切片结果计算完成以后）
                if speedCalcInfo['speedRatio'] != 1:
                    if twod_os:
                        twod_os[0]['xValue'] /= orderSpectrumCalcInfo['convertOrder']
                        threed_os['yValue'] = orderSpectrumCalcInfo['convertOrder']
                # 更新结果到结果数据中
                # 更新结果到结果数据中
                # 更新结果到结果数据中（包括dB转换）
                if basicInfo['dBFlag']:
                    twod_os[0]['yUnit'] = 'dB'
                    twod_os[0]['yValue'] = db_convertion(twod_os[0]['yValue'],taskInfo['refValue'][sensor_index])
                # 更新yValue由ndarray变为list(),
                # 之前会用twod_os[0]['yValue']进行运算，在运算完成之后进行更新
                twod_os[0]['yValue'] = twod_os[0]['yValue'].tolist()
                test_result['resultData'][sensor_index]['dataSection'][0]['twodOS'] = twod_os

                # 更新二维阶次切片和三维阶次谱的X轴信息
                twod_oc, threed_os = convert_time_speed_for_const(twod_oc, threed_os=threed_os)
                test_result['resultData'][sensor_index]['dataSection'][0]['twodOC'] = twod_oc

                del threed_os["tempXi2"]

        except Exception:
            print("test result recalc error")
            traceback.print_exc()
            logging.warning("config info change failed, failed msg:" + traceback.format_exc())
            sys.exit()


        try:
            # print(json.dumps(test_result))
            write_json(os.path.join(config['filepath'], 'recalcTestResult.json'), test_result)
            # 计算完成后打印Done，告诉后端计算完成
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
