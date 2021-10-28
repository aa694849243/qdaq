import configparser
import json
import logging
import os.path
import platform
import sys
import time
import traceback
import zlib

import h5py
from nptdms import TdmsFile, ChannelObject, TdmsWriter
import numpy as np
from scipy import stats, fftpack
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import csv

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

def resolver(speed_sin, speed_cos, cut_level, coil, ppr, last_angle_l1f, loc_l1f, last_angle_l2f,
             loc_l2f, counter,
             frame):
    """
    双路旋变转速信号处理函数
    功能： 基于双路旋变信号（正弦余弦信号），找到每一个上升沿之间对应的角度差，根据每一个上升沿所在的角度去模拟上升沿位置，最终输出上升沿位置序列
    输入：
    1. 单帧正弦信号
    2. 单帧余弦信号
    3. 检测电平位（主要为了剔除0值附近的信号，防止干扰）
    4. 极对数
    5. 要输出的ppr
    6. 上一帧的最后一个角度值
    7. 上一帧的最后一个角度值对应的X轴（即时间点）
    8. 上两帧对应的最后一个角度值
    9. 上两帧的最后一个角度值对应的X轴（即时间点）
    10. 计数器，记录转过了多少个180度
    11. 帧长
    返回：
    1. 脉冲位置信息
    2. 上一帧的最后一个角度值
    3. 上一帧的最后一个角度值对应的X轴（即时间点）
    4. 上两帧对应的最后一个角度值
    5. 上两帧的最后一个角度值对应的X轴（即时间点）
    6. 计数器，记录转过了多少个180度
    function: to detect trigger point based on resolver signal(sin and cos)
    :param
    speed_sin(list): raw sin signal(1 frame)
    speed_cos(list): raw cos signal(1 frame)
    cut_level(float): to filter noise and detect out angle
    coil(int): coil num of resolver device(refer to the structure)
    ppr(int): target ppr num
    last_angle_l1f(float): last angle value of last frame
    loc_l1f(float): corresponding location of last_angle_l1f
    last_angle_l2f(float): last 2nd angle value of last frame
    loc_l2f(float): corresponding location of last_angle_l2f
    counter(int): index of data frame
    frame(int): length of 1 frame data
    :return:
    tl(list): trigger location of present frame
    following values just to real-time update:
        last_angle_l1f, loc_l1f, last_angle_l2f, loc_l2f, counter
    """
    # initial the trigger location
    tl = list()

    def sigma_calc(data):
        # 计算标准差
        # return np.sqrt(sum(np.power((data - np.mean(data)), 2)) / (len(data))) # 标准公式
        return np.std(data)

    def kick_out_exceptions(x_data, y_data, param_sigma, param_fit):
        # 剔除异常点
        # to remove the exception points
        counter_filter = 0
        while 1:
            # 进行线性拟合，param_fit为多项式系数，通常为1，表示线性拟合
            target_curve = np.poly1d(np.polyfit(x_data, y_data, param_fit))
            # 计算实际数据与拟合线的差值
            delta_y_value = np.array(y_data) - target_curve(x_data)
            # 计算差值的方差
            sigma_value = sigma_calc(delta_y_value)
            # 根据设定sigma去除倍数（倍数内的保留）来设定过滤条件
            condition1 = np.abs(delta_y_value - np.mean(delta_y_value)) <= (param_sigma * sigma_value)
            # 数据过滤
            filter_y_data = np.array(y_data)[condition1]
            filter_x_data = np.array(x_data)[condition1]
            # 更新计数器
            counter_filter += 1
            # 更新数据重新进行过滤，直到无法剔除
            if len(y_data) > len(filter_y_data):
                # 顾虑前后的数据长度一致表示无法再过滤了
                y_data, x_data = filter_y_data, filter_x_data
            else:
                break

        return filter_x_data, filter_y_data, counter_filter

    # 去除零附近的数据点（这些数据点干扰严重），后续进行插值补上去掉的数据点
    # cut out the better signal
    cutted_sin = speed_sin[(abs(speed_sin) > cut_level) & (abs(speed_cos) > cut_level)]
    cutted_cos = speed_cos[(abs(speed_sin) > cut_level) & (abs(speed_cos) > cut_level)]
    # 记录剩余数据点的索引信息
    cutted_loc = np.where((abs(speed_sin) > cut_level) & (abs(speed_cos) > cut_level))[0]

    # convert into angle（0，180）
    # 计算角度值（基本正余弦信号）
    angle = 90 + np.arctan(cutted_sin / cutted_cos) * 180 / np.pi
    # 根据极对数转换点击旋转的实际角度参考值，每个ppr转过多少度
    angle_ppr = 360 * coil / ppr  # 该角度是旋变信号的角度，电机的角度为360/ppr
    # 设定判断角度跳跃的标准
    shift_angle = -76

    # 计算累积角度
    if len(angle) > 0:
        # 有数据才进行计算
        # The angle angle (0180) is changed into monotonically increasing angle_ Cum
        angle_cum = list()

        # If there is a previous frame, add the last point of the last frame first
        if loc_l1f:
            # 如果存在上一帧的数据点则进行判断是否需要更新角度值（主要处理第一个数据点）
            start_angle = last_angle_l1f - counter * 180
            if angle[0] - start_angle < shift_angle:
                # 如果出现角度跳跃，则认为需要加180度
                counter += 1
        # 记录累积角度
        angle_cum.append(angle[0] + counter * 180)

        # 处理剩余数据点（除第一个数据点外）
        for i in range(1, len(angle)):
            # 两个条件需要同时满足：1. 角度发生跳跃；2. 不是连续的数据点（防止噪点）
            if angle[i] - angle[i - 1] < shift_angle and cutted_loc[i] - cutted_loc[i - 1] > 1:
                counter += 1
            # 更新累积角度序列
            angle_cum.append(angle[i] + counter * 180)

        if len(angle) > 1:
            # 只有超过两个数据点才能进行拟合，所以必须确保数据量才进行剔除，这里设置的线性回归，5倍的标准差（超过5倍才去除）
            # At least two points can be merged to remove noise
            filter_scl, filter_angle_cum, _ = kick_out_exceptions(cutted_loc, angle_cum, 5, 1)
        else:
            # 数据量很少就不用进行剔除操作了
            # There is only one point in the frame
            filter_scl, filter_angle_cum = cutted_loc, angle_cum
        if loc_l1f:
            # 如果存在上一帧的数据点需要结合前一帧的数据点进行插值计算
            # There are data points in the last frame
            if loc_l2f:
                # 如果前两帧存在数据点则需要结合前两帧的数据进行插值计算
                # There are also data points in the previous frame, and the quadratic difference is performed
                # 生产插值曲线函数，基于角度及其对应的位置信息，这里需要注意用了二次方差值（主要考虑转速拐点），前两帧的数据点只是用于构造曲线，
                # 因为要更新上一帧的数据点
                Curve = interp1d(np.array([loc_l2f - 2 * frame, loc_l1f - frame, filter_scl[-1]]),
                                 np.array([last_angle_l2f, last_angle_l1f, filter_angle_cum[-1]]),
                                 kind='quadratic',
                                 assume_sorted=True)
            else:
                # 前两帧不存在数据的情况下，则直接根据前一帧的数据进行插值计算，这里用的是线性差值
                # If there is no data point in the previous frame, linear difference is made
                Curve = interp1d(np.array([loc_l1f - frame, filter_scl[-1]]),
                                 np.array([last_angle_l1f, filter_angle_cum[-1]]), kind='linear',
                                 assume_sorted=True)
            # 生成序列的X值（补全X轴，包括上一帧最后一个数据点与当前帧第一个数据点之间的数据）
            index_serial = np.arange(loc_l1f - frame, filter_scl[-1] + 1, 1)
            # 生成角度序列（累积角度），该序列已经补上了最开始剔除的数据点处的角度
            angle_serial = Curve(index_serial)
            # 基于每个ppr对应的角度可以得到锯齿形状的数据序列（每个跳变的位置对应的一个脉冲信号）
            angle_serial_remainder = angle_serial % angle_ppr
            # 识别脉冲触发点，angle_ppr * 0.5是为了保证能识别到（防止干扰）
            triggermarker = 1 * (np.diff(angle_serial_remainder) < -(angle_ppr * 0.5))
            # 记录每个脉冲位置信息（之后便可用于转速计算和角度域重采样），前一帧的脉冲位置应该为负
            tl.extend(list(np.where(triggermarker == 1)[0] - (frame - loc_l1f)))
            # 更新要返回的信息（前一帧，前两帧的最后一个数据点）
            last_angle_l2f = last_angle_l1f
            last_angle_l1f = filter_angle_cum[-1]
            loc_l2f = loc_l1f
            loc_l1f = filter_scl[-1]
            last_angle_l1f = filter_angle_cum[-1]
            loc_l1f = filter_scl[-1]
        else:
            # 如果上一帧不存在数据点，前两帧也认为不存在数据点（事实上也应该不存在这种上一帧没有但是上上帧有数的情况），只处理当前帧
            Curve = interp1d(np.array([0, filter_scl[-1]]),
                             np.array([filter_angle_cum[0], filter_angle_cum[-1]]),
                             kind='linear', assume_sorted=True)
            # 生成序列的X值（补全X轴）
            index_serial = np.arange(filter_scl[0], filter_scl[-1] + 1, 1)
            # 生成角度序列（累积角度），该序列已经补上了最开始剔除的数据点处的角度
            angle_serial = Curve(index_serial)
            # 基于每个ppr对应的角度可以得到锯齿形状的数据序列（每个跳变的位置对应的一个脉冲信号）
            angle_serial_remainder = angle_serial % angle_ppr
            # 识别脉冲触发点，angle_ppr * 0.5是为了保证能识别到（防止干扰）
            triggermarker = 1 * (np.diff(angle_serial_remainder) < -(angle_ppr * 0.5))
            # 记录每个脉冲位置信息（之后便可用于转速计算和角度域重采样）
            tl.extend(list(np.where(triggermarker == 1)[0]))
            # 更新要返回的信息（前一帧，前两帧的最后一个数据点）
            last_angle_l2f = last_angle_l1f
            last_angle_l1f = filter_angle_cum[-1]
            loc_l2f = loc_l1f
            loc_l1f = filter_scl[-1]
    else:
        # 如果当前帧不存在任何数据，则直接更新要返回的信息
        # no angle points
        loc_l2f = loc_l1f
        loc_l1f = None
        last_angle_l2f = last_angle_l1f
        last_angle_l1f = None

    return tl, last_angle_l1f, loc_l1f, last_angle_l2f, loc_l2f, counter


def write_tdms(filename, group_name, channel_name, data, properties=None, mode='a'):
    """
    功能：按通道写入数据到TDMS文件中（tdms数据结构，文件（file）->数据组（group）->数据通道（channel））
    输入:
    1. 文件名（全路径）
    2. tdms的数据组名称
    3. tdms的数据通道
    4. 要写入指定通道数据
    5. 属性信息，默认为空
    6. 写入模式，默认为添加模式
    返回：本地tdms文件
    function: write the data into TDMS file
    :param
    filename(string): the full path of target TDMS file
    groupname(string): the group name for TDMS write
    channelname(string): the channel name for TDMS write
    data(list): data need to write into TDMS
    properties(dict): the properties of channel data, default as {}
    mode(char): 'w' or 'a', 'w' means it will remove all the existed data and write new data,
    'a' means it just append new data, hold the existed data, default as 'a'
    :return:
    existed TDMS file
    """
    # 创建数据通道对象
    channel_object = ChannelObject(group_name, channel_name, data, properties)
    # 写入数据到文件
    with TdmsWriter(filename, mode) as tdms_writer:
        tdms_writer.write_segment([channel_object])

def rpm_calc_for_share(trigger_array, start, end, sampleRate, average, step, rpmFactor, rpml_array,
                       rpm_array,
                       rpm_index):
    """

    Args:
        trigger_array: 上升沿/下降沿位置
        start: 该帧计算出了第7，8，9，10，11，12（从0计数）个trigger,start必定小于7，可以为6，可以为5，
        因为上一帧计算到的最后的trigger不确定，特例是start为0
        end: 该帧计算出了第7，8，9，10，11，12（从0计数）个trigger,则end传递进来的是12
        sampleRate: 采样率
        speed_calc_info: 转速计算信息
        rpml_array: 整个测试的转速曲线数组 时间
        rpm_array: 整个测试的转速曲线数组  转速值
        rpm_index: 下一个rpm点要写入的索引

    Returns:

    """

    if start == 0:
        # 第一次算
        if end < average:
            # 算不出来
            return start, rpm_index
        else:
            # 能算出值
            start = average - step

    # 本次计算能计算出来多少个
    len = (end - start) // step
    for i in range(len):
        # 中间是speed_calc_info["averageNum"]个间隔
        delta_t = (trigger_array[start + step] - trigger_array[start + step - average]) / sampleRate
        rpm_array[rpm_index] = rpmFactor / delta_t
        # 取中间时刻作为speed_time
        # rpml_array[rpm_index]=(trigger_array[start+step]+
        #          trigger_array[start+step-average])/2/sampleRate
        # 取averager最后一个上升沿为时刻点,将来乘以采样率得到的值取int即为上升沿/下降沿的位置
        rpml_array[rpm_index] = (trigger_array[start + step]) / sampleRate
        rpm_index += 1
        start = start + step
    return start, rpm_index



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


def butter_filter(signal, wn, fs, order=3, btype='lowpass'):
    """
    function: signal filter user butter
    :param
    signal(list): input signal(before filtering)
    Wn(list): normalized cutoff frequency （one for high/low pass filter，two for band pass/stop filter)
    sampleRate(int): sample rate of input signal
    order(int)：filter order（generally the order higher， transition band narrower）default as 5
    btyte(str): filter type（low pass：'low'；high pass：'high'；band pass：'bandpass'；band stop：'bandstop'）
    default as low
    # analog: digital or analog filter，cutoff of analog filter is angular frequency，for digital is relative
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




def single_resolver_version3(speed_array, left_index, right_index, speed_last_frame, sampleRate, coils,
                    trigger_array, index_trigger, ppr):
    # left_index为实际需要的段，滤波时要向左取帧长1/10
    # right_index为原始数据存储到的位置，左帧长1/10处在该帧滤波后的数据不使用
    right_more=int((right_index-left_index)/10)
    if left_index==0:
        left_more=0
    else:
        left_more=right_more
    frame = speed_array[left_index-left_more:right_index]
    # 除第一帧，frame以上一帧最后一个零点的右中值跳跃点开始
    try:
        frame_filter=butter_filter(frame,[int((10000+10000/60)*1.4)],sampleRate,order=3)
    except Exception:
        time.sleep(0.1)
    hSin = fftpack.hilbert(frame_filter)
    envSin = np.sqrt(frame_filter ** 2 + hSin ** 2)
    max_envSin = np.max(envSin)
    low_level = max_envSin / 5
    half_loc = np.where(envSin > max_envSin / 2)[0]
    half_loc_skip = np.where(np.diff(half_loc) != 1)[0]
    point_per_fish = sampleRate / (speed_last_frame / 60 * 2 * coils)
    min_list = list()
    trigger_per_fish =int(ppr / 2 / coils)

    if not len(half_loc_skip):
        return index_trigger,left_index


    left_index_for_next_frame = None
    for i in range(len(half_loc_skip)):
        if (half_loc[half_loc_skip[i] + 1] - half_loc[half_loc_skip[i]]) > (len(frame)/len(half_loc_skip)/3*0.6):
            min_loc = np.argmin(envSin[half_loc[half_loc_skip[i]]:half_loc[half_loc_skip[i] + 1]])
            if envSin[min_loc + half_loc[half_loc_skip[i]]] < low_level:
                min_list.append(min_loc + half_loc[half_loc_skip[i]])
                # left_index_for_next_frame = half_loc[half_loc_skip[i]]-int(point_per_fish/4)

    # min_list 和 min_array都是以帧起点为0点
    min_array = np.array(min_list)
    for loc in min_array:
        envSin[loc:] *= -1

    try:
        envSin_filter = butter_filter(envSin, [int(len(half_loc_skip)*(sampleRate/len(frame) * 1.4 ))], sampleRate,order=3)[left_more:-right_more]
    except Exception:
        traceback.print_exc()

    zeroLoc = np.where(np.diff(1 * (envSin_filter >= 0)) != 0)[0]


    for i in range(len(zeroLoc) - 1):
        # 下面用的列表生成器的长度等于np.int(ppr/2/coil),即不包括一条鱼的最后一个点，这个点是下一条鱼的第一个点
        trigger_array[index_trigger:index_trigger + trigger_per_fish] = \
            np.linspace(zeroLoc[i], zeroLoc[i + 1], trigger_per_fish, endpoint=False) \
            + left_index
        index_trigger += trigger_per_fish

    if len(zeroLoc)==0:
        return index_trigger,left_index
    else:
        return index_trigger, zeroLoc[-1]-int(len(frame)/len(half_loc_skip)/4) + left_index


def speed_calc_single_resolver(speed_sin,sampleRate,coils):


    time_after_readallrawdata=time.time()
    trigger_array=np.zeros(len(speed_sin)//2)
    speed_last_frame=3000
    samplePerChan=8192
    ppr=32
    average_Num=64
    step=4
    ppr=72
    average_Num=144
    step=4
    rpm_factor=60*average_Num/ppr
    left_index=0
    right_index=0
    index_trigger=0
    counter=0

    rpml_array=np.zeros(len(speed_sin)//2)
    rpm_array=np.zeros(len(speed_sin)//2)
    rpm_index=0
    last_trigger_for_rpm_cal_index=0

    # left_index=57009
    # right_index=65536
    # speed_last_frame=2489.969604863222


    while counter*samplePerChan < len(speed_sin):
        # print(left_index)
        if counter==12:
            time.sleep(0.1)
        index_trigger,left_index=single_resolver_version3(speed_sin,left_index,(counter+1)*samplePerChan,speed_last_frame,sampleRate,coils,trigger_array,
                        index_trigger,ppr)
        # index_trigger,left_index=single_resolver(speed_sin,left_index,right_index,speed_last_frame,sampleRate,coils,trigger_array,
        #                 index_trigger,ppr)
        last_trigger_for_rpm_cal_index, rpm_index = rpm_calc_for_share(trigger_array,
                                                           last_trigger_for_rpm_cal_index,
                                                           index_trigger - 1,
                                                           sampleRate,
                                                           average_Num,
                                                           step,
                                                           rpm_factor,
                                                           rpml_array,
                                                           rpm_array,
                                                           rpm_index)
        if rpm_array[rpm_index-1] - speed_last_frame<-500:
            time.sleep(0.1)
        speed_last_frame=rpm_array[rpm_index-1]
        counter+=1
    time_finish_speed_calc=time.time()
    # print(time_finish_speed_calc-time_after_readallrawdata)
    # plt.plot(rpml_array[:rpm_index],rpm_array[:rpm_index])
    # plt.show()
    # save_path="e:/resolver_compare/"
    # write_tdms(os.path.join(save_path,serial_no+"-"+channelName+"-resolvernew.tdms"),"speedData","speedLoc",rpml_array[:rpm_index])
    # write_tdms(os.path.join(save_path,serial_no+"-"+channelName+"-resolvernew.tdms"),"speedData","speedValue",rpm_array[:rpm_index])
    # write_tdms(os.path.join(save_path,serial_no+"-"+channelName+"-resolvernew.tdms"),"trigger","trigger",trigger_array[:index_trigger])
    return rpml_array[:rpm_index],rpm_array[:rpm_index]


def speed_calc_resolver2(speed_sin,speed_cos,sampleRate,coils):
    last_angle_l1f=0
    loc_l1f=0
    last_angle_l2f=0
    loc_l2f=0
    counter_angle=0
    sampsPerChan=8192

    ppr=72
    triggerLevel=0.5
    average_Num=144
    step=4
    rpm_factor=60*average_Num/ppr
    gv_dict_rawdata=dict()
    gv_dict_rawdata['Sin']=speed_sin
    gv_dict_rawdata['Cos']=speed_cos
    index_rawdata_backup=0
    index_rawdata=0
    trigger_array=np.zeros(len(gv_dict_rawdata["Sin"])//2)
    rpml_array=np.zeros(len(gv_dict_rawdata["Sin"])//2)
    rpm_array=np.zeros(len(gv_dict_rawdata["Sin"])//2)
    index_trigger=0
    rpm_index=0
    last_trigger_for_rpm_cal_index=0

    while index_rawdata <len(gv_dict_rawdata["Sin"]):
        index_rawdata_backup=index_rawdata
        index_rawdata+=sampsPerChan
        temptl, last_angle_l1f, loc_l1f, last_angle_l2f, loc_l2f, counter_angle = resolver(
            gv_dict_rawdata['Sin'][index_rawdata_backup:index_rawdata],
            gv_dict_rawdata['Cos'][index_rawdata_backup:index_rawdata],
            triggerLevel,
            coils,
            ppr,
            last_angle_l1f,
            loc_l1f,
            last_angle_l2f,
            loc_l2f,
            counter_angle,
            sampsPerChan)
        trigger_array[index_trigger:index_trigger + len(temptl)] = np.array(temptl) + index_rawdata_backup
        index_trigger += len(temptl)

        last_trigger_for_rpm_cal_index, rpm_index = rpm_calc_for_share(trigger_array,
                                                           last_trigger_for_rpm_cal_index,
                                                           index_trigger - 1,
                                                           sampleRate,
                                                           average_Num,
                                                           step,
                                                           rpm_factor,
                                                           rpml_array,
                                                           rpm_array,
                                                           rpm_index)
    # save_path = "e:/resolver_compare/"
    # write_tdms(os.path.join(save_path,serial_no+"-resolver2.tdms"),"speedData","speedLoc",rpml_array[:rpm_index])
    # write_tdms(os.path.join(save_path,serial_no+"-resolver2.tdms"),"speedData","speedValue",rpm_array[:rpm_index])
    # write_tdms(os.path.join(save_path,serial_no+"-resolver2.tdms"),"trigger","trigger",trigger_array[:index_trigger])
    return rpml_array[:rpm_index],rpm_array[:rpm_index]

# 原始数据读取函数
def FileReader(configJson, pathToFolder):
    vib = list()
    sensorNum = 0
    sin = list()
    cos = list()
    for i in range(0, len(configJson["files"])):
        if "Sin" in configJson["files"][i]:
            with open(os.path.join(pathToFolder, configJson["files"][i]), 'rb') as f:
                sin = f.read()
            sin = json.loads(str(zlib.decompress(sin), encoding="utf-8").replace("\'", "\""))["data"]
        if "Cos" in configJson["files"][i]:
            # in case Tacho is the speed channel name
            with open(os.path.join(pathToFolder, configJson["files"][i]), 'rb') as f:
                cos = f.read()
            cos = json.loads(str(zlib.decompress(cos), encoding="utf-8").replace("\'", "\""))["data"]
        if "Vib" in configJson["files"][i] and "Vib2" not in configJson["files"][i]:
            with open(os.path.join(pathToFolder, configJson["files"][i]), 'rb') as f:
                vibration1 = f.read()
            vib1 = json.loads(str(zlib.decompress(vibration1), encoding="utf-8").replace("\'", "\""))
            sensorNum += 1
            vib.append(vib1["data"])
        if "Vib2" in configJson["files"][i]:
            with open(os.path.join(pathToFolder, configJson["files"][i]), 'rb') as f:
                vibration2 = f.read()
            vib2 = json.loads(str(zlib.decompress(vibration2), encoding="utf-8").replace("\'", "\""))
            sensorNum += 1
            vib.append(vib2["data"])

    return np.array(sin), np.array(cos), vib, sensorNum

if __name__ == "__main__":


    # config.ini path
    # 读取配置文件config.ini里的信息
    # 定义配置文件路径
    basic_config_path = 'config.ini'
    with open(basic_config_path, 'r') as fp:
        basic_config = configparser.ConfigParser()
        basic_config.read_file(fp)

    read_path=str(basic_config["Path"]["read_path"])
    pic_savepath=basic_config["Path"]["save_path"]
    sampleRate=int(basic_config["Param"]["sampleRate"])
    coils=int(basic_config["Param"]["coils"])
    version=int(basic_config["Version"]["version"])
    if not os.path.exists(pic_savepath):
        os.makedirs(pic_savepath)

    # 原始数据保存为tdms或者hdf5格式
    if version==0:
        # 每一个文件的绝对路径
        abs_filenames=list()

        for fp, d, f in os.walk(read_path):
            for f_ in f:
                abs_filenames.append(os.path.join(fp, f_))

        abs_filenames=[r"D:\qdaq\rawdata\PM61191\Data\017700944N201986.zip"]
        for filename in abs_filenames:
            serial_no = os.path.split(filename)[-1].split(".")[0]

            if filename.endswith("tdms"):
                allrawdata = TdmsFile.read(filename)
            elif filename.endswith("h5"):
                allrawdata = h5py.File(filename, "r")
            else:
                continue
            rpml_resolver=dict()
            rpm_resolver=dict()
            try:
                for channelName in ["Sin","Cos"]:
                    rpml_resolver[channelName],rpm_resolver[channelName]=speed_calc_single_resolver(allrawdata["AIData"][channelName],sampleRate, coils)
                rpml_resolver2,rpm_resolver2=speed_calc_resolver2(allrawdata["AIData"]["Sin"],allrawdata["AIData"]["Cos"],sampleRate,coils)
                plt.figure(os.path.splitext(filename)[0])
                plt.plot(rpml_resolver2,rpm_resolver2,c="g",label="resolver2")
                plt.plot(rpml_resolver["Sin"],rpm_resolver["Sin"],c="r",label="Sin")
                plt.plot(rpml_resolver["Cos"],rpm_resolver["Cos"],c="b",label="Cos")
                plt.title("speedCurve")
                plt.xlabel("t/s")
                plt.ylabel("rpm")
                plt.legend()
                plt.savefig(os.path.join(pic_savepath,serial_no+".png"),dpi=500)
                plt.show()
                logging.info(filename+"计算成功")

            except Exception as e:
                logging.error(traceback.format_exc())
                logging.error(filename+"转速计算出错")
                traceback.print_exc()
                print(filename+"计算出现问题")
            print(filename)
    else:
        # 所有文件夹
        abs_dirs=list()
        for fp,d,f in os.walk(read_path):
            abs_dirs.append(fp)
        for abs_dir in abs_dirs:
            rpml_resolver=dict()
            rpm_resolver=dict()
            abs_sin_filename=os.path.join(abs_dir,"Sin")
            if os.path.exists(abs_sin_filename):
                try:
                    serial_no=os.path.split(os.path.split(abs_dir)[0])[-1]
                    # 存在Sin通道，开始计算
                    with open(abs_sin_filename, 'rb') as f:
                        sin = f.read()
                    sin = json.loads(str(zlib.decompress(sin), encoding="utf-8").replace("\'", "\""))["data"]
                    rpml_resolver["Sin"], rpm_resolver["Sin"] = speed_calc_single_resolver(
                        sin, sampleRate, coils)
                    with open(os.path.join(abs_dir,"Cos"), 'rb') as f:
                        cos = f.read()
                    cos = json.loads(str(zlib.decompress(cos), encoding="utf-8").replace("\'", "\""))["data"]
                    rpml_resolver["Cos"], rpm_resolver["Cos"] = speed_calc_single_resolver(
                        cos, sampleRate, coils)
                    rpml_resolver2, rpm_resolver2 = speed_calc_resolver2(np.array(sin),np.array(cos), sampleRate, coils)

                    plt.figure(serial_no)
                    plt.plot(rpml_resolver2, rpm_resolver2, c="g", label="resolver2")
                    plt.plot(rpml_resolver["Sin"], rpm_resolver["Sin"], c="r", label="Sin")
                    plt.plot(rpml_resolver["Cos"], rpm_resolver["Cos"], c="b", label="Cos")
                    plt.title("speedCurve")
                    plt.xlabel("t/s")
                    plt.ylabel("rpm")
                    plt.legend()
                    plt.savefig(os.path.join(pic_savepath, serial_no + ".png"), dpi=500)
                    plt.show()
                    logging.info(abs_dir + "计算成功")
                    print(abs_dir)
                except Exception as e:
                    logging.error(traceback.format_exc())
                    logging.error(abs_dir + "转速计算出错")
                    traceback.print_exc()
                    print(abs_dir + "计算出现问题")
            else:
                continue

        print(abs_dirs)


