import time
import h5py
from nptdms import TdmsFile
import numpy as np
from scipy import stats, fftpack
import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import butter, filtfilt, lfilter
from scipy.signal import resample
from scipy.interpolate import interp1d
import math
import numpy as np
import traceback
import os
import sys
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



# 下面这四个值可能要根据电机以及采集到的信号进行调整，
lowLevel = 6  # 下极值点在 0~lowLevel 度之间。
highLevel = 35  # 上极值点。Hilbert变换出来的上极值点以上的角度不可用，扔掉后靠下面的点拟合后插值补上。
# 寻找跳点的时候，要求当前跳点以及下一个跳点的值
min_value = 20
max_value = 20

allrawdata = read_raw_data("D:/qdaq/debug/210904-1/error_data/017700948N900018_210903191408.h5", ["Sin","Cos"], "hdf5")
# allrawdata = read_raw_data("D:/qdaq/debug/210904-1/error_data/ref_no_error/017700948N900072_210903212824.h5", ["Sin"], "hdf5")
speed_sin=np.array(allrawdata["Sin"][196250:204800])
speed_cos=np.array(allrawdata["Cos"][196250:204800])
print(1)
raw_sin=speed_sin
speed_sin = butter_filter(speed_sin, [int(10167*1.28)], 102400)
# speed_sin=np.array(allrawdata["Sin"][:16384])
hSin = fftpack.hilbert(speed_sin)
raw_hSin = fftpack.hilbert(raw_sin)

# 舍弃前10个点，后10个点
# speed_sin=speed_sin[10:len(speed_sin)-10]
# hSin=hSin[10:len(hSin)-10]
# left_index+=10

envSin = np.sqrt(speed_sin ** 2 + hSin ** 2)
raw_envSin = np.sqrt(raw_sin ** 2 + raw_hSin ** 2)



amp = np.max(envSin)
raw_amp=np.max(raw_envSin)
angle0 = np.arcsin(envSin / amp) * 180 / np.pi
raw_angle0 = np.arcsin(raw_envSin / raw_amp) * 180 / np.pi


# plt.figure("envSin")
# plt.plot(angle0)
# plt.show()

cutted_angle = angle0[(angle0 > lowLevel) & (angle0 < highLevel)]
raw_cutted_angle = raw_angle0[(raw_angle0 > lowLevel) & (raw_angle0 < highLevel)]
# cutted_loc为所有点的索引，并非是第一个点的索引
# 注意数组中存的数据的意义，
cutted_loc = np.where((angle0 > lowLevel) & (angle0 < highLevel))[0]  # cutted_angle在angle0中的索引
raw_cutted_loc = np.where((raw_angle0 > lowLevel) & (raw_angle0 < highLevel))[0]  # cutted_angle在angle0中的索引

skipP = np.where(np.diff(cutted_loc) > 1)[0]
raw_skipP = np.where(np.diff(raw_cutted_loc) > 1)[0]


# 状态机
status = -1

# 该list中的数据为该帧sin数据中的索引，为小数
skipP = np.where(np.diff(cutted_loc) > 1)[0]  # 跳点在cutted_loc数组中的索引

# 状态机
status = -1

# 该list中的数据为该帧sin数据中的索引，为整数
zero_points = []

# hilbert变换后开头的一部分数据不能用,转速为10000rpm时，每条鱼内有102400/(10000/60*4*2)=76.8个点
# 所以在(status == 4 or status == -1) 判断中加上cutted_loc[skipP[i]]>10
# 如果转速或者采样率发生变化，这个点应该会发生改变

plt.figure("rawdata")
plt.plot(speed_sin,c="r")
plt.plot(raw_sin,c="b")
# plt.plot(speed_cos)
plt.figure("angle")
plt.plot(angle0,c="r")
plt.plot(raw_angle0,c="b")
plt.figure("cutted")
plt.plot(cutted_loc,cutted_angle,c="r")
plt.plot(raw_cutted_loc,raw_cutted_angle,c="b")
plt.figure("hSin")
plt.plot(hSin)
plt.figure("envSin")
plt.plot(envSin)
plt.show()



for i in range(np.size(skipP)):
    # 上左最后一个跳点 在跳点位置可能有噪声，要求该跳点的下一个跳点

    if (status == 4 or status == -1) and cutted_angle[skipP[i]] > max_value and cutted_loc[
        skipP[i]] > 10 and (i == np.size(skipP) - 1 or cutted_angle[skipP[i + 1]] < min_value):
        # if (status == 4 or status == -1) and cutted_angle[skipP[i]] > max_value  \
        #     and (i == np.size(skipP) - 1 or cutted_angle[skipP[i + 1]] < min_value):
        # 找到紧挨跳点右边的点在cutted_loc中的索引
        upLeft_index = skipP[i] + 1
        status = 1
        continue

    # 下左第一个跳点
    if status == 1 and cutted_angle[skipP[i - 1]] > max_value and (
            i == np.size(skipP) - 1 or cutted_angle[skipP[i]] < min_value):
        # 找到跳点在cutted_loc中的索引
        downLeft_index = skipP[i]
        status = 2
        kb = np.polyfit(cutted_loc[upLeft_index:downLeft_index],
                        cutted_angle[upLeft_index:downLeft_index], 1)
        left_zero_point = -kb[1] / kb[0]

    # 找到右下最后一个跳点
    if status == 2 and cutted_angle[skipP[i]] < min_value and (
            i == np.size(skipP) - 1 or cutted_angle[skipP[i + 1]] > max_value):
        # 找到紧挨跳点右边的点在cutted_loc中的索引
        downRight_index = skipP[i] + 1

        status = 3
        continue

    # 找到右上第一个跳点
    if status == 3 and cutted_angle[skipP[i]] > max_value and cutted_angle[skipP[i - 1]] < min_value:
        # 找到跳点在cutted_loc中的索引
        upRight_index = skipP[i]
        status = 4
        # downLeftP 与 upLeftP 两个点之间的采样点，形成线段。延长线段到0轴，得到0点

        # # 线性拟合，返回系数k,b
        try:
            kb = np.polyfit(cutted_loc[downRight_index:upRight_index],
                            cutted_angle[downRight_index:upRight_index], 1)
        except Exception:
            time.sleep(0.1)
        right_zero_point = -kb[1] / kb[0]

        ave = (left_zero_point + right_zero_point) / 2
        zero_points.append(ave)
        if (i == np.size(skipP) - 1 or cutted_angle[skipP[i + 1]] < 20):  # 如果同时也是上左最后一个跳点
            upLeft_index = skipP[i] + 1
            status = 1


print(1)