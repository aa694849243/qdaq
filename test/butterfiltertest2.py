import random
import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, lfilter
from scipy.signal import resample
from scipy.interpolate import interp1d


def sin_wave(A, f, fs, phi, t):
    '''
    :params A:    振幅
    :params f:    信号频率
    :params fs:   采样频率
    :params phi:  相位
    :params t:    时间长度
    '''
    # 若时间序列长度为 t=1s,
    # 采样频率 fs=1000 Hz, 则采样时间间隔 Ts=1/fs=0.001s
    # 对于时间序列采样点个数为 n=t/Ts=1/0.001=1000, 即有1000个点,每个点间隔为 Ts
    Ts = 1 / fs
    n = t / Ts
    n = np.arange(n)
    y = A * np.sin(2 * np.pi * f * n * Ts + phi * (np.pi / 180))
    return y


def butter_filter(signal, wn, fs, order=3, btype='lowpass'):

    # order最大取9
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


def rms(data):
    return np.sqrt(np.sum(np.square(data)) / len(data))


if __name__ == "__main__":
    fs = 102400
    sin_bf=np.zeros(1*fs)
    freq_list=np.linspace(100,2000,951,endpoint=True,dtype="i")
    freq_list=np.linspace(100,2000,191,endpoint=True,dtype="i")
    for freq in freq_list:
        sin_bf+=sin_wave(1,freq,fs,0,1)

    coff_list = [1, 1.1, 1.2, 1.28, 1.3, 1.4, 1.5, 1.56, 1.6, 1.7, 1.8, 1.9, 2, 2.1]
    coff_list = [1]
    rms_dict = dict()
    max_value_dict=dict()
    order_list=[3,4,5,6,7]
    # order_list=[3]
    time_dict=dict()
    for i,order in enumerate(order_list):
        for coff in coff_list:

            # time_start=time.time()
            # for count in range(100):
            #     sin_af = butter_filter(sin_bf, [1000 * coff], fs,order=order)
            # time_dict[order]=time.time()-time_start
            sin_af = butter_filter(sin_bf, [1000 * coff], fs,order=order)


            fft_abs=np.abs(np.fft.rfft(sin_af))/len(sin_af)*2

            rms_dict[coff] = rms(sin_af[int(0.1 * fs):int(0.9 * fs)]) * np.sqrt(2)
            max_value_dict[order]=fft_abs[freq_list]
            plt.figure("fft")
            # plt.title("order={}")
            # plt.plot(sin_bf,c="r",label="bf")
            # plt.plot(sin_af, c="b", label="af")
            plt.plot(np.fft.rfftfreq(len(sin_af),1/fs)[:2500],fft_abs[:2500],c=list(mcolors.BASE_COLORS.keys())[i],label=str(order))
            plt.legend()
            plt.figure("al")
            index=np.sort(np.argsort(fft_abs)[-len(freq_list):])
            plt.plot(freq_list,fft_abs[freq_list],c=list(mcolors.BASE_COLORS.keys())[i],label=str(order))
            plt.xlabel("freq")
            plt.grid()

    print(time_dict)
    plt.legend()
    plt.figure("al")
    plt.plot(rms_dict.keys(), rms_dict.values())
    plt.legend()
    plt.show()
