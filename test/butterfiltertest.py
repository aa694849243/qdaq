import matplotlib.pyplot as plt
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
    sin_1000hz = sin_wave(1, 1000, fs, 0, 1)
    # sin_800hz = sin_wave(1, 800, fs, 0, 1)
    # sin_900hz = sin_wave(1, 900, fs, 0, 1)
    # sin_1100hz = sin_wave(1, 1100, fs, 0, 1)
    sin_2000hz = sin_wave(1, 2000, fs, 0, 1)
    # sin_bf = sin_1000hz + sin_2000hz+sin_800hz+sin_900hz+sin_1100hz
    sin_bf = sin_1000hz + sin_2000hz

    coff_list = [1, 1.1, 1.2, 1.28, 1.3, 1.4, 1.5, 1.56, 1.6, 1.7, 1.8, 1.9, 2, 2.1]
    rms_dict = dict()
    for coff in coff_list:
        sin_af = butter_filter(sin_bf, [1000 * coff], fs)
        fft_abs=np.abs(np.fft.rfft(sin_af))/len(sin_af)*2

        rms_dict[coff] = rms(sin_af[int(0.1 * fs):int(0.9 * fs)]) * np.sqrt(2)
        plt.figure(str(coff))
        # plt.plot(sin_bf,c="r",label="bf")
        # plt.plot(sin_af, c="b", label="af")
        plt.plot(np.fft.rfftfreq(len(sin_af),1/fs)[:2500],fft_abs[:2500])
        plt.xlabel("freq")

    plt.figure("al")
    plt.plot(rms_dict.keys(), rms_dict.values())
    plt.legend()
    plt.show()
