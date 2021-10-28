from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile
from scipy.fft import fft, fftfreq
import os

from qdaq.tools.tools import get_timefrequency_colormap

angle=list()
vib_ifft=list()
def resample1(vib, min_speed,max_speed, order, fs,points):
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

    n = len(vib)
    vib_fft = np.fft.rfft(vib)
    freq = np.fft.rfftfreq(n, d=1 / fs)

    # frequency range for speed
    left = min_speed / 60 * (order - 0.5)
    right = max_speed / 60 * (order + 0.5)
    idx = (freq >= left) & (freq <= right)

    # find target frequency
    target = np.argmax(np.abs(vib_fft[idx]))
    # target_min = np.argmin(vib_fft[idx])
    speed_cali = freq[idx][target] / order * 60
    array_for_ifft=np.zeros(len(vib_fft),dtype="complex")
    index=np.argmax(idx)+target
    array_for_ifft[index-points//2:index+points//2+1]=vib_fft[index-points//2:index+points//2+1]
    # array_for_ifft[index-1:index+1+1]=vib_fft[index-1:index+1+1]
    vib_ifft_temp=np.fft.irfft(array_for_ifft)/2*len(array_for_ifft)
    vib_ifft.extend(vib_ifft_temp[:len(vib_ifft_temp)//4])
    # print(len(vib_ifft_temp))
    # print(len(vib_ifft))
    # angle.append(np.angle(vib_fft[idx][target]))
    # speed_cali = (freq[idx][target] / order * 60+freq[idx][target_min] / order * 60)/2
    return speed_cali


def test1():
    path = 'D:/qdaq/rawdata/test01'
    file = os.path.join(path, 'test01_3mm_24k_210802064516.tdms')

    # files=[x for x in os.walk(path)]
    files = [x for x in os.listdir(path) if
             os.path.isfile(os.path.join(path, x)) and os.path.splitext(x)[
                 1] == ".tdms" and x.startswith("test01_3")]
    # files=["test01_3mm_76k_210802065445.tdms"]
    # files=["test01_3mm_54k_210802065030.tdms"]
    files=["test01_3mm_8k_210802064140.tdms"]
    fs = 102400
    speed_frame_length = 10240 * 4
    speed_frame_length = 8192

    speed_dict = defaultdict(list)
    spectromgram_dict = dict()
    o_list=list()
    r_list=list()
    step=speed_frame_length//4
    for file in files:
        ref_speed = int(file.split("_")[2][:-1]) * 1000
        data = TdmsFile.read(os.path.join(path, file))

        vib = data['AIData']['Mic'].data
        count=len(vib)//step-1
        for i in range(count):
            frame=vib[i * step:i*step+speed_frame_length]
            resample1(frame,7500,9500,13,fs,1)

    plt.figure("vib_ifft")
    plt.plot(vib_ifft)


    plt.show()

if __name__ == "__main__":
    test1()
    print(1)