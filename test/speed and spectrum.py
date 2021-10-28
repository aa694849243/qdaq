import json
import os
import time

import numpy as np
from nptdms import TdmsFile
from scipy.fft import fft, fftfreq
from nptdms import TdmsFile
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.signal as sig

from qdaq.tools.tools import get_timefrequency_colormap
# from utils import read_raw_data

filename1 = "D:\\qdaq\\对比\\fluctuation_norsp\\fluctuation_norsp8k_test01_3mm_8k_210802064140_210820023006.json"
# filename3 = "D:\\qdaq\\对比\\v2v3\\PM61191-2vibs-resolver2-v2.json"

with open(filename1, encoding="utf-8") as f:
    data1 = json.load(f)

orderSpectrum1 = data1["resultData"][0]["dataSection"][0]["twodOS"][0]


def speed_calibration(vib, min_speed,max_speed, order, fs):
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
    # vib_fft = fft(vib)[1:n // 2]
    # freq = fftfreq(n, d=1 / fs)[1:n // 2]
    vib_fft=np.fft.rfft(vib)
    freq=np.fft.rfftfreq(n,d=1/fs)

    # frequency range for speed
    # left = min_speed / 60 * (order - 0.5)
    # right = max_speed / 60 * (order + 0.5)
    left = min_speed / 60 * (order)
    right = max_speed / 60 * (order)
    idx = (freq >= left) & (freq <= right)

    # find target frequency
    target = np.argmax(np.abs(vib_fft[idx]))
    print(target)
    # target_min = np.argmin(vib_fft[idx])
    speed_cali = freq[idx][target] / order * 60
    # speed_cali = (freq[idx][target] / order * 60+freq[idx][target_min] / order * 60)/2
    return speed_cali


def get_order_spectrum2(vib, d_revolution, n_revolution, max_order=200, normalize=False):
    '''
    draw order spectrum from vibration data for constant speed
    ***************************************************
    # 要保证足够计算一次
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
    print("d_revolution:{}".format(d_revolution))
    print("n:{}".format(n))
    o = np.fft.rfftfreq(n, d=d_revolution)
    print("o[1]-o[0]:{}".format(o[1]-o[0]))
    roi = o < max_order
    o=o[roi]
    # o=o[:int(max_order*d_revolution)]
    print("len(o):{}".format(len(o)))


    step = int(n / 4)

    # result varibale
    yy = []

    if len(vib)<=n:
        raise Exception("长度不够做一次傅式变换")
    else:
        count=(len(vib)-n)//step+1

    for i in range(count):
        frame = vib[i * step:i * step + n]
        if len(frame)<n:
            raise Exception("长度不够做一次傅式变换")
        # fy = abs(np.fft.rfft(np.array(frame) * np.kaiser(len(frame),9),n)) / len(frame)*2
        fy = np.abs(np.fft.rfft(np.array(frame),n)) / len(frame)*2*np.sqrt(len(frame))/np.sqrt(n)
        fy[0]/=2
        fy*=np.sqrt(2)/2
        yy.append(fy[roi])
        # yy.append(fy[:int(max_order*d_revolution)])

    # if normalize == True:
    #     specy = np.mean(yy, axis=0) / max(np.mean(yy, axis=0))
    # else:
    #     specy = np.mean(yy, axis=0)
    #     # specy=np.sqrt(np.sum(np.power(yy,2),axis=0)/)

    rms1=np.sqrt(np.sum(np.power(vib,2))/len(vib))

    os_temp=np.sqrt(np.sum(np.power(yy,2),axis=0)/len(yy))
    os_temp[0]*=np.sqrt(2)
    rms2=np.sqrt(np.sum(np.power(os_temp,2)))

    return o, yy



#################      function test        ####################

# ------------------- speed calibration test -------------------- #
def test1():
    path = 'D:/qdaq/rawdata/test01'
    file = os.path.join(path, 'test01_3mm_24k_210802064516.tdms')

    # files=[x for x in os.walk(path)]
    files = [x for x in os.listdir(path) if
             os.path.isfile(os.path.join(path, x)) and os.path.splitext(x)[
                 1] == ".tdms" and x.startswith("test01_3")]
    start_speed=4000
    end_speed=5000


    fs = 102400
    speed_frame_length = 10240 * 10
    speed_frame_length = 8192
    path='D:/qdaq/rawdata/xiaodianji_umik2/210917'
    files=["surround-mic_210917085347.tdms"]
    path=r"D:\qdaq\rawdata\xiaodianji\21082609"
    files=["105OK2-1_210826092446.tdms"]

    fs=102400
    start_speed=90000
    end_speed=107000
    order=13


    speed_dict = defaultdict(list)
    spectromgram_dict = dict()
    o_list=list()
    r_list=list()
    for file in files:
        # ref_speed = int(file.split("_")[2][:-1]) * 1000
        ref_speed=3150
        data = TdmsFile.read(os.path.join(path, file))
        # vib = data['AIData']['Umic1'].data
        vib = data['AIData']['Mic1'].data
        plt.plot(vib)
        plt.show()

        # spectromgram_dict[ref_speed] = sig.spectrogram(vib, fs=102400)
        for i in range(len(vib) // speed_frame_length):
            frame=vib[i * speed_frame_length:(i + 1) * speed_frame_length]
            if i==18:
                time.sleep(0.1)
            speed = speed_calibration(frame,start_speed,end_speed, order, fs=fs)
            # speed=3150

            speed_dict[ref_speed].append(speed)
            # speed=4917
            dr=(speed/60)/fs
            o, r = get_order_spectrum2(frame, dr, 32, max_order=25, normalize=False)
            o_list.append(o)
            r_list.extend(r)
            # r_list.append(r)

        get_timefrequency_colormap(vib,n_frame=speed_frame_length,fs=fs,roi_freq=[0,2000])
    for key in speed_dict.keys():
        plt.figure(key)
        plt.plot(speed_frame_length/fs*np.arange(len(speed_dict[key])),speed_dict[key])
        # plt.axhline(key)
        # plt.figure(key)
        # plt.pcolormesh(spectromgram_dict[key][1],spectromgram_dict[key][0],spectromgram_dict[key][2],cmap="jet")
    # key = 8000
    # plt.figure(key)
    # plt.plot(speed_dict[key])
    # plt.axhline(key)

    plt.show()
    dr=(ref_speed/60)/fs
    # dr = 1/32
    plt.figure("os")
    # o, r = get_order_spectrum(vib, dr, 32, max_order=50, normalize=True)

    # o2,r2=get_order_spectrum2(vib,2500/60/102400,32,max_order=50,normalize=True)
    # plt.plot(o2,r2,c='r')
    print("len(o_list[0]):{}".format(len(o_list[0])))
    print("r_list:{}".format(len(r_list)))
    o_list=np.arange(len(o_list[0]))*0.03125
    r_list=np.sqrt(np.sum(np.power(r_list,2),axis=0)/len(r_list))
    plt.plot(o_list,r_list,c='b')

    plt.show()


"""

def test2():
    path = "D:/qdaq/Simu/"
    file = "byd-alltests.h5"
    speed_frame_length = int(102400*1)
    fs = 102400
    allVib = read_raw_data(os.path.join(path, file), ["Vib1"], "hdf5")["Vib1"]
    speed_dict = defaultdict(list)

    startTimeList = [
        3.53248046875,
        26.827861328125,
        46.40154296875,
        65.1748046875,
        83.86537109375,
        102.808154296875,
        117.405380859375
    ]
    endTimeList = [
        15.5334375,
        38.828457031250004,
        56.402001953125,
        75.17515625,
        93.86541015625001,
        110.808349609375,
        119.405400390625
    ]
    orderList = [33, 72, 67, 26.88, 10.06, 33.1, 30.25]
    refSpeedList = [3000, 4800, 6000, 8000, 10000, 12000, 13000]
    refSpeedList = [3000]

    o_list=list()
    r_list=list()
    step=speed_frame_length//10
    for s in range(len(refSpeedList)):
        ref_speed = refSpeedList[s]
        ref_order = orderList[s]
        vib = allVib[int(startTimeList[s] * 102400):int(endTimeList[s] * 102400)]
        for i in range((len(vib)-speed_frame_length)//step):
            # speed = speed_calibration(vib[i*speed_frame_length-200 if i*speed_frame_length >200 else  0:(i+1)*speed_frame_length], ref_speed, 13, fs=fs)
            frame=vib[i * step:(i) * step+speed_frame_length]
            speed = speed_calibration(frame,2800,3500, ref_order, fs=fs)
            speed_dict[ref_speed].append(speed)
            speed=3000
            dr=(speed/60)/fs
            o, r = get_order_spectrum2(frame, dr, 32, max_order=50, normalize=False)
            o_list.append(o)
            r_list.append(r)
            print(i)
    for key in speed_dict.keys():
        plt.figure(key)
        plt.plot(speed_dict[key])
        plt.axhline(key)
    plt.figure("os")
    o_list=np.arange(len(o_list[0]))*0.03125
    # r_list=np.mean(r_list,axis=0)
    r_list=np.mean(r_list,axis=0)/np.max(np.mean(r_list,axis=0))
    plt.plot(o_list,r_list,c='b')
    plt.show()



def test3():
    path = 'D:/qdaq/对比/xuanbiandianji'
    file = os.path.join(path, 'test01_3mm_24k_210802064516.tdms')

    # files=[x for x in os.walk(path)]
    files = [x for x in os.listdir(path) if
             os.path.isfile(os.path.join(path, x)) and os.path.splitext(x)[
                 1] == ".tdms" and x.startswith("test01_3")]
    # files=["test01_3mm_76k_210802065445.tdms"]
    # files=["test01_3mm_54k_210802065030.tdms"]
    files=["demo1_210816064115.h5"]
    fs = 102400
    speed_frame_length = 10240 * 4
    speed_frame_length = 8192

    speed_dict = defaultdict(list)
    spectromgram_dict = dict()
    o_list=list()
    r_list=list()
    for file in files:
        ref_speed = 2500
        # data = TdmsFile.read(os.path.join(path, file))
        data=read_raw_data(os.path.join(path, file),["Mic"],"hdf5")
        vib=data["Mic"][102400*20:]

        plt.figure("mic")
        plt.plot(vib)
        plt.show()
        # spectromgram_dict[ref_speed] = sig.spectrogram(vib, fs=102400)
        for i in range(len(vib) // speed_frame_length):
            # if i == 79 or i == 102 or i == 127:
            #     # if i ==80 or i==103 or i==128:
            #     time.sleep(0.1)
            # speed = speed_calibration(vib[i*speed_frame_length-200 if i*speed_frame_length >200 else  0:(i+1)*speed_frame_length], ref_speed, 13, fs=fs)
            frame=vib[i * speed_frame_length:(i + 1) * speed_frame_length]
            speed = speed_calibration(frame,2400,2600, 7.75, fs=fs)
            speed_dict[ref_speed].append(speed)
            dr=(speed/60)/fs
            o, r = get_order_spectrum2(frame, dr, 32, max_order=30, normalize=False)
            o_list.append(o)
            r_list.append(r)
            print(i)


        get_timefrequency_colormap(vib,n_frame=8192)
    for key in speed_dict.keys():
        plt.figure(key)
        plt.plot(speed_dict[key])
        plt.axhline(key)
        # plt.figure(key)
        # plt.pcolormesh(spectromgram_dict[key][1],spectromgram_dict[key][0],spectromgram_dict[key][2],cmap="jet")
    # key = 8000
    # plt.figure(key)
    # plt.plot(speed_dict[key])
    # plt.axhline(key)

    plt.show()
    dr=(ref_speed/60)/fs
    # dr = 1/32
    plt.figure("os")
    # o, r = get_order_spectrum(vib, dr, 32, max_order=50, normalize=True)

    # o2,r2=get_order_spectrum2(vib,dr,32,max_order=50,normalize=True)
    # plt.plot(o2,r2,c='r')
    o_list=np.arange(len(o_list[0]))*0.03125
    r_list=np.mean(r_list,axis=0)
    # r_list=np.mean(r_list,axis=0)/np.max(np.mean(r_list,axis=0))
    plt.plot(o_list,r_list,c='b')
    plt.show()
    print(1)

"""
if __name__ == '__main__':
    test1()
    # test2()
    # test3()
