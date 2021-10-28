from nptdms import TdmsFile
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.fft import fftfreq
import os, re


def readdata(filename, groupName, channelName):
    print(filename)
    with TdmsFile.open(filename) as tdms_file:
        for channelName in channelName:
            allrawdata = tdms_file[groupName][channelName][:]
    print("over")
    return allrawdata


def freq_value(data, length, samples_per_second):
    fft_ni5v10hz = fft(data[:length])
    abs_fft_ni5v10hz = np.abs(fft_ni5v10hz)[1:length // 2]
    # abs_fft_ni5v10hz=fft_ni5v10hz[1:length//2]
    abs_fft_ni5v10hz = abs_fft_ni5v10hz / length * 2

    angle_fft_ni5v10hz=np.angle(fft_ni5v10hz,deg=True)

    # abs_fft_ni5v10hz=fft_ni5v10hz[1:len(ni5v10hz[:5120])//2]/5120*2

    freq = fftfreq(n=len(fft_ni5v10hz), d=1 / samples_per_second)[1:length // 2]


    return freq, abs_fft_ni5v10hz, fft_ni5v10hz,angle_fft_ni5v10hz


def THD_calculate(values, index):
    # numerator=np.sum(values**2)
    # numerator=np.sum(values**2)-values[int(index)]**2
    numerator = np.sum(values[int(index) + 1:] ** 2)  # 不计算小于等于该频率的值
    denominator = values[int(index)] ** 2
    return np.sqrt(numerator / denominator)


def delta_phi_calculate(values1, values2, index):
    index = int(index)

    print("values1[{}]:{},values2[{}]:{}".format(index, values1[index], index, values2[index]))
    angle = (np.arctan(np.imag(values1[index]) / np.real(values1[index])) - np.arctan(
        np.imag(values2[index]) / np.real(values2[index]))) * 360 / np.pi

    return angle


if __name__ == '__main__':

    # sine 1k和8k的数据来自于示波器
    # sine 10 20 10的数据来自于信号发生器

    path = "D:/ShirohaUmi/work_document/mcc/aqsrt"
    files = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x)) and os.path.splitext(x)[1] == ".tdms"]
    thd_d = dict()
    rawdata = dict()
    freq = dict()
    abs_value = dict()
    value = dict()

    origin_rawdata = dict()
    origin_freq = dict()
    origin_abs_value = dict()
    origin_value = dict()
    origin_angle=dict()

    thd = dict()
    after_freq = dict()
    after_value = dict()
    delta_phi = dict()
    angle=dict()
    samples_per_second = 51200

    # 方波信号的offset，对比相谱图
    offset=dict()
    origin_offset=dict()

    offset["mcc_ni_aqsrt_0"]=361000-247500+1
    offset["ni_mcc_aqsrt_0"]=361000-408
    offset["ni_square_4v_20hz"]=0
    offset["mcc_square_4v_20hz"]=1234

    # origin_offset[]

    length = 51200

    for file in files:

        name = os.path.splitext(file)[0]

        if not name in offset.keys():
            offset[name]=0

        rawdata[name] = readdata(os.path.join(path, file), "AIData" , ['Vibration'])

        # if name.startswith("mcc"):
        #     rawdata[name]/=9.85

        freq[name], abs_value[name], value[name] ,angle[name]=freq_value(rawdata[name][offset[name]:],length,samples_per_second)


    # print(thd)
    # plt.figure(1)
    # plt.plot(freq['mcc_square_4v_10hz'][:len(freq['mcc_square_4v_10hz']) // 16],
    #          abs_value['mcc_square_4v_10hz'][:len(freq['mcc_square_4v_10hz']) // 16], c='r', label='mcc_square_4v_10hz')
    # # plt.figure(3)
    # plt.plot(freq['ni_square_4v_10hz'][:len(freq['mcc_square_4v_10hz']) // 16],
    #          abs_value['ni_square_4v_10hz'][:len(freq['mcc_square_4v_10hz']) // 16], c='b', label='ni_square_4v_10hz')
    # # plt.plot(after_freq['ni_square_4v_10hz'][:len(freq['mcc_square_4v_10hz'])//16],after_value['ni_square_4v_10hz'][:len(freq['mcc_square_4v_10hz'])//16],c='r',label='after_ni_square_4v_10hz')
    # plt.legend()


    # plt.figure(num=2,figsize=(100,60),dpi=20)
    plt.figure(2)
    plt.plot(rawdata['mcc_ni_aqsrt_0'][offset['mcc_ni_aqsrt_0']:offset['mcc_ni_aqsrt_0']+10**6], c='r', label='mcc_ni_aqsrt_0')
    plt.plot(rawdata['ni_mcc_aqsrt_0'][offset['ni_mcc_aqsrt_0']:offset['ni_mcc_aqsrt_0']+10**6], c='b', label='ni_mcc_aqsrt_0')
    # plt.plot(rawdata['mcc_ni_aqsrt_0'], c='r', label='mcc_ni_aqsrt_0')
    # plt.plot(rawdata['ni_mcc_aqsrt_0'], c='b', label='ni_mcc_aqsrt_0')
    plt.legend()

    rawdata_for_fft=dict()
    rawdata_for_fft['mcc_ni_aqsrt_0']=rawdata['mcc_ni_aqsrt_0'][97200:97500]
    rawdata_for_fft['ni_mcc_aqsrt_0']=rawdata['ni_mcc_aqsrt_0'][97200:97500]

    value=dict()
    for key in rawdata_for_fft.keys():
        value[key]=fft(rawdata_for_fft[key],len(rawdata_for_fft[key]),samples_per_second)
    # freq_value(raw)



    plt.show()

