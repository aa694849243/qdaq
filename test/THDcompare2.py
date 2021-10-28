from nptdms import TdmsFile
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.fft import fftfreq
import os, re


def readdata(filename, groupName, channelName):
    with TdmsFile.open(filename) as tdms_file:
        for channelName in channelName:
            allrawdata = tdms_file[groupName][channelName][:]
    return allrawdata


def freq_value(data, length, samples_per_second):
    fft_ni5v10hz = fft(data[:length])
    abs_fft_ni5v10hz = np.abs(fft_ni5v10hz)[1:length // 2]
    # abs_fft_ni5v10hz=fft_ni5v10hz[1:length//2]
    abs_fft_ni5v10hz = abs_fft_ni5v10hz / length * 2

    angle_fft_ni5v10hz=np.angle(fft_ni5v10hz,deg=True)[1:length // 2]

    # abs_fft_ni5v10hz=fft_ni5v10hz[1:len(ni5v10hz[:5120])//2]/5120*2


    freq = fftfreq(n=len(fft_ni5v10hz), d=1 / samples_per_second)[1:length // 2]


    return freq, abs_fft_ni5v10hz, fft_ni5v10hz[1:length // 2]/length * 2,angle_fft_ni5v10hz
    # return freq, abs_fft_ni5v10hz, fft_ni5v10hz[1:length // 2]/ length * 2,angle_fft_ni5v10hz


def THD_calculate(values, index):
    # numerator=np.sum(values**2)
    # numerator=np.sum(values**2)-values[int(index)]**2
    numerator = np.sum(values[int(index) + 1:] ** 2)  # 不计算小于等于该频率的值
    denominator = values[int(index)] ** 2
    return np.sqrt(numerator / denominator)


def delta_phi_calculate(values1, values2, index):
    index = int(index)


    angle = (np.arctan(np.imag(values1[index]) / np.real(values1[index])) - np.arctan(
        np.imag(values2[index]) / np.real(values2[index]))) * 180 / np.pi

    return angle


if __name__ == '__main__':

    # sine 1k和8k的数据来自于示波器
    # sine 10 20 10的数据来自于信号发生器

    path = "D:/ShirohaUmi/work_document/mcc/DHT"
    # files = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x)) and os.path.splitext(x)[1] == ".tdms" and os.path.splitext(x)[0].split('_')[1] == "square"]
    files = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x)) and os.path.splitext(x)[1] == ".tdms" ]
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

    # offset["ni_square_4v_10hz"]=0
    # offset["mcc_square_4v_10hz"]=1016
    # offset["ni_square_4v_20hz"]=0
    # offset["mcc_square_4v_20hz"]=1234

    # origin_offset[]

    length = 51200

    for file in files:

        name = os.path.splitext(file)[0]

        rawdata[name] = readdata(os.path.join(path, file), "NI" if file.startswith("ni") else "mcc", ["THD"])

        if not name in offset.keys():
            offset[name]=0

        if file.startswith("mcc"):
            rawdata[name] /= 1000
        # if name.startswith("ni"):
        #     rawdata[name][rawdata[name]>4]=4
        #     rawdata[name][rawdata[name]<-4]=-4
        # if name.startswith("mcc"):
        #     rawdata[name][rawdata[name] > 4.5] = 4.5
        #     rawdata[name][rawdata[name] < -4.5] = -4.5
        # after=np.where(rawdata[name]>4.0, 4,-4)
        # after=np.where(rawdata[name]>0, 4,-4)

        # after_freq[name],after_value[name]=freq_value(after,length,samples_per_second)
        # 采集到的信号进行fft变换的结果
        freq[name], abs_value[name], value[name] ,angle[name]= freq_value(rawdata[name][offset[name]:], length, samples_per_second)
        # 该信号对应的标准信号fft结果
        if name.split('_')[1] == "square":

            # origin_rawdata[name] 与rawdata[name]位于相同相位
            origin_rawdata[name] = np.where(rawdata[name] > 0, 4, -4)


            origin_freq[name], origin_abs_value[name], origin_value[name],origin_angle[name] = freq_value(origin_rawdata[name], length,
                                                                                       samples_per_second)

            index=int(int(re.findall('\d+', name.split('_')[-1])[0]) / (samples_per_second / length )-1)
            print("value[{}][{}]:{},origin_value[{}][{}]:{}".format(name,index+1, value[name][index], name,index+1,origin_value[name][index]))
            delta_phi[name] = delta_phi_calculate(value[name], origin_value[name],index)

        thd[name] = THD_calculate(abs_value[name],
                                  int(re.findall('\d+', name.split('_')[-1])[0]) / (samples_per_second / length) - 1)

    # for file in files:
    #     if file.startswith("ni"):
    #         name = os.path.splitext(file)[0]
    #         delta_phi[int(re.findall('\d+',name.split('_')[-1])[0])]=delta_phi_calculate(value[name],value[name.replace('ni','mcc')],int(re.findall('\d+',name.split('_')[-1])[0])/(samples_per_second/length))

    # 在基频上的相位差
    
    length_for_figure=len(freq['mcc_square_4v_10hz']) // 16

    print(thd)
    plt.figure(1)
    plt.plot(freq['mcc_square_4v_10hz'][:length_for_figure],
             abs_value['mcc_square_4v_10hz'][:length_for_figure], c='r', label='mcc_square_4v_10hz')
    # plt.figure(3)
    plt.plot(freq['ni_square_4v_10hz'][:length_for_figure],
             abs_value['ni_square_4v_10hz'][:length_for_figure], c='b', label='ni_square_4v_10hz')
    # plt.plot(after_freq['ni_square_4v_10hz'][:len(freq['mcc_square_4v_10hz'])//16],after_value['ni_square_4v_10hz'][:len(freq['mcc_square_4v_10hz'])//16],c='r',label='after_ni_square_4v_10hz')
    plt.legend()

    # plt.plot(freq['mcc_square_4v_10hz'][:len(freq['mcc_square_4v_10hz'])//16],np.subtract(abs_value['mcc_square_4v_10hz'][:len(freq['mcc_square_4v_10hz'])//16],abs_value['ni_square_4v_10hz'][:len(freq['mcc_square_4v_10hz'])//16]),label="mcc-ni")
    # plt.plot(rawdata['mcc_square_4v_10hz'][:length])

    plt.figure(2)
    plt.plot(rawdata['mcc_square_4v_20hz'][offset['mcc_square_4v_20hz']:offset['mcc_square_4v_20hz']+length], c='r', label='mcc_square_4v_20hz')
    plt.plot(rawdata['ni_square_4v_20hz'][offset['ni_square_4v_20hz']:offset['mcc_square_4v_20hz']+length], c='b', label='ni_square_4v_20hz')
    plt.plot(origin_rawdata['ni_square_4v_20hz'][offset['ni_square_4v_20hz']:offset['mcc_square_4v_20hz']+length], c='g', label='origin_square_4v_20hz')
    plt.legend()

    ifft_mcc_square_4v_10hz = ifft(value['mcc_square_4v_10hz'])
    ifft_ni_square_4v_10hz = ifft(value['ni_square_4v_10hz'])

    plt.figure(3)
    plt.plot(ifft_mcc_square_4v_10hz, c='r', label="ifft_mcc_square_4v_10hz")
    plt.plot(ifft_ni_square_4v_10hz, c='b', label="ifft_ni_square_4v_10hz")
    plt.legend()

    # plt.show()

    plt.figure(4)
    plt.plot()
    plt.plot(freq['mcc_square_4v_20hz'][:length_for_figure],
             angle['mcc_square_4v_20hz'][:length_for_figure], c='r', label='mcc_square_4v_20hz')
    # plt.figure(3)
    plt.plot(freq['ni_square_4v_20hz'][:length_for_figure],
             angle['ni_square_4v_20hz'][:length_for_figure], c='b', label='ni_square_4v_20hz')
    # plt.plot(after_freq['ni_square_4v_10hz'][:len(freq['mcc_square_4v_10hz'])//16],after_value['ni_square_4v_10hz'][:len(freq['mcc_square_4v_10hz'])//16],c='r',label='after_ni_square_4v_10hz')
    plt.plot(origin_freq['ni_square_4v_20hz'][:length_for_figure],
             origin_angle['ni_square_4v_20hz'][:length_for_figure], c='g', label='origin_square_4v_20hz')

    plt.legend()
    plt.figure(5)
    plt.plot()
    plt.plot(freq['mcc_square_4v_10hz'][:length_for_figure],
             angle['mcc_square_4v_10hz'][:length_for_figure], c='r', label='mcc_square_4v_10hz')
    # plt.figure(3)
    plt.plot(freq['ni_square_4v_10hz'][:length_for_figure],
             angle['ni_square_4v_10hz'][:length_for_figure], c='b', label='ni_square_4v_10hz')
    # plt.plot(after_freq['ni_square_4v_10hz'][:len(freq['mcc_square_4v_10hz'])//16],after_value['ni_square_4v_10hz'][:len(freq['mcc_square_4v_10hz'])//16],c='r',label='after_ni_square_4v_10hz')
    plt.plot(origin_freq['ni_square_4v_10hz'][:length_for_figure],
             origin_angle['ni_square_4v_10hz'][:length_for_figure], c='g', label='origin_square_4v_10hz')
    # plt.plot(after_freq['ni_square_4v_10hz'][:len(freq['mcc_square_4v_10hz'])//16],after_value['ni_square_4v_10hz'][:len(freq['mcc_square_4v_10hz'])//16],c='r',label='after_ni_square_4v_10hz')

    plt.legend()

    mcc_origin_4v_20hz=np.subtract(angle['mcc_square_4v_20hz'][:length_for_figure],origin_angle['ni_square_4v_20hz'][:length_for_figure])
    ni_origin_4v_20hz=np.subtract(angle['ni_square_4v_20hz'][:length_for_figure],
                origin_angle['ni_square_4v_20hz'][:length_for_figure])
    plt.figure(6)

    plt.plot(freq['mcc_square_4v_20hz'][:length_for_figure],
             mcc_origin_4v_20hz,c='r',label='mcc_origin_square_4v_20hz')

    plt.plot(freq['ni_square_4v_20hz'][:length_for_figure],
             ni_origin_4v_20hz,c='b',label='ni_origin_square_4v_20hz')
    plt.legend()
    # plt.show()

    plt.figure(7)
    # 不画幅值小于0.0001处的相位谱
    index=np.where(abs_value['mcc_square_4v_20hz'][:length_for_figure]>0.001)
    plt.plot(freq['mcc_square_4v_20hz'][index],mcc_origin_4v_20hz[index],c='r',label='mcc_origin_square_4v_20hz')
    plt.plot(freq['ni_square_4v_20hz'][index],ni_origin_4v_20hz[index],c='b',label='ni_origin_square_4v_20hz')
    plt.legend()
    # plt.show()
    # plt.plot(freq,abs_fft_ni5v10hz)
    # plt.scatter(freq,abs_fft_ni5v10hz,s=0.3)
    # plt.show()

    # plt.plot(ni5v10hz[:102400],color='r',label="ni5v10hz")
    # phi=1701-210+1+1
    # plt.plot(mcc5v10hz[phi:102400+phi],color='b',label="mcc5v10hz")
    # plt.legend()
    # plt.show()

    plt.figure(8)
    # 基频相位差
    a=["10",'20','50','100','200','300','400','500']
    # mcc_origin_list=['mcc_square_4v_'+x+'hz' for x in a]
    # ni_origin_list=['ni_square_4v_'+x+'hz' for x in a]
    mcc_origin=[delta_phi['mcc_square_4v_'+x+'hz'] for x in a]
    ni_origin=[delta_phi['ni_square_4v_'+x+'hz'] for x in a]
    # mcc_origin=delta_phi[mcc_origin_list]
    # ni_origin=delta_phi[ni_origin_list]
    # mcc_origin=[delta_phi['mcc_square_4v_10hz'],delta_phi['mcc_square_4v_20hz'],delta_phi['mcc_square_4v_50hz'],delta_phi['mcc_square_4v_1000hz'],delta_phi['mcc_square_4v_8000hz']]
    # ni_origin=[delta_phi['ni_square_4v_10hz'],delta_phi['ni_square_4v_20hz'],delta_phi['ni_square_4v_50hz'],delta_phi['ni_square_4v_1000hz'],delta_phi['ni_square_4v_8000hz']]

    length_for_delta_phi=len(a)
    plt.plot(a[:length_for_delta_phi],mcc_origin[:length_for_delta_phi],c='r',label="mcc-origin")
    plt.plot(a[:length_for_delta_phi],ni_origin[:length_for_delta_phi],c='b',label="ni-origin")
    plt.legend()

    plt.figure(9)
    plt.plot(rawdata['ni_square_4v_400hz'][offset['ni_square_4v_400hz']:offset['ni_square_4v_400hz'] + length//40], c='r',
             label='ni_square_4v_400hz')
    plt.plot(rawdata['ni_square_4v_4000hz'][offset['ni_square_4v_4000hz']+50:offset['ni_square_4v_4000hz']+50 + length//40], c='b',
             label='ni_square_4v_400hz_old')
    plt.legend()

    plt.figure(10)
    plt.plot(rawdata['mcc_square_4v_8000hz'][offset['mcc_square_4v_8000hz']:offset['mcc_square_4v_8000hz'] + length//40], c='r',
             label='mcc_square_4v_8000hz')
    plt.plot(rawdata['ni_square_4v_8000hz'][offset['ni_square_4v_8000hz']:offset['ni_square_4v_8000hz'] + length//40], c='b',
             label='ni_square_4v_8000hz')
    plt.legend()

    plt.figure(11)
    # 4v8k方波幅频谱
    plt.plot(freq['mcc_square_4v_8000hz'],
             abs_value['mcc_square_4v_8000hz'], c='r', label='mcc_square_4v_8000hz')
    # plt.figure(3)
    plt.plot(freq['ni_square_4v_8000hz']-1000,
             abs_value['ni_square_4v_8000hz'], c='b', label='ni_square_4v_8000hz')
    # plt.plot(after_freq['ni_square_4v_10hz'][:len(freq['mcc_square_4v_10hz'])//16],after_value['ni_square_4v_10hz'][:len(freq['mcc_square_4v_10hz'])//16],c='r',label='after_ni_square_4v_10hz')
    plt.legend()

    plt.figure(12)
    plt.plot(rawdata['mcc_sine_2v_8000hz'][offset['mcc_sine_2v_8000hz']:offset['mcc_sine_2v_8000hz'] + length//40], c='r',
             label='mcc_sine_2v_8000hz')
    plt.plot(rawdata['ni_sine_2v_8000hz'][offset['ni_sine_2v_8000hz']:offset['ni_sine_2v_8000hz'] + length//40], c='b',
             label='ni_sine_2v_8000hz')
    plt.legend()

    plt.figure(13)
    # 2v8k正弦波幅频谱
    plt.plot(freq['mcc_sine_2v_8000hz'],
             abs_value['mcc_sine_2v_8000hz'], c='r', label='mcc_sine_2v_8000hz')
    # plt.figure(3)
    plt.plot(freq['ni_sine_2v_8000hz']-1000,
             abs_value['ni_sine_2v_8000hz'], c='b', label='ni_sine_2v_8000hz')
    # plt.plot(after_freq['ni_square_4v_10hz'][:len(freq['mcc_square_4v_10hz'])//16],after_value['ni_square_4v_10hz'][:len(freq['mcc_square_4v_10hz'])//16],c='r',label='after_ni_square_4v_10hz')
    plt.legend()


    plt.show()
