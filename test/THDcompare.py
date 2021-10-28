from nptdms import TdmsFile
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft,ifft
from scipy.fft import fftfreq

def readAllRawData(filename,groupName,channelName):

    with TdmsFile.open(filename) as tdms_file:
        for channelName in channelName:
            allrawdata=tdms_file[groupName][channelName][:]
    return allrawdata

def frq_value(data,length,samples_per_second):
    fft_ni5v10hz = fft(data[:length])
    abs_fft_ni5v10hz = np.abs(fft_ni5v10hz)[1:length // 2]
    # abs_fft_ni5v10hz=fft_ni5v10hz[1:length//2]
    abs_fft_ni5v10hz = abs_fft_ni5v10hz / length * 2

    # abs_fft_ni5v10hz=fft_ni5v10hz[1:len(ni5v10hz[:5120])//2]/5120*2

    freq = fftfreq(n=len(fft_ni5v10hz), d=1 / samples_per_second)[1:length // 2]

    return freq,abs_fft_ni5v10hz

def THD_calculate(values,index):
    # numerator=np.sum(values**2)
    # numerator=np.sum(values**2)-values[int(index)]**2
    numerator=np.sum(values[int(index)+1:]**2) #不计算小于等于该频率的值
    denominator=values[int(index)]**2
    return np.sqrt(numerator/denominator)



if __name__ == '__main__':

    # 1k和8k的数据来自于示波器
    # 10 20 50的数据来自于信号发生器

    ni5v10hz=readAllRawData("D:/ShirohaUmi/work_document/mcc/DHT/ni_sine_5v_10hz.tdms","NI",["THD"])
    samples_per_second=51200

    length=10240

    freq_ni5v10hz,value_ni5v10hz=frq_value(ni5v10hz,length,samples_per_second)
    ni5v20hz=readAllRawData("D:/ShirohaUmi/work_document/mcc/DHT/ni5v20hz.tdms","NI",["THD"])
    ni5v50hz=readAllRawData("D:/ShirohaUmi/work_document/mcc/DHT/ni5v50hz.tdms","NI",["THD"])
    ni2v1khz=readAllRawData("D:/ShirohaUmi/work_document/mcc/DHT/ni2v1khz.tdms","NI",["THD"])
    ni2v8khz=readAllRawData("D:/ShirohaUmi/work_document/mcc/DHT/ni2v8khz.tdms","NI",["THD"])

    mcc5v10hz=readAllRawData("D:/ShirohaUmi/work_document/mcc/DHT/mcc5v10hz.tdms","mcc",["THD"])/1000
    mcc5v20hz=readAllRawData("D:/ShirohaUmi/work_document/mcc/DHT/mcc5v20hz.tdms","mcc",["THD"])/1000
    mcc5v50hz=readAllRawData("D:/ShirohaUmi/work_document/mcc/DHT/mcc5v50hz.tdms","mcc",["THD"])/1000
    mcc2v1khz=readAllRawData("D:/ShirohaUmi/work_document/mcc/DHT/mcc2v1khz.tdms","mcc",["THD"])/1000
    mcc2v8khz=readAllRawData("D:/ShirohaUmi/work_document/mcc/DHT/mcc2v8khz.tdms","mcc",["THD"])/1000




    freq_ni5v20hz,value_ni5v20hz=frq_value(ni5v20hz,length,samples_per_second)
    freq_ni5v50hz,value_ni5v50hz=frq_value(ni5v50hz,length,samples_per_second)
    freq_ni2v1khz,value_ni2v1khz=frq_value(ni2v1khz,length,samples_per_second)
    freq_ni2v8khz,value_ni2v8khz=frq_value(ni2v8khz,length,samples_per_second)

    freq_mcc5v10hz,value_mcc5v10hz=frq_value(mcc5v10hz,length,samples_per_second)
    freq_mcc5v20hz,value_mcc5v20hz=frq_value(mcc5v20hz,length,samples_per_second)
    freq_mcc5v50hz,value_mcc5v50hz=frq_value(mcc5v50hz,length,samples_per_second)
    freq_mcc2v1khz,value_mcc2v1khz=frq_value(mcc2v1khz,length,samples_per_second)
    freq_mcc2v8khz,value_mcc2v8khz=frq_value(mcc2v8khz,length,samples_per_second)
    # plt.plot(freq_mcc5v50hz,value_mcc5v50hz)
    # plt.scatter(freq_mcc5v50hz,value_mcc5v50hz,s=0.3)
    # plt.show()

    thd_ni5v10hz=THD_calculate(value_ni5v10hz,10/(samples_per_second/length)-1)
    thd_ni5v20hz=THD_calculate(value_ni5v20hz,20/(samples_per_second/length)-1)
    thd_ni5v50hz=THD_calculate(value_ni5v50hz,50/(samples_per_second/length)-1)
    thd_ni2v1khz=THD_calculate(value_ni2v1khz,1000/(samples_per_second/length)-1)
    thd_ni2v8khz=THD_calculate(value_ni2v8khz,8000/(samples_per_second/length)-1)
    thd_mcc5v10hz=THD_calculate(value_mcc5v10hz,10/(samples_per_second/length)-1)
    thd_mcc5v20hz=THD_calculate(value_mcc5v20hz,20/(samples_per_second/length)-1)
    thd_mcc5v50hz=THD_calculate(value_mcc5v50hz,50/(samples_per_second/length)-1)
    thd_mcc2v1khz=THD_calculate(value_mcc2v1khz,1000/(samples_per_second/length)-1)
    thd_mcc2v8khz=THD_calculate(value_mcc2v8khz,8000/(samples_per_second/length)-1)




    print("thd_ni5v10hz=",thd_ni5v10hz)
    print("thd_ni5v20hz=",thd_ni5v20hz)
    print("thd_ni5v50hz=",thd_ni5v50hz)
    print("thd_ni2v1khz=",thd_ni2v1khz)
    print("thd_ni2v8khz=",thd_ni2v8khz)
    print("thd_mcc5v10hz=",thd_mcc5v10hz)
    print("thd_mcc5v20hz=",thd_mcc5v20hz)
    print("thd_mcc5v50hz=",thd_mcc5v50hz)
    print("thd_mcc2v1khz=",thd_mcc2v1khz)
    print("thd_mcc2v8khz=",thd_mcc2v8khz)





    # plt.plot(freq,abs_fft_ni5v10hz)
    # plt.scatter(freq,abs_fft_ni5v10hz,s=0.3)
    # plt.show()

    # plt.plot(ni5v10hz[:102400],color='r',label="ni5v10hz")
    # phi=1701-210+1+1
    # plt.plot(mcc5v10hz[phi:102400+phi],color='b',label="mcc5v10hz")
    # plt.legend()
    # plt.show()