import time
import h5py
from nptdms import TdmsFile
import numpy as np
from scipy import stats, fftpack
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import csv


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
        cutoff = wn[0]/nyq
        b, a = butter(order, cutoff, btype, analog=False)
    else:
        lowcut = wn[0]/nyq
        highcut = wn[1]/nyq
        b, a = butter(order, [lowcut, highcut], btype, analog=False)
    return filtfilt(b, a, signal)






startTime = time.time()

fs = 102400  #采样频率
Ts = 1/fs
# 下极值点在 0~lowLevel 度之间。
lowLevel = 0.3
PolePair =4 #4极对，8条鱼算一圈。
#对于10000RPM，每秒167转，4极对，每转8条鱼，共1333条鱼。对应采样率102400次，每条鱼77个点。所以包络线的任意两个最低点之间的距离不会超过77*0.5 = 38(取偶数）个点
maxDis = 38  #偶数。左右各19个点
lastFlag = 1 #最后一条鱼时的符号状态

allrawdata = read_raw_data("D:/qdaq/debug/210904-1/error_data/017700948N900018_210903191408.h5", ["Sin","Cos"], "hdf5")
# allrawdata = read_raw_data("D:/qdaq/debug/210904-1/error_data/ref_no_error/017700948N900072_210903212824.h5", ["Sin"], "hdf5")
speed_sin=np.array(allrawdata["Sin"][:])
# speed_sin=np.array(allrawdata["Sin"][196250:204800])
# speed_sin=np.array(allrawdata["Sin"][:16384])

read_file_time = time.time() 
#低通滤波 截止频率 = （10000+10000/60)*1.28
speed_sin = butter_filter(speed_sin, [int((10000+10000/60)*2.56)], fs)
 
hSin = fftpack.hilbert(speed_sin)

# 舍弃前10个点，后10个点
# speed_sin=speed_sin[10:len(speed_sin)-10]
# hSin=hSin[10:len(hSin)-10]
# left_index+=10
envSin = np.sqrt(speed_sin ** 2 + hSin ** 2)


#查找包络线的正负符号
#env    0.4  0.3  0.2  0.1  0.05  0.04 0.1  0.2 0.03  0.02  0.2  0.01
#minLoc           2    3    4     5    6     7   8     9     10   11
#minLoc2               0    1     2              5     6           8     

sign = np.ones(len(envSin))
minLoc = np.where(envSin < lowLevel)[0]
minLoc2 = np.where(np.diff(envSin[minLoc]) < 0)[0]  
zeroLoc = []
for i in range(0,len(minLoc2)):
    loc2 = minLoc2[i]
    if i < len(minLoc2)-1 and minLoc2[i+1] - loc2 == 1:
        continue
    loc = minLoc[loc2 + 1]
    env = envSin[loc]
    interrupted = 0
    #往右找maxDis/2 个点
    for i in range(1,min(len(envSin) - loc, int(maxDis/2))):
        if envSin[i+loc] < env:
            interrupted = 1
            break
    #往左找maxDis/2个点
    if interrupted == 0:
        for i in range(1,min(loc , int(maxDis/2))):
            if envSin[loc - i] < env:
                interrupted = 1
                break
    if interrupted == 0:
        zeroLoc.append(loc)


    
#print(zeroLoc)    
    


# plt.plot(zeroLoc)

#plt.figure("hSin")
#plt.plot(speed_sin)
#plt.plot(hSin)
#plt.figure("envSin")
#plt.plot(envSin)


#每个零点反转一次
t = np.arange(0,len(envSin)) *Ts
for loc in zeroLoc:
    envSin[loc:] *= -1
#plt.plot(t,envSin,color = 'brown')

# plt.figure("envSin")
# plt.plot(envSin)

# plt.figure("zero")
# plt.scatter(np.array(zeroLoc)*Ts,envSin[zeroLoc],color = 'black')

envSin = butter_filter(envSin, [int((10000/60)*2.56*PolePair)], fs)   

# plt.figure("envSin_afterenv")
# plt.plot(envSin)
#plt.plot(t,envSin)

zeroLoc = np.array(np.where(np.diff(1 * (envSin >= 0)) != 0 )[0])


# plt.scatter(zeroLoc*Ts,envSin[zeroLoc],color = 'r')


#plt.show()


 
#求转速。
average_Num = 8 # 在多少个0点内做平均。要求大于2.

speedList = []
for index in range(average_Num, np.size(zeroLoc)):
    #计算转速 
    timeElapsed = (zeroLoc[index] - zeroLoc[index - average_Num])*Ts
    
    speed = average_Num/PolePair/2/timeElapsed*60
    #speed = 10000
    speedList.append(speed)   

    #print("index=",index, "speed =",speed, "sizeof filter_x_data=", np.size(filter_x_data))



end_time = time.time()
print("find speed time = ",end_time - read_file_time)

plt.figure("speed")
plt.scatter(zeroLoc[average_Num:]*Ts,np.array(speedList), s = 5,color="r" )

#将转速记入本地csv文件
filename = '../speed.csv'
csvFile3 = open(filename,'w',newline='')
writer2 = csv.writer(csvFile3)
for i in range(len(speedList)):
      writer2.writerow([zeroLoc[i+average_Num]*Ts,speedList[i]])
csvFile3.close()

plt.show()
