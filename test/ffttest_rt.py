import os
import time

import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile

vib_ifft=np.array([])

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
    Ts = 1/fs
    n = t / Ts
    n = np.arange(n)
    y = A*np.sin(2*np.pi*f*n*Ts + phi*(np.pi/180))
    return y

def resample1(vib, min_speed,max_speed, order, fs,points,slice_points):
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

    len_vib = len(vib)
    n=65536
    # n=102400
    # 65536/102400=0.64
    # n=51200
    vib_fft = np.fft.rfft(vib,n=n)/len(vib)*2
    freq = np.fft.rfftfreq(n, d=1 / fs)
    plt.figure(1)
    plt.plot(freq,np.abs(vib_fft))

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
    print(freq[index])
    array_for_ifft[index-points//2:index+points//2+1]=vib_fft[index-points//2:index+points//2+1]
    order_slice=np.sqrt(np.sum(np.power(np.abs(vib_fft[index-slice_points//2:index+slice_points//2+1]), 2)))
    print(freq[index-points//2])
    print(freq[index+points//2])
    # array_for_ifft[3000]=vib_fft[index-points//2:index+points//2+1]
    # array_for_ifft[index-1:index+1+1]=vib_fft[index-1:index+1+1]

    # vib_ifft_temp=np.fft.irfft(vib_fft)*len(array_for_ifft)/2

    vib_ifft_temp=np.fft.irfft(array_for_ifft)*len(vib)/2

    print(len(vib_ifft_temp))


    # print(len(vib_ifft_temp))
    # print(len(vib_ifft))
    # angle.append(np.angle(vib_fft[idx][target]))
    # speed_cali = (freq[idx][target] / order * 60+freq[idx][target_min] / order * 60)/2
    return speed_cali,order_slice,vib_ifft_temp[:len(vib)]


def rms(data):
    return np.sqrt(np.sum(np.power(data,2))/len(data))



if __name__ == '__main__':
    fs=102400
    t=2
    n=int(fs*t)
    # n=102400
    sin_999hz=sin_wave(1,620,102400,0,t)
    sin_1000hz=sin_wave(1,650,102400,0,t)
    sin_1001hz=sin_wave(1,680,102400,0,t)
    rawdata=sin_999hz+sin_1000hz+sin_1001hz
    # # sin_999hz=sin_wave(1,640,102400,0,t)
    # sin_650hz=sin_wave(1,650,102400,0,t)
    # # sin_1001hz=sin_wave(1,660,102400,0,t)
    # rawdata=sin_650hz

    # sampleLength=int(1*65536)

    path = 'D:/qdaq/rawdata/test01'
    file = "test01_3mm_8k_210802064140.tdms"
    startSpeed=7500
    endSpeed=9500
    order=13

    # file = "test01_3mm_110k_210802070234.tdms"
    # startSpeed=100000
    # endSpeed=110000
    # order=13

    data = TdmsFile.read(os.path.join(path, file))
    rawdata = data['AIData']['Mic'].data
    # rawdata=rawdata[:int(1*102400)]
    # sampleLength=int(1*8192)
    sampleLength=int(1*2048)
    step=sampleLength//8*5

    count=len(rawdata)//step-1
    speed_list=list()
    slice_list=list()
    # count=1

    t1=time.time()
    for i in range(count):
        frame=rawdata[i*step:i*step+sampleLength]
        # plt.figure("rawdata")
        # plt.plot(frame)
        speedtemp,order_slice,vib_ifft_temp_frame=resample1(frame,startSpeed,endSpeed,order,fs,33,7)
        speed_list.append(speedtemp)
        slice_list.append(order_slice)
        # 舍弃第一帧前8192//4个点
        # plt.figure(i)
        # plt.plot(np.arange(len(vib_ifft_temp_frame)),vib_ifft_temp_frame)
        if i==0:
            vib_ifft=np.append(vib_ifft,vib_ifft_temp_frame[sampleLength//4:sampleLength//8*7])
        else:
            vib_ifft[-sampleLength//8:]=\
            (vib_ifft[-sampleLength//8:]+vib_ifft_temp_frame[sampleLength//8:sampleLength//4])/2
            vib_ifft=np.append(vib_ifft,vib_ifft_temp_frame[sampleLength//4:sampleLength//8*7])

    t2=time.time()

    plt.figure("order slice")
    index_sort_speed=np.argsort(speed_list)
    plt.plot(np.array(speed_list)[index_sort_speed],np.array(slice_list)[index_sort_speed])


    plt.figure("vib_ifft")
    plt.plot(vib_ifft)

    # index_1=np.array([4096*i for i in range(10)])
    index_2=np.array([sampleLength//8*5*i for i in range(1,count+1)])
    index_1=index_2-sampleLength//8
    index=np.append(index_1,index_2[:len(index_2)-1])

    plt.scatter(index,vib_ifft[index],c='r',marker="*")
    plt.show()


    t3=time.time()
    ttl=np.where(vib_ifft>0,1,-1)
    diff=np.diff(ttl)
    trigger=np.where(diff!=0)[0]
    t4=time.time()
    # 最间隔小点数 / 采样率 * 2 * 关注阶次 > 1 / 转频 * 0.7
    # 最小间隔点数>1/转频*采样率/关注阶次/2 *0.7
    diff_min=60/startSpeed*fs/order/2*0.7
    diff_trigger=np.diff(trigger)
    triggererror=np.where(diff_trigger<diff_min)[0]
    trigger_error_vib_index=trigger[triggererror+1]
    plt.ion()
    plt.show()

    os_list=list()
    os_count=(len(trigger)-2*order*32)//(2*order*8)
    os_count=(len(trigger))//(2*order*32)

    step_num=8
    for i in range(os_count):
        # frame=rawdata[trigger[i*2*order*8]:trigger[i*2*order*8+2*order*32]]
        frame=rawdata[trigger[i*2*order*32]:trigger[(i+1)*2*order*32]]
        print("len(frame)1:{}".format(len(frame)))
        # os_temp=np.abs(np.fft.rfft(frame*np.kaiser(len(frame),9),))/len(frame)*2
        os_temp=np.abs(np.fft.rfft(frame))/len(frame)*2
        # print(np.fft.rfftfreq(n=len(frame),d=32/len(frame)))
        os_temp[0]/=2
        os_list.append(os_temp[:30*32])

    os=np.sqrt(np.sum(np.power(os_list,2),axis=0)/len(os_list))
    plt.figure("rawdata_frame")
    plt.plot(np.arange(30*32)*0.03125,os)
    plt.show()

    ppr=26
    rpm_num=52
    rpm_step=26
    rpm=list()
    rpml=list()
    i=0
    while i*rpm_step+rpm_num<len(trigger):
        rpm.append(2/((trigger[i*rpm_step+rpm_num]-trigger[i*rpm_step])/fs)*60)
        rpml.append(trigger[i*rpm_step]/fs)
        i+=1

    t5 = time.time()
    rawdata = rawdata[trigger[0]:trigger[-1]]
    # plt.figure("rpm")
    # plt.plot(rpml,rpm)
    angle=list()
    angle_list=[list(np.linspace(360/2/order*i,360/2/order*(i+1),num=trigger[i+1]-trigger[i],endpoint=False)) for i in range(len(trigger)-1)]
    for a_l in angle_list:
        angle.extend(a_l)



    # rawdata与angle的长度相同

    dr=1/30/2/1.28/1.28
    angle_insert=np.arange(0,angle[-1],dr*360)
    vib_rsp=np.interp(angle_insert,angle,rawdata)
    t6=time.time()
    rms_rawdata=np.sqrt(np.sum(np.power(rawdata,2),axis=0)/len(rawdata))
    rms_insert=np.sqrt(np.sum(np.power(vib_rsp,2),axis=0)/len(vib_rsp))

    plt.figure("vib_rsp_whole")
    os1_freq=np.fft.rfftfreq(n=len(vib_rsp),d=dr)
    # os1_abs=np.abs(np.fft.rfft(vib_rsp*np.kaiser(len(vib_rsp),9)))/len(vib_rsp)*2
    os1_abs=np.abs(np.fft.rfft(vib_rsp))/len(vib_rsp)*2
    os1_abs[0]/=np.sqrt(2)
    rms_os1=np.sqrt(np.sum(np.power(os1_abs,2))/2)
    oc_1 = np.sqrt(np.sum(np.power(os1_abs[23755:24497], 2)))
    os1_abs[0] /= np.sqrt(2)
    plt.plot(os1_freq,os1_abs)


    t7=time.time()
    num=int(32//dr)
    step=num//4
    # os_count=(len(vib_rsp)-num)//step
    os_count=len(vib_rsp)//num
    os_list=list()
    for i in range(os_count):
        if i == os_count-1:
            time.sleep(0.1)
        # frame=vib_rsp[i*step:i*step+num]
        frame=vib_rsp[i*num:(i+1)*num]
        print("len(frame)2:{}".format(len(frame)))
        # os_temp=np.abs(np.fft.rfft(np.array(frame)*np.kaiser(num,9)))/num*2
        os_temp=np.abs(np.fft.rfft(np.array(frame)))/num*2
        os_temp[0]/=2
        os_temp*=np.sqrt(2)/2
        os_list.append(os_temp)


    os=np.sqrt(np.sum(np.power(os_list,2),axis=0)/len(os_list))
    # os=np.mean(os_list,axis=0)

    rms_os_1=np.sqrt(np.sum(np.power(os,2)))
    plt.figure("vib_rsp_frame")
    t8=time.time()
    os_rsp_freq=np.fft.rfftfreq(num,d=dr)
    oc_2=np.sqrt(np.sum(np.power(os[410:422],2)))
    plt.plot(os_rsp_freq,os)

    t=t2-t1+t4-t3+t6-t5+t8-t7
    plt.show()






    # 向trigger之间插入振动点




    print(1)

    #

    # plt.figure("rawdata")
    # plt.plot(rawdata)

    # freq_rawdata=np.fft.rfftfreq(n,d=1/fs)
    # complex_fft_rawdata=np.fft.rfft(rawdata,n=n)/len(rawdata)*2
    # abs_fft_rawdata=abs(complex_fft_rawdata)
    #
    # complex_filter=np.zeros(len(abs_fft_rawdata),dtype="complex")
    #
    # complex_filter[3000]=complex_fft_rawdata[3000]
    # complex_filter[799:802]=complex_fft_rawdata[799:802]
    # print(complex_filter[1000])

    # ifft_filter=np.fft.irfft(complex_filter)*n
    #
    #
    # fft_ifft=np.fft.rfft(ifft_filter)
    # plt.figure("fft")
    # plt.plot(freq_rawdata, abs_fft_rawdata)
    # plt.figure("ifft")
    # plt.plot(ifft_filter)
    # print("len(sig):{}".format(len(ifft_filter)))
    # plt.figure("fft_ifft")
    # plt.plot(freq_rawdata,fft_ifft)
    # plt.legend()
    # plt.show()
