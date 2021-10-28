import numpy as np
import matplotlib.pyplot as plt

vib_ifft=list()

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
    n=65536
    # n=51200
    vib_fft = np.fft.rfft(vib,n=n)/len(vib)*2
    freq = np.fft.rfftfreq(n, d=1 / fs)

    # frequency range for speed
    left = min_speed / 60 * (order - 0.5)
    right = max_speed / 60 * (order + 0.5)
    left=649.5
    right=650.5
    idx = (freq >= left) & (freq <= right)

    # find target frequency
    target = np.argmax(np.abs(vib_fft[idx]))
    # target_min = np.argmin(vib_fft[idx])
    speed_cali = freq[idx][target] / order * 60
    array_for_ifft=np.zeros(len(vib_fft),dtype="complex")
    index=np.argmax(idx)+target
    print(freq[index])
    array_for_ifft[index-points//2:index+points//2+1]=vib_fft[index-points//2:index+points//2+1]
    print(freq[index-points//2])
    print(freq[index+points//2])
    # array_for_ifft[3000]=vib_fft[index-points//2:index+points//2+1]
    # array_for_ifft[index-1:index+1+1]=vib_fft[index-1:index+1+1]

    # vib_ifft_temp=np.fft.irfft(vib_fft)*len(array_for_ifft)/2

    vib_ifft_temp=np.fft.irfft(array_for_ifft)*len(vib)/2
    print(len(vib_ifft_temp))
    vib_ifft.extend(vib_ifft_temp[:len(vib)])
    # print(len(vib_ifft_temp))
    # print(len(vib_ifft))
    # angle.append(np.angle(vib_fft[idx][target]))
    # speed_cali = (freq[idx][target] / order * 60+freq[idx][target_min] / order * 60)/2
    return speed_cali




if __name__ == '__main__':
    fs=102400
    t=2
    n=int(fs*t)
    # n=102400
    # sin_999hz=sin_wave(1,640,102400,0,t)
    # sin_1000hz=sin_wave(1,650,102400,0,t)
    # sin_1001hz=sin_wave(1,660,102400,0,t)
    # rawdata=sin_999hz+sin_1000hz+sin_1001hz
    # sin_999hz=sin_wave(1,640,102400,0,t)
    sin_650hz=sin_wave(1,650,102400,0,t)
    # sin_1001hz=sin_wave(1,660,102400,0,t)
    rawdata=sin_650hz

    # sampleLength=int(1*65536)
    sampleLength=int(1*8192)
    step=sampleLength//4

    count=len(rawdata)//sampleLength
    # count=1
    for i in range(count):
        frame=rawdata[i*sampleLength:(i+1)*sampleLength]
        plt.figure("rawdata")
        plt.plot(frame)
        vib_ifft_frame=resample1(frame,100,100,100,fs,33)


    plt.figure("vib_ifft")
    plt.plot(vib_ifft)


    plt.show()
    print(1)

    # plt.figure("rawdata")
    # plt.plot(rawdata)

    # freq_rawdata=np.fft.rfftfreq(n,d=1/fs)
    # complex_fft_rawdata=np.fft.rfft(rawdata,n=n)/len(rawdata)*2
    # abs_fft_rawdata=abs(complex_fft_rawdata)

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
