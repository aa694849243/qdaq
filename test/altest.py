import matplotlib.pyplot as plt
import numpy as np

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

def rms(data):
    return np.sqrt(np.sum(np.power(data,2))/len(data))
if __name__=="__main__":

    t=1
    sin_1000=sin_wave(1, 1000, 102400, 0, t)+1
    sin_900=sin_wave(1, 900, 102400, 0, t)+1
    sin_1100=sin_wave(1, 1100, 102400, 0, t)+1
    rawdata=sin_900+sin_1000+sin_1100
    rawdata=sin_900+1

    rms_rawdata=rms(rawdata)

    com_fft= np.fft.rfft(rawdata,102400) / len(rawdata) * 2
    # com_fft=np.fft.fft(a_102400)
    # com_fft[0]/=np.sqrt(2)
    com_fft[0]/=2
    # com_fft*=np.sqrt(2)/2
    abs_fft=np.abs(com_fft)
    com_freq_2=np.fft.rfftfreq(len(rawdata))



    com_fft_1= np.fft.fft(rawdata) / len(rawdata) * 2
    # com_fft_1[0]/=2
    com_freq_1=np.fft.fftfreq(len(rawdata))

    rms_a = np.sqrt(np.sum(np.power(abs_fft, 2)))
    rms_b= np.sqrt(np.sum(np.power(np.abs(com_fft_1), 2)))/2
    plt.figure(1)
    plt.plot(com_freq_1,np.abs(com_fft_1),c='r')
    plt.plot(com_freq_2,np.abs(com_fft),c='b')
    plt.show()



    print(1)


    count=10
    sampleChan=8192
    os_list=list()
    for i in range(10):
        frame= rawdata[i * sampleChan:(i + 1) * sampleChan]
        os_list_temp=np.fft.rfft(frame)/len(frame)*2
        # os_list_temp[0]/=2
        os_list.append(os_list_temp)

    os=np.sqrt(np.sum(np.power(np.abs(os_list),2),axis=0)/len(os_list))
    plt.figure(2)
    plt.plot(os)
    plt.show()

    print(1)