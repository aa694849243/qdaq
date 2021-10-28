import matplotlib.pyplot as plt
from tools import get_colormap, read_tdms, speed_cal2

if __name__ == '__main__':
    # file_path='D:/qdaq/Simu/mcc_ni_aqsrt_0.tdms'
    file_path='D:/qdaq/Simu/ni_mcc_aqsrt_0.tdms'
    file_path='D:/qdaq/Simu/jinkang-1.tdms'

    vib, speed=read_tdms(file_path)
    # vib=vib[360592+45800:360592+45800+1045000]
    # speed=speed[360592+45800:360592+45800+1045000]
    # plt.plot(speed)
    # plt.show()
    time_list, rpm_list = speed_cal2(speed, 1024,n_avg=256,fs=102400)
    plt.plot(time_list,rpm_list)
    plt.show()
    get_colormap(vib, time_list, rpm_list, fig_type='time-frequency',fs=51200,title=file_path)