import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, fftshift
from nptdms import TdmsFile

def read_tdms(file_path):
    '''
    read TTL type .tdms file
    ********************************************************
    parameters
     file_path: file path of .tdms file

    return
           vib: vibration data
         speed: speed pulse data
    '''
    file = TdmsFile.read(file_path)
    vib = file['AIData']['Vibration'].data
    speed = file['AIData']['Speed'].data
    return vib, speed

def speed_cal2(speed, ppr, n_avg=256, fs=102400):
    '''
    calculate rpm from pulse signal
    ***************************************************
    parameters
         speed: pulse signal data
         n_avg: number of pulse in one calculation
           ppr: pulse per revolution
            fs: sampling frequency

    return
     time_list: list of time
      rpm_list: list of rpm corresponding to time
    '''
    time_list, rpm_list = [], []
    
    thr = (max(speed) - min(speed)) / 2
    nsp = np.where(speed < thr, 0, 1)       # set 0 for low voltage, 1 for high voltage
    pluse = np.diff(nsp)                    # get pluse location
    rising = np.where(pluse == 1)[0]        # start location of pluses
    middle = int((n_avg+1)/2)

    for i in range(middle, len(rising)):
        pluse_signal = rising[i-middle:i+middle+1]
        time = pluse_signal[middle] / fs
        dt = (pluse_signal[-1] - pluse_signal[0]) / fs
        rpm = (len(pluse_signal) - 1) / ppr / dt * 60
        time_list.append(time)
        rpm_list.append(rpm)
    return np.array(time_list), np.array(rpm_list)

def get_colormap(vib, speed_x, speed_y, fig_type='time-frequency', n_frame=8192, pad_width=0, 
                 resolution_level=1, window='hanning', fs=102400,
                 roi_time=None, roi_freq=None, title=''):
    '''
    doing stft and draw colormap
    ********************************************
    parameters
          vib:  vibration data
      speed_x:  time of revolution data
      speed_y:  rpm value of revolution data
     fig_type:  'time-frequency' or 'frequency-rpm'
      n_frame:  frame length for vibration data
    pad_width:  zero padded length of vibration data
      overlap:  overlap between two vibration data frame
       window:  window for fft
           fs:  sampling rate
     roi_time:  time segment for drawing
     roi_freq:  frequency segment for drawing
        title:  title of figure
    '''
    rpm_list = []
    fft_list = []
    time_list = []
    
    step = int(fs / 100 * resolution_level)
    total_length = n_frame + 2 * pad_width
    
    if window == 'hanning':
        win = np.hanning(total_length)
    elif window == 'kaiser':
        win = np.kaiser(total_length, 5)
    elif window == None:
        win = np.ones(total_length)

    if roi_time:
        p_start = int(min(roi_time) * fs)
        p_end = int(max(roi_time) * fs)
    else:
        p_start = 0
        p_end = len(vib)

    fft_count = (p_end - p_start) // step

    for i in range(fft_count):      
        time = (p_start + i * step) / 102400
        rpm = speed_y[np.argmin(abs(np.array(speed_x) - time))]

        vib_seg = vib[p_start+i*step:p_start+i*step+total_length]
        n_add = np.zeros(total_length - len(vib_seg))
        frame = np.concatenate((vib_seg, n_add))
        frame = np.pad(frame, pad_width=pad_width)
        
        vib_fft = 20 * np.log(abs(fft(frame * win))[1:len(frame) // 2] / np.sqrt(len(frame)))
        freq = fftfreq(len(frame), d=1/fs)[1:len(frame) // 2]
        if roi_freq == None:
            selected_freq = freq
            selected_vib = vib_fft
        else:
            selected_freq = freq[(freq > min(roi_freq)) & (freq < max(roi_freq))]
            selected_vib = vib_fft[(freq > min(roi_freq)) & (freq < max(roi_freq))]

        time_list.append(time)
        rpm_list.append(rpm)
        fft_list.append(selected_vib)
    
    '''
    if roi_time == None:
        selected_time = time_list
        selected_fft = fft_list
        selected_rpm = rpm_list
    else:
        time_list = np.array(time_list)
        fft_list = np.array(fft_list)
        rpm_list = np.array(rpm_list)
        selected_time = time_list[(time_list > min(roi_time)) & (time_list < max(roi_time))]
        selected_fft = fft_list[(time_list > min(roi_time)) & (time_list < max(roi_time)),:]
        selected_rpm = rpm_list[(time_list > min(roi_time)) & (time_list < max(roi_time))]
    '''
    if fig_type == 'time-frequency':
        X = [time_list, selected_freq]
    elif fig_type == 'frequency-rpm':
        X = [selected_freq, rpm_list]
    else:
        X = [0, 0]
    return colormap_plot(fig_type, X, fft_list, title)

def colormap_plot(fig_type, X, fft_value, title):
    '''
    colormap plot
    ******************************************
    parameters
    fig_type:  'time-frequency' or 'frequency-rpm', draw corresponding figure
           X:  data for plot, [time, frequency] if fig_type = 'time-frequency'
                              [frequency, rpm]  if fig_type = 'frequency-rpm'
    fft_value:  stft value for plot
    '''
    if fig_type == 'time-frequency':
        plt.pcolormesh(X[0], X[1], np.array(fft_value).T, cmap='jet')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar()
        plt.title(title)
        plt.show()
    elif fig_type == 'frequency-rpm':
        plt.pcolormesh(X[0], X[1], fft_value, cmap='jet')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Speed (rpm)')
        plt.title(title)
        plt.colorbar()
        plt.show()
    else:
        print("Unsupported colormap! Supported types are 'time-frequency' and 'frequency-rpm'!")


def main():
    print("Please input file path:")
    file_path = input()
    print("Please input ppr number:")
    ppr = int(input())
    print("Please input sampling rate:")
    fs = int(input())
    print("Please input resolution level, from 1 to 10:")
    rl = int(input())

    while 1:
        print("Please choose the desired colormap type:\n1. time-frequency    2.frequency-rpm")
        type_plot = input()
        if type_plot == '1':
            fig_type = 'time-frequency'
            break
        elif type_plot == '2':
            fig_type = 'frequency-rpm'
            break
        else:
            print("Invalid input!")

    print("Please input interested rpm region, seperated by ',':")
    roi_rpm = input().replace(' ', '').replace('，', ',').split(',')
    print("Processing.....")
    roi_min, roi_max = float(roi_rpm[0]), float(roi_rpm[1])
    vib, speed = read_tdms(file_path)
    speedx, speedy = speed_cal2(speed, ppr, fs=fs)
    
    idx_for_rpm = []
    for rr in roi_rpm:
        rp = float(rr)
        point = np.where(np.array(speedy) < rp, 0, 1)
        idx_rpm = np.diff(point) != 0
        idx_for_rpm.append(idx_rpm)

    plt.plot(speedx, speedy)
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (rpm)")
    plt.scatter(speedx[1:][idx_for_rpm[0]], speedy[1:][idx_for_rpm[0]], marker='o', color='green')
    plt.scatter(speedx[1:][idx_for_rpm[1]], speedy[1:][idx_for_rpm[1]], marker='o', color='blue')
    plt.title("Please left click to choose desired rpm region, right click to cancel.")
    pos = plt.ginput(n=2)
    plt.close()
    t = speedx[1:][idx_for_rpm[0] + idx_for_rpm[1]]
    t_selected = []
    for p in pos:
        x = p[0]
        t_select = t[np.argmin(abs(t-x))]
        t_selected.append(t_select)
    print("Ploting...Please wait a moment...")
    get_colormap(vib=vib, speed_x=speedx, speed_y=speedy, fig_type=fig_type, n_frame=8192*2, pad_width=8192*0, 
             resolution_level=rl, window='hanning', fs=fs, roi_time=[min(t_selected), max(t_selected)], roi_freq=[0, 20000], title='')


