import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, fftshift
from nptdms import TdmsFile
import sys

def get_colormap(vib, speed_x, speed_y, fig_type='time-frequency', n_frame=8192, pad_width=4096, 
                 overlap=0.8, window='hanning', fs=102400,
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
        title: title of plot
    '''
    rpm_list = []
    fft_list = []
    time_list = []
    
    step = 1024 * 1#int(n_frame * (1 - overlap))
    total_length = n_frame + 2 * pad_width
    
    if window == 'hanning':
        win = np.hanning(total_length)
    elif window == 'kaiser':
        win = np.kaiser(total_length, 5)
    elif window == None:
        win = np.ones(total_length)

    for i in range(len(vib) // step):      
        time = i * step / 102400
        rpm = speed_y[np.argmin(abs(np.array(speed_x) - time))]

        vib_seg = vib[i*step:i*step+n_frame]
        n_add = np.zeros(n_frame - len(vib_seg))
        frame = np.concatenate((vib_seg, n_add))
        frame = np.pad(frame, pad_width=pad_width)
        
        vib_fft = np.log(abs(fft(frame * win))[1:len(frame) // 2])
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


    if fig_type == 'time-frequency':
        X = [selected_time, selected_freq]
    elif fig_type == 'frequency-rpm':
        X = [selected_freq, selected_rpm]
    else:
        X = [0, 0]
    return colormap_plot(fig_type, X, selected_fft, title)


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
        sc = plt.pcolormesh(X[0], X[1], np.array(fft_value).T, cmap='jet')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        #pos = plt.ginput(n=-1)
        #time, rpm = rpm_cal(pos, 1)
        #plt.show()
        #plt.plot(time, rpm)
        plt.title(title)
        plt.colorbar()
        plt.show()
    elif fig_type == 'frequency-rpm':
        plt.pcolormesh(X[0], X[1], fft_value, cmap='jet', vmin=-10, vmax=10)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Speed (rpm)')
        plt.title(title)
        plt.colorbar()
        plt.show()
    else:
        print("Unsupported colormap! Supported types are 'time-frequency' and 'frequency-rpm'!")

def rpm_cal(pos, order):
    if pos[0][0] > pos[1][0]:
        start_point = pos[1]
        end_point = pos[0]
    else:
        start_point = pos[0]
        end_point = pos[1]

    rpm_s = start_point[1] / order * 60
    rpm_e = end_point[1] / order * 60

    num = int(end_point[0] - start_point[0]) * 100
    time = np.linspace(start_point[0], end_point[0], num=num)
    rpm = np.linspace(rpm_s, rpm_e, num=num)
    return time, rpm


def speed_cal2(speed, n_avg, ppr, fs=102400):
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
        rpm = 2 * middle / ppr / dt * 60
        time_list.append(time)
        rpm_list.append(rpm)
    return time_list, rpm_list

def revolution_count(speed, thr):
    '''
    count revolution by pulse
    *******************************************************
    parameters
                 speed: pulse signal data
                   thr: threshold value for high or low voltage

    return
        cycle_location: revolution location in time
    '''
    nsp = np.where(speed < thr, 0, 1)
    pluse = np.diff(nsp)
    rising = np.where(pluse == 1)[0]
    cycle_location = rising[::65] / 102400
    return cycle_location


input_path = 'D:/SpeedFromAmplitudeV2.0/data/Inovance/018200314KC00043_191214053650.tdms'
file = TdmsFile.read(input_path)
vib = file['AIData']['Vibration'].data
speed = file['AIData']['Speed'].data
speed_x, speed_y = speed_cal2(speed, 256, ppr=64)


get_colormap(vib=vib, speed_x=speed_x, speed_y=speed_y, fig_type='frequency-rpm', n_frame=8192*2, pad_width=8192*0, 
             overlap=0.8, window='hanning', fs=102400, roi_time=None, roi_freq=[0, 2000], title='')       



