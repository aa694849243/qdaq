import numpy as np
from nptdms import TdmsFile
from scipy.fft import fft, fftfreq, fftshift, ifft, ifftshift
import matplotlib.pyplot as plt

def read_tdms(file_path):
    '''
    read TTL signal .tdms file
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

def read_new_tdms(file_path):
    '''
    read resolver signal .tdms file
    *********************************************************
    parameters
     file_path: file path of .tdms file

    return
          vib1: vibration data of channel 1
          vib2: vibration data of channel 2
           sin: sin voltage data
           cos: cos voltage data
    '''
    file = TdmsFile.read(file_path)
    vib1 = file['AIData']['Vib1'].data
    vib2 = file['AIData']['Vib2'].data
    sin = file['AIData']['Sin'].data
    cos = file['AIData']['Cos'].data
    return vib1, vib2, sin, cos

def speed_cal2(speed, ppr, n_avg=256, fs=102400):
    '''
    calculate speed from TTL signal by full-time based pulse
    **********************************************************
    parameters
         speed: pulse signal
         n_avg: number of pulse to do average
           thr: threshold to identify high or low voltage
            fs: sampling rate

    return
     time_list: list of time
      rpm_list: list of rpm corresponding to time 
    '''
    time_list, rpm_list = [], []
    
    thr = (max(speed) + min(speed)) / 2
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
    return time_list, rpm_list

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
        time = (p_start + i * step) / fs
        rpm = speed_y[np.argmin(abs(np.array(speed_x) - time))]

        vib_seg = vib[p_start+i*step:p_start+i*step+n_frame]
        n_add = np.zeros(n_frame - len(vib_seg))
        frame = np.concatenate((vib_seg, n_add))
        frame = np.pad(frame, pad_width=pad_width)
        
        vib_fft = 20 * np.log10(abs(fft(frame * win))[1:len(frame) // 2] / len(frame) * 2 / (2 * 10**-5) * 0.707)
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
        plt.title(title)
        plt.colorbar()
        #plt.show()
    elif fig_type == 'frequency-rpm':
        plt.pcolormesh(X[0], X[1], fft_value, cmap='jet')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Speed (rpm)')
        plt.title(title)
        plt.colorbar()
        plt.show()
    else:
        print("Unsupported colormap! Supported types are 'time-frequency' and 'frequency-rpm'!")

def revolution_count(speed, ppr):
    '''
    count revolution by pulse
    *********************************************************
    parameters
                 speed: pulse signal data
                   thr: threshold value for high or low voltage

    return
        cycle_location: revolution location in time
    '''
    thr = (max(speed) - min(speed)) / 2
    nsp = np.where(speed < thr, 0, 1)
    pluse = np.diff(nsp)
    rising = np.where(pluse == 1)[0]
    cycle_location = rising[::ppr+1] / 102400
    return cycle_location

def get_rpm_candidate(vib_list, freq_list, last_rpm, order):
    '''
    built-in function in rpm track
    '''
    if order:
        freq_list = np.array(freq_list)
        freq_left = (order - 0.3) * last_rpm / 60
        freq_right = (order + 0.3) * last_rpm / 60
        freq_target = freq_list[(freq_list >= freq_left) & (freq_list <= freq_right)]
        if len(freq_target) > 0:
            vib_target = vib_list[(freq_list >= freq_left) & (freq_list <= freq_right)]
            freq_obj = freq_target[vib_target > max(vib_target) * 0.5]
            rpm_list = freq_obj / order * 60
            return np.mean(rpm_list)
        else:
            return last_rpm

def get_rpm_by_order_sequence(vib_list, freq_list, last_rpm, order):
    '''
    built-in function in rpm track
    '''
    ls = []
    for o in order:
        rpm_o = get_rpm_candidate(vib_list, freq_list, last_rpm, o)
        ls.append(rpm_o)
    return ls

def get_revolution_location_by_speed(tn, tn1, vn, vn1, N):
    '''
    angular resampling (deprecated)
    *****************************************************
    parameters
           tn: time instance 1
          tn1: time instance 2
           vn: speed at tn
          vn1: speed at tn1
            N: angle in revolution
    return
          cnt: time at every N revolution
    '''
    cnt = []
    
    a = 0.5 * (vn1 - vn) / (tn1 - tn)
    b = (vn*tn1 - vn1*tn) / (tn1 - tn)
    c = -a*tn**2 - b*tn
    while True:
        c -= N

        x = np.roots([a, b, c])
        r = x[(x >= tn) & (x <= tn1)]
        if len(r) == 0:
            break
        else:
            cnt.append(r[0])
    return cnt

def get_revolution_by_speed(tn, tn1, vn, vn1, N):
    '''
    angular resampling
    *****************************************************
    parameters
           tn: time instance 1
          tn1: time instance 2
           vn: speed at tn
          vn1: speed at tn1
            N: angle in revolution
    return
          cnt: time at every N revolution
    '''
    cnt = []

    k = (vn1 - vn) / (tn1 - tn)
    if k == 0:
        T = vn * (tn1 - tn)
        cnt = tn + np.arange(1, T//N + 1) * (N / vn)
    else:
        a1 = tn - vn / k
        revolution = 0
        while True:
            revolution += N
            loc = a1 + np.sqrt(vn**2 + 2*k*revolution) / k
            if loc > tn1:
                break
            cnt.append(loc)
    return cnt

def get_order_spectrum(vib, d_revolution, n_revolution, max_order=200, normalize=False, title=''):
    '''
    draw order spectrum from vibration data
    ***************************************************
    parameters
             vib: vibration data
    d_revolution: revolutions between two points
    n_revolution: revolutions to do fft
       max_order: required max order
           title: title of charts
    
    returns

    '''
    #d_revolution = 1 / max_order
    n = int(n_revolution/d_revolution)

    o = fftfreq(n, d=d_revolution)[1:n//2]
    roi = o < max_order
    step = int(n/30)

    yy = []

    for i in range(len(vib)//step):
        frame = vib[i*step:i*step+n]
        if len(frame) < n:
            continue
        #frame = np.pad(frame)
        fy = abs(fft(frame * np.hanning(len(frame))))[1:len(frame)//2] / np.sqrt(len(frame))
        yy.append(fy[roi])

    if normalize == True:
        specy = np.mean(yy, axis=0) / max(np.mean(yy, axis=0))
    else:
        specy = np.mean(yy, axis=0)    
    plt.plot(o[roi], specy, label='order spectrum')
    plt.legend()
    plt.title(title)
    plt.show()
    return o[roi], specy

def get_order_cut(vib, time_list, speed_list, target_order, fs=102400, title=''):
    '''
    draw order cut figure
    *********************************************
    parameters
             vib: vibration data
       time_list: time
      speed_list: speed
    target_order: order for order cut
              fs: sampling rate
           title: title of plot
    return

    '''

    start, end = int(time_list[0] * fs), int(time_list[-1] * fs)
    vib_for_cut = vib[start:end]
    target_order = list([target_order])
    rms_list = []
    rpm_list = []

    step = 1024
    n = 8192 * 2
    for i in range(len(vib_for_cut) // step):
        frame = vib_for_cut[i*step:i*step+n]
        if len(frame) < n:
            continue
        vib_fft = abs(fft(frame))[1:len(frame)//2] / (len(frame)/2) / 1.414
        freq = fftfreq(len(frame), d=1/fs)[1:len(frame)//2]
        t = start / fs + i * step / fs
        idx = np.argmin(abs(np.array(time_list) - t))
        rpm = speed_list[idx]
        for o in target_order:
            freq_left = (o - 0.25) * rpm / 60
            freq_right = (o + 0.25) * rpm / 60
            vib_obj = vib_fft[(freq <= freq_right) & (freq >= freq_left)]
            rms = np.sqrt(np.sum(vib_obj ** 2))
            rms_list.append(rms)
        rpm_list.append(rpm)

    rms_list = np.array(rms_list).reshape(-1, len(target_order))

    for i in range(len(target_order)):
        plt.plot(rpm_list, rms_list[:,i], label='order{}'.format(target_order[i]))
    
    plt.legend()
    plt.title(title)
    plt.show()