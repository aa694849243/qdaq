import numpy as np
from scipy import signal
from nptdms import TdmsFile
from scipy.fft import fft, fftfreq
from tools import *
import timeit

def get_nlargest(array, n=3):
    '''
    get n largest number in array
    ********************************************
    parameters
         array: array to be processed
    return
           arg: index of n largest numbers
    '''
    arr = np.array(array)
    arg = arr.argsort()[-n:]
    return arg

def get_freq_from_order(rpm, order, resolution, peak_freq, peak_vib):
    '''
    get frequency for orders
    *******************************************
    parameters
           rpm: last rpm value
         order: targeted order
    resolution: order resolution
     peak_freq: frequency of peaks
      peak_vib: amplitude of peaks after fft
    
    return
       rpm_cal: calculated rpm value
    '''
    order_left, order_right = (order-resolution) * rpm / 60, (order+resolution) * rpm / 60
    twin_peak = np.array([freq for freq in peak_freq if order_left <= freq <= order_right])
    if len(twin_peak) > 0:
        twin_peak_id = [i for x in twin_peak for i, v in enumerate(peak_freq) if x == v]
        twin_vib = peak_vib[twin_peak_id]
        twin_freq = twin_peak[twin_vib > .5 * max(twin_vib)]
        if len(twin_freq) == 1:
            freq_list = twin_freq
        elif len(twin_freq) >= 2:
            freq_list = np.append(twin_freq, np.mean(twin_freq))
        rpm_list = np.concatenate((freq_list/3.14, freq_list/3.15)) * 60
    else:
        freq_list, rpm_list = [], []
    return freq_list, rpm_list

def get_speed_from_vib(vib, fs, machine_type):
    '''
    extract speed for 3 different types motors from vibration data
    *****************************************************************
    parameters
             vib: vibration data
    machine_type: type of motor

    return
            time: time for speed
           speed: extracted speed
    '''
    if machine_type == 'type1':
        time, speed = speed_from_vib_type1(vib, fs)
    elif machine_type == 'type2':
        time, speed = speed_from_vib_type2(vib, fs)
    elif machine_type == 'type3':
        time, speed = speed_from_vib_type3(vib, fs)
    
    return time, speed

def speed_from_vib_type1(vib, fs):
    '''
    extract speed from vibration data for type1 motor
    ****************************************************
    parameters
           vib: vibration data
            fs: sampling rate
    
    return
             t: time for speed
           rpm: speed including initial speed
    '''
    n_frame = 8192 * 6                                  # frame length
    overlap = 0.8                                       # overlap
    pad_length = 8192                                   # zero pad length
    step = int(n_frame * (1 - overlap) * 0.25)          # step length
    rpm = [150]                                         # find rpm list with initial rpm, determined by RMS value
    t = []

    for i in range(0, len(vib) // step):
        time = step * i / fs
        frame = vib[i*step:i*step+n_frame]
    
        # ----------check RMS value-------------
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 0.2 and rpm[-1] < 160:
            t.append(time)
            rpm.append(150)
            continue
    
        # ----------------start doing fft--------------------
        frame = np.pad(frame, pad_width=(n_frame-len(frame))//2, mode='constant', constant_values=0)
        frame = np.pad(frame, pad_width=pad_length, mode='constant', constant_values=0)
        window = np.hanning(len(frame))
        vib_fft = (abs(fft(frame * window))[1:len(frame)//2])
        vib_freq = fftfreq(len(frame), d=1/fs)[1:len(frame)//2]

        # --------------------get region of interest--------------------
        roi_vib = vib_fft[vib_freq < 250]
        roi_freq = vib_freq[vib_freq < 250]
        roi_peaks, _ = signal.find_peaks(roi_vib)
        large_peak_freq = roi_freq[roi_peaks]

        # ----------------------get max 10 peaks and corresponding freq, amplitude, remove noise frequency(57.8Hz)------------------------
        try:
            max10 = get_nlargest(roi_vib[roi_peaks], 10)
            max10_freq = large_peak_freq[max10]
            max10_vib = roi_vib[roi_peaks][max10]
            freq_denoise = max10_freq[np.where(abs(max10_freq - 57.8) > 0.5)]
            vib_denoise = max10_vib[np.where(abs(max10_freq - 57.8) > 0.5)]
        except:
            continue
        # ---------------------calculate rpm from max 10 peaks------------------------------   

        # --------------------- denoise ----------------------
        if rpm[-1] > 1100:
            freq_list, rpm_list = get_freq_from_order(rpm[-1], 3.15, resolution=0.15, peak_freq=freq_denoise, peak_vib=vib_denoise)
            if len(freq_list) > 0:
                if len(freq_list) == 1:
                    rpm_idx = np.nanargmin(abs(rpm_list - rpm[-1]))
                    freq_cal = np.tile(freq_list, 2)[rpm_idx]
                    rpm_cal = rpm_list[rpm_idx]
                    if abs(rpm_cal - rpm[-1]) > 120:
                        freq_list = freq_denoise
                        rpm_list = (freq_list/1) * 60
                
            else:
                freq_list = freq_denoise
                rpm_list = np.concatenate((freq_list/1, freq_list/2)) * 60

        else:
            freq_list = freq_denoise
            rpm_list = np.concatenate((freq_list/1, freq_list/1)) * 60
            rpm_idx = np.nanargmin(abs(rpm_list - rpm[-1]))
            freq_cal = np.tile(freq_list, 2)[rpm_idx]
            rpm_cal = rpm_list[rpm_idx]
            if abs(rpm_cal - rpm[-1]) > 120:
                freq_list = freq_denoise
                rpm_list = np.concatenate((freq_list/3.14, freq_list/3.15)) * 60

        # -------------------choose most close to last rpm---------------------
        rpm_idx = np.nanargmin(abs(rpm_list - rpm[-1]))
        freq_cal = np.tile(freq_list, 2)[rpm_idx]
        rpm_cal = rpm_list[rpm_idx]

        # -------------------pretend noise frequency overlap with true order frequency---------------------
        noise_rpm = max10_freq[np.where(abs(max10_freq - 57.8) < 0.5)] / 3.15 * 60
        if len(noise_rpm):
            if (abs(noise_rpm[0] - rpm[-1]) <= abs(rpm_cal - rpm[-1])) and (abs(noise_rpm[0] - rpm[-1]) < 50):
                rpm_cal = noise_rpm[0]

        # -------------------pretend order 1 and order 3 peaks are not in largest 10 peaks------------------
        if rpm[-1] < 200:
            if abs(rpm[-1] - rpm_cal) > 100:
                rpm_cal = rpm[-1]
    
        rpm_cal = max(rpm_cal, 150)
        rpm.append(rpm_cal)
        t.append(time)

    return t, rpm[1:]

def speed_from_vib_type2(vib, fs):
    '''
    extract speed from vibration data for type2 motor
    ****************************************************
    parameters
           vib: vibration data
            fs: sampling rate
    
    return
             t: time for speed
           rpm: speed including initial speed
    '''

    step = int(fs/100)                      # step length
    rpm = [2490]                            # find rpm list with initial rpm, determined by RMS value
    t = []
    order_list = []
    for i in range(0, len(vib) // step):
        time = step * i / fs
        if time < 25.51:
            n_frame = int(8192 * 2)
        else:
            n_frame = 8192
        frame = vib[i*step:i*step+n_frame]

        # ----------check RMS value------------- #
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 0.4:
            t.append(time)
            rpm.append(rpm[-1])
            order_list.append(0)
            continue
        
        # ----------------start doing fft-------------------- #
        frame = np.pad(frame, pad_width=(n_frame-len(frame))//2, mode='constant', constant_values=0)
        if time < 25.51:
            pad_length = int(8192 * 7)
        else:
            pad_length = int(8192*0.5)
        
        frame = np.pad(frame, pad_width=pad_length)
        window = np.hanning(len(frame))
        vib_fft = (abs(fft(frame * window))[1:len(frame)//2])
        vib_freq = fftfreq(len(frame), d=1/fs)[1:len(frame)//2]

        # --------------------get region of interest-------------------- #
        roi_vib = vib_fft[vib_freq < 2000]
        roi_freq = vib_freq[vib_freq < 2000]
        roi_peaks, _ = signal.find_peaks(roi_vib)
        roi_peak_freq = roi_freq[roi_peaks]

        # ----------------------get max 10 peaks and corresponding freq, amplitude, remove noise frequency(57.8Hz)-------------------------- #
        max10 = get_nlargest(roi_vib[roi_peaks], 50)
        max10_freq = roi_peak_freq[max10]
        max10_vib = roi_vib[roi_peaks][max10]

        # ---------------------calculate rpm from max 10 peaks------------------------------ #

        # use last order first
        try:
            rpm_for_last_order = get_rpm_candidate(vib_list=max10_vib, freq_list=max10_freq, order=choose_order, last_rpm=rpm[-1])
            if abs(rpm_for_last_order / rpm[-1] - 1) < 0.01:
                rpm_cal = rpm_for_last_order
                rpm_list = max10_freq / choose_order * 60
                rpm.append(rpm_cal)
                t.append(time)
                order_list.append(choose_order)
                continue
        except:
            pass

        # if last order not good, use default order
        if rpm[-1] < 6000:
            order_queue = [8.1, 2, 1]
        else:
            order_queue = [2, 1, 8.1]
        rpm_list = get_rpm_by_order_sequence(vib_list=max10_vib, order=order_queue, freq_list=max10_freq, last_rpm=rpm[-1])
        
        # ---------------------------- select rpm from rpm_list --------------------------- #
        # stop until first rpm meet requirement (less than 1% volatility)
        find = False                                                                  # tag for find or not
        for x in range(len(rpm_list)):
            if abs(rpm_list[x] / rpm[-1] - 1) < 0.01:                                 # 1% volatility 
                find = True    
                break
        if find:                                                                      # if find, use this order next frame
            rpm_cal = rpm_list[x]
            choose_order = order_queue[x]
        else:
            rpm_cal = rpm[-1]
            choose_order = None

        rpm.append(rpm_cal)
        t.append(time)
        order_list.append(choose_order)
    
    return t, rpm[1:]

def speed_from_vib_type3(vib, fs):
    '''
    extract speed from vibration data for type2 motor
    ****************************************************
    parameters
           vib: vibration data
            fs: sampling rate
    
    return
             t: time for speed
           rpm: speed including initial speed
    '''

    rpm = [2490]
    step = int(fs / 100)
    n_frame = 8192 * 2
    time_list = []

    for i in range(len(vib) // step):

        time = i * step / fs
        # ------------------------- preprocess for vibration data ---------------------- #
        frame = vib[i*step:i*step+n_frame]
        frame = np.pad(frame, pad_width=8192*1)
        frame = np.hanning(len(frame)) * frame
        
        # ------------------------- fft ----------------------------#
        fft_vib = abs(fft(frame))[1:len(frame)//2]
        fft_freq = fftfreq(len(frame), d=1/fs)[1:len(frame)//2]

        # ------------------------- get region of interest ----------------------- #
        roi = fft_freq < 2000
        roi_vib = fft_vib[roi]
        roi_freq = fft_freq[roi]

        # ------------------------- find peaks in region of interest ---------------------- #
        peaks, _ = signal.find_peaks(roi_vib)
        peak_vib = roi_vib[peaks]
        peak_freq = roi_freq[peaks]

        # ------------------------- get max 10 peaks -------------------------- #
        max10 = np.argsort(peak_vib)[-20:]
        max10_vib = peak_vib[max10]
        max10_freq = peak_freq[max10]

        if time > 18:
            order1 = 1
        else:
            order1 = 1.01

        # ------------------------- get speed candidate list ----------------------- #
        rpm_ls = max10_freq / order1 * 60
        rpm_cal = rpm_ls[np.argmin(abs(rpm_ls - rpm[-1]))]

        # ------------------------- select rpm ------------------------------- #        
        if (rpm_cal - rpm[-1]) > 1000:
            rpm_ls = max10_freq / 2 * 60
            rpm_cal = rpm_ls[np.argmin(abs(rpm_ls - rpm[-1]))]

        rpm.append(rpm_cal)
        time_list.append(time)
    
    return time_list, rpm[1:]

def get_speed_from_vib(vib, fs, machine_type):
    '''
    extract speed for 3 different types motors from vibration data
    *****************************************************************
    parameters
             vib: vibration data
    machine_type: type of motor

    return
            time: time for speed
           speed: extracted speed
    '''
    if machine_type == 'type1':
        time, speed = speed_from_vib_type1(vib, fs)
    elif machine_type == 'type2':
        time, speed = speed_from_vib_type2(vib, fs)
    elif machine_type == 'type3':
        time, speed = speed_from_vib_type3(vib, fs)
    
    return time, speed

def speed_from_vib_type1_stream(vib, fs, rpm0, count):
    '''
    extract speed from vibration data for type1 motor
    ****************************************************
    parameters
           vib: vibration data
            fs: sampling rate
          rpm0: initial speed value
         count: counter for stream data
    
    return
             t: time for speed
           rpm: speed including initial speed
         count: counter for stream data
    '''
    n_frame = 8192 * 6                                  # frame length
    pad_length = 8192                                   # zero pad length
    step = int(fs * 0.02)                               # step length
    t0 = count * step / fs
    rpm = [rpm0]
    t = [t0]
    vib = vib[int(count*step):]
    if len(vib) < n_frame:
        return [], [], count

    for i in range(len(vib) // step):
        time = step * i / fs + count * step / fs
        frame = vib[i*step:i*step+n_frame]
        if len(frame) < n_frame:
            continue

        # ----------check RMS value-------------
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 0.2 and rpm[-1] < 160:
            t.append(time)
            rpm.append(rpm0)
            continue
    
        # ----------------start doing fft--------------------
        frame = np.pad(frame, pad_width=(n_frame-len(frame))//2, mode='constant', constant_values=0)
        frame = np.pad(frame, pad_width=pad_length, mode='constant', constant_values=0)
        window = np.hanning(len(frame))
        vib_fft = (abs(fft(frame * window))[1:len(frame)//2])
        vib_freq = fftfreq(len(frame), d=1/fs)[1:len(frame)//2]

        # --------------------get region of interest--------------------
        roi_vib = vib_fft[vib_freq < 250]
        roi_freq = vib_freq[vib_freq < 250]
        roi_peaks, _ = signal.find_peaks(roi_vib)
        large_peak_freq = roi_freq[roi_peaks]

        # ----------------------get max 10 peaks and corresponding freq, amplitude, remove noise frequency(57.8Hz)------------------------
        try:
            max10 = get_nlargest(roi_vib[roi_peaks], 10)
            max10_freq = large_peak_freq[max10]
            max10_vib = roi_vib[roi_peaks][max10]
            freq_denoise = max10_freq[np.where(abs(max10_freq - 57.8) > 0.5)]
            vib_denoise = max10_vib[np.where(abs(max10_freq - 57.8) > 0.5)]
        except:
            continue
        # ---------------------calculate rpm from max 10 peaks------------------------------   

        # --------------------- denoise ----------------------
        if rpm[-1] > 1100:
            freq_list, rpm_list = get_freq_from_order(rpm[-1], 3.15, resolution=0.15, peak_freq=freq_denoise, peak_vib=vib_denoise)
            if len(freq_list) > 0:
                if len(freq_list) == 1:
                    rpm_idx = np.nanargmin(abs(rpm_list - rpm[-1]))
                    freq_cal = np.tile(freq_list, 2)[rpm_idx]
                    rpm_cal = rpm_list[rpm_idx]
                    if abs(rpm_cal - rpm[-1]) > 120:
                        freq_list = freq_denoise
                        rpm_list = (freq_list/1) * 60
                
            else:
                freq_list = freq_denoise
                rpm_list = np.concatenate((freq_list/1, freq_list/2)) * 60

        else:
            freq_list = freq_denoise
            rpm_list = np.concatenate((freq_list/1, freq_list/1)) * 60
            rpm_idx = np.nanargmin(abs(rpm_list - rpm[-1]))
            rpm_cal = rpm_list[rpm_idx]
            if abs(rpm_cal - rpm[-1]) > 120:
                freq_list = freq_denoise
                rpm_list = np.concatenate((freq_list/3.14, freq_list/3.15)) * 60

        # -------------------choose most close to last rpm---------------------
        rpm_idx = np.nanargmin(abs(rpm_list - rpm[-1]))
        rpm_cal = rpm_list[rpm_idx]

        # -------------------pretend noise frequency overlap with true order frequency---------------------
        noise_rpm = max10_freq[np.where(abs(max10_freq - 57.8) < 0.5)] / 3.15 * 60
        if len(noise_rpm):
            if (abs(noise_rpm[0] - rpm[-1]) <= abs(rpm_cal - rpm[-1])) and (abs(noise_rpm[0] - rpm[-1]) < 50):
                rpm_cal = noise_rpm[0]

        # -------------------pretend order 1 and order 3 peaks are not in largest 10 peaks------------------
        if rpm[-1] < 200:
            if abs(rpm[-1] - rpm_cal) > 100:
                rpm_cal = rpm[-1]
    
        rpm_cal = max(rpm_cal, 150)
        rpm.append(rpm_cal)
        t.append(time)
    count += (len(rpm) - 1)

    return t[1:], rpm[1:], count

def speed_from_vib_type2_stream(vib, fs, rpm0, count):
    '''
    extract speed from vibration data for type2 motor
    ****************************************************
    parameters
           vib: vibration data
            fs: sampling rate
    
    return
             t: time for speed
           rpm: speed including initial speed
    '''

    step = int(fs/100)                      # step length
    t0 = count * step / fs
    rpm = [rpm0]                            # find rpm list with initial rpm, determined by RMS value
    t = [t0]
    vib = vib[count*step:]
    if len(vib) < 8192 * 2:
        return [], [], count

    for i in range(len(vib) // step):
        time = step * i / fs + count * step / fs
        if time < 25.51:
            n_frame = int(8192 * 2)
        else:
            n_frame = 8192
        frame = vib[i*step:i*step+n_frame]
        if len(frame) < n_frame:
            continue

        # ----------check RMS value------------- #
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 0.4:
            t.append(time)
            rpm.append(rpm[-1])
            continue
        
        # ----------------start doing fft-------------------- #
        frame = np.pad(frame, pad_width=(n_frame-len(frame))//2, mode='constant', constant_values=0)
        if time < 25.51:
            pad_length = int(8192 * 7)
        else:
            pad_length = int(8192*0.5)
        
        frame = np.pad(frame, pad_width=pad_length)
        window = np.hanning(len(frame))
        vib_fft = (abs(fft(frame * window))[1:len(frame)//2])
        vib_freq = fftfreq(len(frame), d=1/fs)[1:len(frame)//2]

        # --------------------get region of interest-------------------- #
        roi_vib = vib_fft[vib_freq < 2000]
        roi_freq = vib_freq[vib_freq < 2000]
        roi_peaks, _ = signal.find_peaks(roi_vib)
        roi_peak_freq = roi_freq[roi_peaks]

        # ----------------------get max 10 peaks and corresponding freq, amplitude, remove noise frequency(57.8Hz)-------------------------- #
        max10 = get_nlargest(roi_vib[roi_peaks], 50)
        max10_freq = roi_peak_freq[max10]
        max10_vib = roi_vib[roi_peaks][max10]

        # ---------------------calculate rpm from max 10 peaks------------------------------ #

        # use last order first
        try:
            rpm_for_last_order = get_rpm_candidate(vib_list=max10_vib, freq_list=max10_freq, order=choose_order, last_rpm=rpm[-1])
            if abs(rpm_for_last_order / rpm[-1] - 1) < 0.01:
                rpm_cal = rpm_for_last_order
                rpm_list = max10_freq / choose_order * 60
                rpm.append(rpm_cal)
                t.append(time)
                continue
        except:
            pass

        # if last order not good, use default order
        if rpm[-1] < 6000:
            order_queue = [8.1, 2, 1]
        else:
            order_queue = [2, 1, 8.1]
        rpm_list = get_rpm_by_order_sequence(vib_list=max10_vib, order=order_queue, freq_list=max10_freq, last_rpm=rpm[-1])
        
        # ---------------------------- select rpm from rpm_list --------------------------- #
        # stop until first rpm meet requirement (less than 1% volatility)
        find = False                                                                  # tag for find or not
        for x in range(len(rpm_list)):
            if abs(rpm_list[x] / rpm[-1] - 1) < 0.01:                                 # 1% volatility 
                find = True    
                break
        if find:                                                                      # if find, use this order next frame
            rpm_cal = rpm_list[x]
            choose_order = order_queue[x]
        else:
            rpm_cal = rpm[-1]
            choose_order = None

        rpm.append(rpm_cal)
        t.append(time)
    
    count += (len(rpm) - 1)
    return t[1:], rpm[1:], count

def speed_from_vib_type3_stream(vib, fs, rpm0, count):
    '''
    extract speed from vibration data for type2 motor
    ****************************************************
    parameters
           vib: vibration data
            fs: sampling rate
          rpm0: initial speed value
         count: counter for stream data
    
    return
             t: time for speed
           rpm: speed including initial speed
         count: counter for stream data
    '''

    step = int(fs / 100)
    t0 = count * step / fs
    t = [t0]
    rpm = [rpm0]
    n_frame = 8192 * 2
    vib = vib[count*step:]
    if len(vib) < n_frame:
        return [], [], count

    for i in range(len(vib) // step):
        time = i * step / fs + count * step / fs
        # ------------------------- preprocess for vibration data ---------------------- #
        frame = vib[i*step:i*step+n_frame]
        if len(frame) < n_frame:
            continue
        frame = np.pad(frame, pad_width=8192*1)
        frame = np.hanning(len(frame)) * frame
        
        # ------------------------- fft ----------------------------#
        fft_vib = abs(fft(frame))[1:len(frame)//2]
        fft_freq = fftfreq(len(frame), d=1/fs)[1:len(frame)//2]

        # ------------------------- get region of interest ----------------------- #
        roi = fft_freq < 2000
        roi_vib = fft_vib[roi]
        roi_freq = fft_freq[roi]

        # ------------------------- find peaks in region of interest ---------------------- #
        peaks, _ = signal.find_peaks(roi_vib)
        peak_vib = roi_vib[peaks]
        peak_freq = roi_freq[peaks]

        # ------------------------- get max 10 peaks -------------------------- #
        max10 = np.argsort(peak_vib)[-20:]
        max10_vib = peak_vib[max10]
        max10_freq = peak_freq[max10]

        if time > 18:
            order1 = 1
        else:
            order1 = 1.01

        # ------------------------- get speed candidate list ----------------------- #
        rpm_ls = max10_freq / order1 * 60
        rpm_cal = rpm_ls[np.argmin(abs(rpm_ls - rpm[-1]))]

        # ------------------------- select rpm ------------------------------- #        
        if (rpm_cal - rpm[-1]) > 1000:
            rpm_ls = max10_freq / 2 * 60
            rpm_cal = rpm_ls[np.argmin(abs(rpm_ls - rpm[-1]))]

        rpm.append(rpm_cal)
        t.append(time)

    count += (len(rpm) - 1)
    return t[1:], rpm[1:], count



#---------------------------- function test ---------------------------#
fs = 102400

######################## type1 machine test #########################

input_file = 'D:/SpeedFromAmplitudeV2.0/data/Inovance/018200314KC00128_191212083225.tdms'
vib, speed = read_tdms(input_file)
t, s = get_speed_from_vib(vib, fs, 'type1')
#plt.plot(t, s)
#plt.show()
#print('type1 time:')
#print(timeit.timeit("get_speed_from_vib(vib, fs, 'type1')", globals=globals(), number=100))


######################## type2 machine test #########################
'''
input_file = 'D:/SpeedFromAmplitudeV2.0/data/rawdata/20070614/3.5Y_200706142057.tdms'
vib, speed = read_tdms(input_file)
t, s = get_speed_from_vib(vib, fs, 'type2')
#plt.plot(t, s)
#plt.show()
print('type2 time:')
#print(timeit.timeit("get_speed_from_vib(vib, fs, 'type2')", globals=globals(), number=100))

######################## type3 machine test ##########################

input_file = 'D:/SpeedFromAmplitudeV2.0/data/tdms/017700944MB00236.tdms'
vib1, vib2, sin, cos = read_new_tdms(input_file)
t, s = get_speed_from_vib(vib1, fs, 'type3')
#plt.plot(t, s)
#plt.show()
print('type3 time:')
#print(timeit.timeit("get_speed_from_vib(vib, fs, 'type3')", globals=globals(), number=100))
'''

####################### type1 machine stream test #######################

# read file
input_file = 'D:/SpeedFromAmplitudeV2.0/data/Inovance/018200314KC00043_191214053650.tdms'
vib, speed = read_tdms(input_file)

# parameter set
fs = 102400
count = 0
rpm0 = 150

# generate simu data
simu_data = []
for i in range(len(vib) // 10000):
    simu_data.append(vib[:(i+1)*10000])

# result variable
rest, resp = [], []

# simu
for data in simu_data:
    time, speed, count = speed_from_vib_type1_stream(data, fs, rpm0, count)
    rest.extend(time)
    resp.extend(speed)
    # avoid no speed data
    if len(speed):
        rpm0 = speed[-1]
plt.plot(rest, resp, label='stream data')
plt.plot(t, s, label='read all file')
plt.legend()
plt.show()
'''
######################## type2 machine stream test ########################

# read file
input_file = 'D:/SpeedFromAmplitudeV2.0/data/rawdata/20070614/3.5Y_200706142057.tdms'
vib, speed = read_tdms(input_file)

# parameter set
count = 0
rpm0 = 2490

# generate simu data
simu_data = []
for i in range(len(vib) // 10000):
    simu_data.append(vib[:(i+1)*10000])

# result data
rest, resp = [], []

# simu
for data in simu_data:
    time, speed, count = speed_from_vib_type2_stream(data, fs, rpm0, count)
    rest.extend(time)
    resp.extend(speed)
    # avoid no speed data
    if len(speed):
        rpm0 = speed[-1]
plt.plot(rest, resp)
plt.show()

####################### type3 machine stream test ########################

# read file
input_file = 'D:/SpeedFromAmplitudeV2.0/data/tdms/017700944MB00236.tdms'
vib1, vib2, sin, cos = read_new_tdms(input_file)

# parameter set
count = 0
rpm0 = 2490

# generate simu data
simu_data = []
for i in range(len(vib) // 10000):
    simu_data.append(vib[:(i+1)*10000])

# result data
rest, resp = [], []

for data in simu_data:
    time, speed, count = speed_from_vib_type3_stream(data, fs, rpm0, count)
    rest.extend(time)
    resp.extend(speed)
    # avoid no speed data
    if len(speed):
        rpm0 = speed[-1]
plt.plot(rest, resp)
plt.show()
'''