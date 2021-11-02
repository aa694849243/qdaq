import os
import time

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from nptdms import TdmsFile
from scipy.fft import fft, fftfreq
from qdaq.qyj.tools import *
import timeit

from qdaq.auxiliary_tools import read_tdms
from qdaq.tools.tools import get_rpm_by_order_sequence, get_rpm_candidate, get_timefrequency_colormap


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
    order_left, order_right = (order - resolution) * rpm / 60, (order + resolution) * rpm / 60
    twin_peak = np.array([freq for freq in peak_freq if order_left <= freq <= order_right])
    if len(twin_peak) > 0:
        twin_peak_id = [i for x in twin_peak for i, v in enumerate(peak_freq) if x == v]
        twin_vib = peak_vib[twin_peak_id]
        twin_freq = twin_peak[twin_vib > .5 * max(twin_vib)]
        if len(twin_freq) == 1:
            freq_list = twin_freq
        elif len(twin_freq) >= 2:
            freq_list = np.append(twin_freq, np.mean(twin_freq))
        rpm_list = np.concatenate((freq_list / 3.14, freq_list / 3.15)) * 60
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
    n_frame = 8192 * 6  # frame length
    overlap = 0.8  # overlap
    pad_length = 8192  # zero pad length
    step = int(n_frame * (1 - overlap) * 0.25)  # step length
    rpm = [150]  # find rpm list with initial rpm, determined by RMS value
    t = []

    for i in range(0, len(vib) // step):
        time = step * i / fs
        frame = vib[i * step:i * step + n_frame]

        # ----------check RMS value-------------
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 0.2 and rpm[-1] < 160:
            t.append(time)
            rpm.append(150)
            continue

        # ----------------start doing fft--------------------
        frame = np.pad(frame, pad_width=(n_frame - len(frame)) // 2, mode='constant', constant_values=0)
        frame = np.pad(frame, pad_width=pad_length, mode='constant', constant_values=0)
        window = np.hanning(len(frame))
        vib_fft = (abs(fft(frame * window))[1:len(frame) // 2])
        vib_freq = fftfreq(len(frame), d=1 / fs)[1:len(frame) // 2]

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
            freq_list, rpm_list = get_freq_from_order(rpm[-1], 3.15, resolution=0.15,
                                                      peak_freq=freq_denoise, peak_vib=vib_denoise)
            if len(freq_list) > 0:
                if len(freq_list) == 1:
                    rpm_idx = np.nanargmin(abs(rpm_list - rpm[-1]))
                    freq_cal = np.tile(freq_list, 2)[rpm_idx]
                    rpm_cal = rpm_list[rpm_idx]
                    if abs(rpm_cal - rpm[-1]) > 120:
                        freq_list = freq_denoise
                        rpm_list = (freq_list / 1) * 60

            else:
                freq_list = freq_denoise
                rpm_list = np.concatenate((freq_list / 1, freq_list / 2)) * 60

        else:
            freq_list = freq_denoise
            rpm_list = np.concatenate((freq_list / 1, freq_list / 1)) * 60
            rpm_idx = np.nanargmin(abs(rpm_list - rpm[-1]))
            freq_cal = np.tile(freq_list, 2)[rpm_idx]
            rpm_cal = rpm_list[rpm_idx]
            if abs(rpm_cal - rpm[-1]) > 120:
                freq_list = freq_denoise
                rpm_list = np.concatenate((freq_list / 3.14, freq_list / 3.15)) * 60

        # -------------------choose most close to last rpm---------------------
        rpm_idx = np.nanargmin(abs(rpm_list - rpm[-1]))
        freq_cal = np.tile(freq_list, 2)[rpm_idx]
        rpm_cal = rpm_list[rpm_idx]

        # -------------------pretend noise frequency overlap with true order frequency---------------------
        noise_rpm = max10_freq[np.where(abs(max10_freq - 57.8) < 0.5)] / 3.15 * 60
        if len(noise_rpm):
            if (abs(noise_rpm[0] - rpm[-1]) <= abs(rpm_cal - rpm[-1])) and (
                    abs(noise_rpm[0] - rpm[-1]) < 50):
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

    step = int(fs / 100)  # step length
    rpm = [2490]  # find rpm list with initial rpm, determined by RMS value
    t = []
    order_list = []
    for i in range(0, len(vib) // step):
        time = step * i / fs
        if time < 25.51:
            n_frame = int(8192 * 2)
        else:
            n_frame = 8192
        frame = vib[i * step:i * step + n_frame]

        # ----------check RMS value------------- #
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 0.4:
            t.append(time)
            rpm.append(rpm[-1])
            order_list.append(0)
            continue

        # ----------------start doing fft-------------------- #
        frame = np.pad(frame, pad_width=(n_frame - len(frame)) // 2, mode='constant', constant_values=0)
        if time < 25.51:
            pad_length = int(8192 * 7)
        else:
            pad_length = int(8192 * 0.5)

        frame = np.pad(frame, pad_width=pad_length)
        window = np.hanning(len(frame))
        vib_fft = (abs(fft(frame * window))[1:len(frame) // 2])
        vib_freq = fftfreq(len(frame), d=1 / fs)[1:len(frame) // 2]

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
            rpm_for_last_order = get_rpm_candidate(vib_list=max10_vib, freq_list=max10_freq,
                                                   order=choose_order, last_rpm=rpm[-1])
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
        rpm_list = get_rpm_by_order_sequence(vib_list=max10_vib, order=order_queue, freq_list=max10_freq,
                                             last_rpm=rpm[-1])

        # ---------------------------- select rpm from rpm_list --------------------------- #
        # stop until first rpm meet requirement (less than 1% volatility)
        find = False  # tag for find or not
        for x in range(len(rpm_list)):
            if abs(rpm_list[x] / rpm[-1] - 1) < 0.01:  # 1% volatility
                find = True
                break
        if find:  # if find, use this order next frame
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
        frame = vib[i * step:i * step + n_frame]
        frame = np.pad(frame, pad_width=8192 * 1)
        frame = np.hanning(len(frame)) * frame

        # ------------------------- fft ----------------------------#
        fft_vib = abs(fft(frame))[1:len(frame) // 2]
        fft_freq = fftfreq(len(frame), d=1 / fs)[1:len(frame) // 2]

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
    n_frame = 8192 * 6  # frame length
    pad_length = 8192  # zero pad length
    step = int(fs * 0.02)  # step length
    t0 = count * step / fs
    rpm = [rpm0]
    t = [t0]
    vib = vib[int(count * step):]
    if len(vib) < n_frame:
        return [], [], count

    for i in range(len(vib) // step):
        time = step * i / fs + count * step / fs
        frame = vib[i * step:i * step + n_frame]
        if len(frame) < n_frame:
            continue

        # ----------check RMS value-------------
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 0.2 and rpm[-1] < 160:
            t.append(time)
            rpm.append(rpm0)
            continue

        # ----------------start doing fft--------------------
        frame = np.pad(frame, pad_width=(n_frame - len(frame)) // 2, mode='constant', constant_values=0)
        frame = np.pad(frame, pad_width=pad_length, mode='constant', constant_values=0)
        window = np.hanning(len(frame))
        vib_fft = (abs(fft(frame * window))[1:len(frame) // 2])
        vib_freq = fftfreq(len(frame), d=1 / fs)[1:len(frame) // 2]

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
            freq_list, rpm_list = get_freq_from_order(rpm[-1], 3.15, resolution=0.15,
                                                      peak_freq=freq_denoise, peak_vib=vib_denoise)
            if len(freq_list) > 0:
                if len(freq_list) == 1:
                    rpm_idx = np.nanargmin(abs(rpm_list - rpm[-1]))
                    freq_cal = np.tile(freq_list, 2)[rpm_idx]
                    rpm_cal = rpm_list[rpm_idx]
                    if abs(rpm_cal - rpm[-1]) > 120:
                        freq_list = freq_denoise
                        rpm_list = (freq_list / 1) * 60

            else:
                freq_list = freq_denoise
                rpm_list = np.concatenate((freq_list / 1, freq_list / 2)) * 60

        else:
            freq_list = freq_denoise
            rpm_list = np.concatenate((freq_list / 1, freq_list / 1)) * 60
            rpm_idx = np.nanargmin(abs(rpm_list - rpm[-1]))
            rpm_cal = rpm_list[rpm_idx]
            if abs(rpm_cal - rpm[-1]) > 120:
                freq_list = freq_denoise
                rpm_list = np.concatenate((freq_list / 3.14, freq_list / 3.15)) * 60

        # -------------------choose most close to last rpm---------------------
        rpm_idx = np.nanargmin(abs(rpm_list - rpm[-1]))
        rpm_cal = rpm_list[rpm_idx]

        # -------------------pretend noise frequency overlap with true order frequency---------------------
        noise_rpm = max10_freq[np.where(abs(max10_freq - 57.8) < 0.5)] / 3.15 * 60
        if len(noise_rpm):
            if (abs(noise_rpm[0] - rpm[-1]) <= abs(rpm_cal - rpm[-1])) and (
                    abs(noise_rpm[0] - rpm[-1]) < 50):
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

    step = int(fs / 100)  # step length
    t0 = count * step / fs
    rpm = [rpm0]  # find rpm list with initial rpm, determined by RMS value
    t = [t0]
    vib = vib[count * step:]
    if len(vib) < 8192 * 2:
        return [], [], count

    for i in range(len(vib) // step):
        time = step * i / fs + count * step / fs
        if time < 25.51:
            n_frame = int(8192 * 2)
        else:
            n_frame = 8192
        frame = vib[i * step:i * step + n_frame]
        if len(frame) < n_frame:
            continue

        # ----------check RMS value------------- #
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 0.4:
            t.append(time)
            rpm.append(rpm[-1])
            continue

        # ----------------start doing fft-------------------- #
        frame = np.pad(frame, pad_width=(n_frame - len(frame)) // 2, mode='constant', constant_values=0)
        if time < 25.51:
            pad_length = int(8192 * 7)
        else:
            pad_length = int(8192 * 0.5)

        frame = np.pad(frame, pad_width=pad_length)
        window = np.hanning(len(frame))
        vib_fft = (abs(fft(frame * window))[1:len(frame) // 2])
        vib_freq = fftfreq(len(frame), d=1 / fs)[1:len(frame) // 2]

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
            rpm_for_last_order = get_rpm_candidate(vib_list=max10_vib, freq_list=max10_freq,
                                                   order=choose_order, last_rpm=rpm[-1])
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
        rpm_list = get_rpm_by_order_sequence(vib_list=max10_vib, order=order_queue, freq_list=max10_freq,
                                             last_rpm=rpm[-1])

        # ---------------------------- select rpm from rpm_list --------------------------- #
        # stop until first rpm meet requirement (less than 1% volatility)
        find = False  # tag for find or not
        for x in range(len(rpm_list)):
            if abs(rpm_list[x] / rpm[-1] - 1) < 0.01:  # 1% volatility
                find = True
                break
        if find:  # if find, use this order next frame
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
    vib = vib[count * step:]
    if len(vib) < n_frame:
        return [], [], count

    for i in range(len(vib) // step):
        time = i * step / fs + count * step / fs
        # ------------------------- preprocess for vibration data ---------------------- #
        frame = vib[i * step:i * step + n_frame]
        if len(frame) < n_frame:
            continue
        frame = np.pad(frame, pad_width=8192 * 1)
        frame = np.hanning(len(frame)) * frame

        # ------------------------- fft ----------------------------#
        fft_vib = abs(fft(frame))[1:len(frame) // 2]
        fft_freq = fftfreq(len(frame), d=1 / fs)[1:len(frame) // 2]

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


def find_argmin_in(a):
    def find_argmin(y):
        return np.argmin(np.abs(a-y))
    return find_argmin

if __name__ =="__main__":

    path = r"D:\qdaq\rawdata\xiaodianji\21082609"
    # filename = "65NG1-1_210826093439.tdms"
    filenames=os.listdir(path)
    # filenames = ["105OK1-1_210826091618.tdms"]
    # samplePerChan=8192*3 counter=4, score[0,1]
    # filenames=["105NG1-2_210826093823.tdms"]
    # samplePerChan=8192*3 counter=6, score[0,1]  [1,16]
    # filenames=["85NG1-1_210826093625.tdms"]
    # filenames=["105OK3-1_210826093151.tdms"]
    # filenames=["105OK2-3_210826092524.tdms"]
    # filenames=["65NG2-1_210826094103.tdms"]
    # filenames=["85OK2-1_210826092316.tdms"]
    # filenames=["85OK3-3_210826093035.tdms"]
    # filenames=["85OK3-2_210826093011.tdms"]
    # filenames=["105NG2-2_210826094453.tdms"]

    for filename in filenames:
        speed=int(filename[:-23])
        speed_min= speed * 1000 *0.5
        speed_max=speed*1000*2
        if speed==105:
            ref_order_list = [1, 13]
            # ref_order_list = [13]
        elif speed==65:
            ref_order_list=[1,13]
            # ref_order_list = [13]
        elif speed==85:
            ref_order_list=[1,13]
            # ref_order_list = [ 13]
        else:
            ref_order_list=list()
        allrawdata=TdmsFile.read(os.path.join(path,filename))

        fs=102400
        samplePerChan=8192




        ref_order_array=np.sort(ref_order_list)



        mic_rawdata=allrawdata["AIData"]["Mic1"]
        counter=0

        # 多个阶次的话为最大阶次的频率寻找范围，单个阶次时为该阶次的寻找范围
        max_freq=speed_max/60*ref_order_array[-1]
        min_freq=speed_min/60*ref_order_array[-1]
        # get_timefrequency_colormap(mic_rawdata, n_frame=8192, fs=fs, roi_freq=[0, 20000])

        rpm_list=list()
        rpml_list=list()

        while (counter+1)*samplePerChan<len(mic_rawdata):


            frame=mic_rawdata[counter*samplePerChan:(counter+1)*samplePerChan]
            frame_fft_abs=np.abs(np.fft.rfft(frame))/len(frame)*2
            frame_freq=np.fft.rfftfreq(len(frame),d=1/fs)

            if len(ref_order_array) == 1:
                idx = (frame_freq >= min_freq) & (frame_freq <= max_freq)
                # find target frequency
                target = np.argmax(np.abs(frame_fft_abs[idx]))
                # target_min = np.argmin(vib_fft[idx])
                speed_cali = frame_freq[idx][target] / ref_order_array[-1] * 60
                rpm_list.append(speed_cali)
                rpml_list.append((counter + 0.5) * samplePerChan / fs)
                counter+=1
                continue

            fft_abs_cut=frame_fft_abs[frame_freq<=max_freq]
            freq_cut=frame_freq[frame_freq<=max_freq]
            average=np.average(fft_abs_cut)
            n=100
            peaks_index,_=signal.find_peaks(fft_abs_cut)

            max_n = get_nlargest(fft_abs_cut[peaks_index], 100)
            max_n=np.sort(max_n)
            max_n_freq = freq_cut[peaks_index][max_n]
            max_n_vib = fft_abs_cut[peaks_index][max_n]


            peak_to_compare_list=list()
            last_peak_to_compare_index=0
            min_diff_frequency= 0.5 * speed_min / 60



            for i in range(0,len(max_n_freq)):
                if max_n_freq[i] < min_diff_frequency:
                    continue
                if  max_n_freq[i]-max_n_freq[last_peak_to_compare_index]<min_diff_frequency:
                    if max_n_vib[i]>max_n_vib[last_peak_to_compare_index]:
                        last_peak_to_compare_index=i
                else:
                    if max_n_freq[last_peak_to_compare_index]>min_diff_frequency:
                        peak_to_compare_list.append(last_peak_to_compare_index)

                    last_peak_to_compare_index=i



            peak_to_compare_list.append(last_peak_to_compare_index)
            peak_to_compare_array=np.array(peak_to_compare_list)
            freq_to_compare=max_n_freq[peak_to_compare_array]
            al_to_compare=max_n_vib[peak_to_compare_array]

            # plt.figure(counter)
            # plt.plot(freq_cut,fft_abs_cut)
            # plt.scatter(max_n_freq,max_n_vib,c="b",marker="*")
            # plt.scatter(max_n_freq[np.array(peak_to_compare_list)],max_n_vib[np.array(peak_to_compare_list)],c="r",marker="+")
            # # # plt.show()
            #
            #
            # #
            # if counter ==11:
            #     time.sleep(0.1)

            # score 0 9

            freq_revolution=fs/len(frame)
            # 每一个点作为第一个参考阶次的评分
            score=list()
            vib_to_compare=max_n_vib[peak_to_compare_array]
            arg_vib_max=np.argsort(vib_to_compare)
            rank=np.zeros(len(vib_to_compare))
            rank[arg_vib_max]=range(1,len(vib_to_compare)+1)
            score_without_al=list()
            score_al=list()
            for i in range(len(freq_to_compare)-1,-1,-1):
                if freq_to_compare[i] < min_freq:
                    break
                speed_temp=freq_to_compare[i]/ref_order_array[-1]
                freq_to_find= speed_temp * ref_order_array
                argmin_index=list(map(find_argmin_in(freq_to_compare), freq_to_find))
                # if i==7:
                #     time.sleep(0.1)

                # if len(np.where(np.abs(freq_to_find - freq_to_compare[argmin_index])>3*freq_revolution)[0])!=0:
                #     score.append(0)
                #     score_without_al.append(0)
                #     score_al.append(0)
                #     continue
                # score.append(np.sum(1 - np.abs(freq_to_find - freq_to_compare[argmin_index]) / freq_to_find) + np.sum(freq_revolution  * rank[argmin_index]))
                # score_without_al.append(np.sum(1 - np.abs(freq_to_find - freq_to_compare[argmin_index]) / freq_to_find))
                # score_al.append(np.sum(freq_revolution  * rank[argmin_index]))

                freq_right_index=np.where(np.abs(freq_to_find - freq_to_compare[argmin_index])<=3*freq_revolution)[0]
                score.append(np.sum(al_to_compare[argmin_index][freq_right_index]))


                # score.append(np.sum(1-np.abs(speed_temp_array-freq_to_compare[l])/speed_temp_array))
                # a=freq_to_compare-speed_temp_array.reshape((len(ref_order_array)-1,1))
                # min_index=np.argmin(np.abs(a), axis=1)
                # if np.max(a[min])>min_diff_frequency:
                #     continue
                # else:
                #     rpm_list.append(speed_temp*60)
                #     break
            index_first_order=np.argmax(score)
            rpm_list.append(freq_to_compare[len(freq_to_compare)-1-index_first_order]/ref_order_array[-1]*60)
            rpml_list.append((counter+0.5)*samplePerChan/fs)
            print("{}:{}".format(counter,rpm_list[-1]))
            counter+=1

        plt.figure(filename)
        plt.plot(rpml_list,rpm_list)
        plt.xlabel("t/s")
        plt.ylabel("speed/rpm")
    plt.legend()
    plt.show()
    print("over")