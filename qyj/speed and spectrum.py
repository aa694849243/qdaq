import numpy as np
from nptdms import TdmsFile
from scipy.fft import fft, fftfreq
from nptdms import TdmsFile
import matplotlib.pyplot as plt


def speed_calibration(vib, speed_ref, order, fs):
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
    vib_fft = fft(vib)[1:n//2]
    freq = fftfreq(n, d=1/fs)[1:n//2]

    # frequency range for speed
    left = speed_ref / 60 * (order - 0.5)
    right = speed_ref / 60 * (order + 0.5)
    idx = (freq >= left) & (freq <= right)

    # find target frequency
    target = np.argmax(vib_fft[idx])
    speed_cali = freq[idx][target] / order * 60

    return speed_cali

def get_order_spectrum(vib, d_revolution, n_revolution, max_order=200, normalize=False):
    '''
    draw order spectrum from vibration data for constant speed
    ***************************************************
    parameters
             vib: vibration data
    d_revolution: revolutions between two points
    n_revolution: revolutions to do fft
       max_order: required max order
       normalize: normalize rms
    
    returns
          o[roi]: x-axis for order spectrum
           specy: y-axis for order spectrum
    '''
    
    # nfft
    n = int(n_revolution/d_revolution)

    # x-axis for order spectrum
    o = fftfreq(n, d=d_revolution)[1:n//2]
    roi = o < max_order
    step = int(n/30)

    # result varibale
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

    return o[roi], specy


'''
#################      function test        ####################

# ------------------- speed calibration test -------------------- #

path = 'D:/SpeedFromAmplitudeV2.0/data/dataGet/test01/'
file = path + 'test01_3mm_24k_210802064516.tdms'

data = TdmsFile.read(file)
vib = data['AIData']['Mic'].data
fs = 102400

speed = speed_calibration(vib[102400:102400*5], 24000, 13, fs=fs)
o, r = get_order_spectrum(vib, dr, 30, max_order=50, normalize=True, title='')
'''
