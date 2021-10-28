import numpy as np
from scipy.fft import fft, ifft, fftfreq, irfft
import matplotlib.pyplot as plt
from tools import *
from scipy import signal
import matplotlib.colors as colors
from scipy.signal import butter, filtfilt, freqz, freqs
from colormapTool import *
import scipy.integrate as si
import cmath


def butter_filter(signal, wn, fs, order=3, btype='lowpass'):
    """
    function: signal filter user butter
    :param
    signal(list): input signal(before filtering)
    Wn(list): normalized cutoff frequency （one for high/low pass filter，two for band pass/stop filter)
    sampleRate(int): sample rate of input signal
    order(int)：filter order（generally the order higher， transition band narrower）default as 5
    btyte(str): filter type（low pass：'low'；high pass：'high'；band pass：'bandpass'；band stop：'bandstop'）
    default as low
    # analog: digital or analog filter，cutoff of analog filter is angular frequency，for digital is relative
    frequency. default is digital filter
    :return
        signal after filter
    """
    nyq = fs / 2  # Nyquist's Law
    if btype == 'lowpass' or btype == 'highpass':
        cutoff = wn[0]/nyq
        b, a = butter(order, cutoff, btype, analog=False)
    else:
        lowcut = wn[0]/nyq
        highcut = wn[1]/nyq
        b, a = butter(order, [lowcut, highcut], btype, analog=False)
    return filtfilt(b, a, signal)


############################################### phase by pulse ##################################
input_path = 'D:/SpeedFromAmplitudeV2.0/data/Inovance/90003703_191012085221.tdms'
vib, speed = read_tdms(input_path)

ts = 15.2
pulse = np.where(speed < 2.5, 0, 1)
time, rpmt = speed_cal2(speed, 64)

pulse_s = np.where(np.diff(pulse) == 1)[0]
pulse_s = pulse_s[pulse_s > ts * 102400]
pulse_e = np.where(np.diff(pulse) == -1)[0]
pulse_e = pulse_e[pulse_e > pulse_s[0]]

ang = []

n = 64 * 3
for i in range(n):
    start = pulse_s[i]
    rpm = rpmt[np.argmin(abs(np.array(time) - start/102400))]
    frame = vib[start:start+8192*16]

    freq_l = rpm / 60 * 0.5
    freq_r = rpm / 60 * 1.5

    fft_v = fft(frame)[1:len(frame)//2]
    fft_f = fftfreq(n=len(frame), d=1/102400)[1:len(frame)//2]

    freq_t = (fft_f < freq_r) & (fft_f > freq_l)
    amp_t = fft_v[freq_t]

    val = amp_t[np.argmax(abs(amp_t))]
    phase = np.angle(val) / np.pi * 180.0
    ang.append(phase)

fig, ax = plt.subplots(nrows=1, ncols=3)
ax[0].plot(ang, marker='*')
ax[0].set_xlabel('pulse number')
ax[0].set_ylabel('Degree ($\circ$)')

ax[1].plot(np.arange(pulse_s[0]-50, pulse_s[n+5])/102400, pulse[pulse_s[0]-50:pulse_s[n+5]])
ax[1].scatter(pulse_s[:n]/102400, [1,]*n, marker='*', color='r')
ax[1].set_xlabel('time (s)')

dangle = np.diff(ang) % 360
ax[2].plot(dangle, marker='*', zorder=1)
ax[2].scatter(np.where(abs(dangle - 5.625) < 0.5)[0], dangle[np.where(abs(dangle - 5.625) < 0.5)[0]], color='r', marker='*', zorder=2)
ax[2].set_xlabel('angle difference')
ax[2].set_ylabel('Degree ($\circ$)')
plt.show()


################################## phase by time ##############################################
angle = []
tplot = []
rpmplot = []

for i in range(50):
    t = ts + i * 0.01
    rpm = rpmt[np.argmin(abs(np.array(time) - t))]
    start = int(t*102400)
    end = int(t*102400+8192*16)
    frame_vib = vib[start:end]
    frame_pulse = np.where(speed[start:end] < 2.5, 0, 1)
    
    freq_l = rpm / 60 * 0.5
    freq_r = rpm / 60 * 1.5
    
    fft_v = fft(frame_vib)[1:len(frame_vib)//2]
    fft_f = fftfreq(n=len(frame_vib), d=1/102400)[1:len(frame_vib)//2]

    
    freq_t = (fft_f < freq_r) & (fft_f > freq_l)
    amp_t = fft_v[freq_t]

    val = amp_t[np.argmax(abs(amp_t))]
    phase = np.angle(val) / np.pi * 180
    angle.append(phase)
    tplot.append(t)
    rpmplot.append(rpm)

idx = (np.array(tplot) * 102400).astype(int)
pulse_loc = pulse[idx]


fig, ax = plt.subplots(nrows=1, ncols=3)
lns1 = ax[0].plot(tplot, angle, label='angle', marker='*')
#ax2 = ax[0].twinx()
#lns2 = ax2.plot(tplot, rpmplot, color='black', label='rpm')
lns = lns1# + lns2
labs = [l.get_label() for l in lns]
ax[0].legend(lns, labs, loc='upper left')


ax[1].plot(np.arange(idx[0], idx[-1]+1)/102400, pulse[idx[0]:idx[-1]+1], label='pulse')
ax[1].scatter(tplot, pulse_loc, marker='*', color='r')


ax3 = ax[2].twinx()
ax3.plot(tplot, rpmplot, label='rpm')
#ax3.plot(tplot[1:], (np.diff(angle) % 360) / 360 / 0.01 * 60, label='rpm by angle')
ax[2].scatter(tplot[1:], np.diff(angle) % 360, marker='*', label='angle difference')
ax[2].legend(loc=0)

plt.show()


#################### phase by order ##############################
input_path = 'D:/SpeedFromAmplitudeV2.0/data/Inovance/90003703_191012085221.tdms'
vib, speed = read_tdms(input_path)
time, rpmt = speed_cal2(speed, 64)

fs = 102400

ts, te = 30, 145

interval = vib[ts*102400:te*102400]
n = 8192 * 10
step = int(n * 0.6)
ls = []
r = []
rr = []

for i in range(50):
    t = ts + 0.01 * i
    rpm = rpmt[np.argmin(abs(np.array(time) - t))]
    frame = interval[i*step:i*step+n]
    low_cut = rpm * .9 / 60 / fs * 2
    high_cut = rpm * 1.1 / 60 / fs * 2

    r.append(round(rpm, 2))

    b, a = butter(8, high_cut, 'lowpass')
    filted = filtfilt(b, a, frame)
    b, a = butter(8, low_cut, 'highpass')
    filted = filtfilt(b, a, filted)

    b, a = butter(8, high_cut, 'lowpass')
    filted = filtfilt(b, a, filted)
    b, a = butter(8, high_cut, 'lowpass')
    filted = filtfilt(b, a, filted)

    roi = filted[int(0.2*n):int(0.8*n)]
    ls.extend(filted[int(0.2*n):int(0.8*n)])
    peaks, _ = signal.find_peaks(roi)
    dp = np.diff(peaks)
    sp = 1 / (dp / 102400) * 60
    rr.append(np.mean(sp))

    fft_vib = fft(filted)[1:len(filted)//2]
    fft_freq = fftfreq(len(filted), d=1/102400)[1:len(filted)//2]

    roi = fft_freq < 2000
    roi_freq = fft_freq[roi]
    roi_vib = fft_vib[roi]