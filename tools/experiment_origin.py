import numpy as np
from scipy.fft import fft, ifft, fftfreq, irfft
import matplotlib.pyplot as plt
from tools import *
from scipy import signal
import matplotlib.colors as colors
from scipy.signal import butter, filtfilt, freqz, freqs
import scipy.integrate as si
import cmath


############################################### phase by pulse ##################################
fs=102400
input_path = 'D:/qdaq/rawdata/90003703_191012085221.tdms'
vib, speed = read_tdms(input_path)

# plt.figure("speed")
# plt.plot(speed)
# plt.show()
ts = 15.2
pulse = np.where(speed < 2.5, 0, 1)
time, rpmt = speed_cal2(speed, 64,fs=102400)

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
    # freq_l = rpm / 60 * 59.5
    # freq_r = rpm / 60 * 60.5

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
# dangle =360-(np.diff(ang) % 360)
ax[2].plot(dangle, marker='*', zorder=1)
ax[2].scatter(np.where(abs(dangle - 5.625) < 0.5)[0], dangle[np.where(abs(dangle - 5.625) < 0.5)[0]], color='r', marker='*', zorder=2)
ax[2].set_xlabel('angle difference')
ax[2].set_ylabel('Degree ($\circ$)')
plt.show()