import numpy as np
from scipy.fft import fft, ifft, fftfreq, irfft
import matplotlib.pyplot as plt
from tools import *
from scipy import signal
import matplotlib.colors as colors
from scipy.signal import butter, filtfilt, freqz, freqs
import scipy.integrate as si
import cmath
import time


############################################### phase by pulse ##################################
fs=102400
input_path = 'D:/qdaq/rawdata/90003703_191012085221.tdms'

vib, speed = read_tdms(input_path)
# vib=vib[int(38.5*fs):int(41.5*fs)]
# speed=speed[int(38.5*fs):int(41.5*fs)]

# speed:1914 1920
vib=vib[int(21*fs):int(23.5*fs)]
speed=speed[int(21*fs):int(23.5*fs)]
ts = 15.2
pulse = np.where(speed < 2.5, 0, 1)
t, rpmt = speed_cal2(speed, 64,n_avg=256, fs=fs)

# plt.figure("rpm")
# plt.plot(t,rpmt)
# # plt.figure("speed")
# # plt.plot(speed)
# plt.show()
pulse_s = np.where(np.diff(pulse) == 1)[0]
# pulse_s = pulse_s[pulse_s > ts * 102400]
# pulse_e = np.where(np.diff(pulse) == -1)[0]
# pulse_e = pulse_e[pulse_e > pulse_s[0]]
# plt.figure("time_frequency")
# get_timefrequency_colormap(vib,n_frame=8192)

ang = []

n = 64*3

rpm_v=list()
for i in range(n):
    start = pulse_s[i]
    rpm = rpmt[np.argmin(abs(np.array(t) - start / 102400))]
    if start+8192*16>len(vib):
        time.sleep(0.1)
    frame = vib[start:start+8192*16]

    freq_l = rpm / 60 * 59.5
    freq_r = rpm / 60 * 60.5

    fft_v = np.fft.rfft(frame)
    fft_f = np.fft.rfftfreq(n=len(frame), d=1/102400)

    freq_t = (fft_f < freq_r) & (fft_f > freq_l)
    amp_t = fft_v[freq_t]
    rpm_v.append(fft_f[freq_t][np.argmax(abs(amp_t))]/60*60)
    val = amp_t[np.argmax(abs(amp_t))]
    phase = np.angle(val) / np.pi * 180.0
    # phase = np.angle(fft_v[30656]) / np.pi * 180.0
    ang.append(phase)

plt.figure("rpm_search")
plt.plot(rpm_v)
plt.show()
fig, ax = plt.subplots(nrows=1, ncols=3)
ax[0].plot(ang, marker='*')
ax[0].set_xlabel('pulse number')
ax[0].set_ylabel('Degree ($\circ$)')

# ax[1].plot(np.arange(pulse_s[0]-50, pulse_s[n+5])/102400, pulse[pulse_s[0]-50:pulse_s[n+5]])
# ax[1].scatter(pulse_s[:n]/102400, [1,]*n, marker='*', color='r')
# ax[1].set_xlabel('time (s)')

dangle = 360-(np.diff(ang) % 360)
# np.where(dangle>180,360-dangle,dangle)
ax[2].plot(dangle, marker='*', zorder=60)
ax[2].scatter(np.where(abs(dangle - 5.625) < 0.5)[0], dangle[np.where(abs(dangle - 5.625) < 0.5)[0]], color='r', marker='*', zorder=2)
ax[2].set_xlabel('angle difference')
ax[2].set_ylabel('Degree ($\circ$)')
plt.show()
print(1)