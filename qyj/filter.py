import numpy as np
import os
from timeit import timeit
import multiprocessing as mp
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz, freqs

fs = 102400

x = np.arange(0, 1, step=1/fs)
y = []

for i in range(5, 15):
    s = np.sin(2*np.pi*i*x)
    y.append(s)


tar = np.array([np.sin(2*np.pi*i*x) for i in range(5, 15)]).sum(axis=0)
sig = np.array(y).sum(axis=0)

fcut = 1 / fs * 2

b, a = butter(1, fcut, 'highpass')
filted = filtfilt(b, a, sig)


plt.plot(filted, label='filted')
plt.plot(tar, label='target')
plt.legend(loc=0)
plt.show()