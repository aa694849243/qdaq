from pydub import AudioSegment
from pydub.playback import play
import matplotlib.pyplot as plt
import wavio
import numpy as np
from scipy import signal
from tools import *
from pydub.silence import split_on_silence

path = 'D:/SpeedFromAmplitudeV2.0/data/声音文件 from 奇瑞 20210801/'

f1 = 'yichang.wav'
f2 = 'zhengchang.wav'
f3 = '9月9日 08点22分，车内异响.wav'

data = wavio.read(path+f1).data
data = np.ravel(data)

fs = 48000
n_frame = 8192
step = n_frame
#step = 4800

l, r = wavio.read(path+f3).data.T
y = []

freq = fftfreq(n_frame, d=1/fs)
idx = (-9000 < freq) & (freq < 9000)
for i in range(len(l) // step):
    frame = l[i*step:i*step+n_frame]
    if len(frame) < n_frame:
        continue
    yfft = fft(frame)
    yfft[idx] = 0
    reversey = ifft(yfft)
    y = np.concatenate((y, np.real(reversey)))

#y = y.astype(np.int8)
wavio.write(path+'noise.wav', y, rate=48000, sampwidth=1)

get_colormap(y, [0], [0], fig_type='time-frequency', n_frame=8192*2, pad_width=0, 
                 resolution_level=1, window='hanning', fs=fs,
                 roi_time=None, roi_freq=None, title='left channel')

b, a = signal.butter(5, 2*12000/fs, btype='lowpass', analog=False, output='ba')
data = signal.lfilter(b, a, l) 

wavio.write(path+'noise.wav', data, rate=fs, sampwidth=1)

song = AudioSegment.from_wav(path+f3)

x = np.linspace(0, len(data)/fs, num=3000)
y = np.linspace(1000, 0, num=3000)

get_colormap(data, x, y, fig_type='time-frequency', n_frame=8192*2, pad_width=0, 
                 resolution_level=1, window='hanning', fs=fs,
                 roi_time=None, roi_freq=None, title='left channel')

