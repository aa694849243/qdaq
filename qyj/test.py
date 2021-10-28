from scipy import stats
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nptdms import TdmsFile
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from speedFromVib import *
from tools import *
import time
from scipy.signal import butter, filtfilt


def read_tdms(file_path):
    file = TdmsFile.read(file_path)
    vib = file['AIData']['Vibration'].data
    speed = file['AIData']['Speed'].data
    return vib, speed

def read_new_tdms(file_path):
    file = TdmsFile.read(file_path)
    vib1 = file['AIData']['Vib1'].data
    vib2 = file['AIData']['Vib2'].data
    sin = file['AIData']['Sin'].data
    cos = file['AIData']['Cos'].data
    return vib1, vib2, sin, cos

def butter_filter(signal, wn, fs, order=3, btype='lowpass'):
    nyq = fs / 2  # Nyquist's Law
    if btype == 'lowpass' or btype == 'highpass':
        cutoff = wn[0]/nyq
        b, a = butter(order, cutoff, btype, analog=False)
    else:
        lowcut = wn[0]/nyq
        highcut = wn[1]/nyq
        b, a = butter(order, [lowcut, highcut], btype, analog=False)
    return filtfilt(b, a, signal)

def intercnt(cnt, vib):
    #sos = butter(20, 25000, output='sos', fs=102400)
    #vib = signal.sosfilt(sos, vib)
    #vib = butter_filter(vib, [1, 20000], 102400, btype='bandpass')
    loc = np.array(cnt) * 102400
    #revib = vib[loc.astype(np.int32)]
    fracp, intp = np.modf(loc)
    intp = intp.astype(np.int64)
    revib = vib[intp] + np.diff(vib)[intp] * fracp

    return revib


input_path = 'D:/SpeedFromAmplitudeV2.0/data/order_cut/017700944M600034_200628061633.tdms'

vib, speed = read_tdms(input_path)

sos = butter(20, 20000, output='sos', fs=102400)
vib = signal.sosfilt(sos, vib)

refx, refy = get_speed_from_vib(vib, 'type2')
refx, refy = np.array(refx), np.array(refy)
x, y = speed_cal2(speed, 64)
x, y = np.array(x), np.array(y)

roix = x[(x >= 0.3) & (x <= 23.5)]
roiy = y[(x >= 0.3) & (x <= 23.5)]

roi = (refx >= 0.3) & (refx <= 23.5)
roit = refx[roi]
roirpm = refy[roi]

res = stats.linregress(roit, roirpm)
k, b = res.slope, res.intercept

rpm = k * roit + b

nr = 0.2
cnt = get_revolution_by_speed(roit[0], roit[-1], roirpm[0], roirpm[-1], nr)

loc = (np.array(cnt) * 102400).astype(np.int64)
#sos = butter(1 , 50000, output='sos', fs=102400)
#vib = signal.sosfilt(sos, vib)
revib = intercnt(cnt, vib)
yy = []

n = int(30*60/nr)

pad_width = int(0.5*n)
nn = n + 2*pad_width

o = fftfreq(nn, d=nr/60)[1:nn//2]
roi = o < 500

step = int(n/30)

for i in range(len(revib)//step):
    frame = revib[i*step:i*step+n]
    if len(frame) < n:
        continue
    frame = np.pad(frame, pad_width=pad_width)
    fy = abs(fft(frame * np.hanning(len(frame))))[1:len(frame)//2] / np.sqrt(len(frame))
    yy.append(fy[roi])


#get_colormap(vib, x, y, fig_type='frequency-rpm', n_frame=8192*4, pad_width=8192*2, 
#             resolution_level=1, window='hanning', fs=102400,
#             roi_time=[0.3, 24], roi_freq=[0, 20000], title='')

plt.plot(o[roi], np.mean(yy, axis=0))
plt.show()


#vib1, vib2, sin, cos = read_new_tdms(input_path)
#rms_list = []
#vib = vib2
'''
step = int(8192*2*0.8)
n_frame = 8192 * 2
order = [40, 48]
resl = 0.25
time = []
for t in refx[::2]:
    if t > 23.8:
        continue
    time.append(t)
    frame = vib[int(t*102400):int(t*102400+n_frame)]
    frame = np.pad(frame, pad_width=8192*2)
    window = np.hanning(len(frame))
    vib_fft = abs(fft(frame * window))[1:len(frame)//2] / (len(frame)/2) * 1.414
    freq = fftfreq(len(frame), d=1/102400)[1:len(frame)//2]
    for o in order:
        fl0,10 fr = (o - resl) * (7500/24.035*t+2500) / 60, (o + resl) * (7500/24.035*t+2500) / 60
        vib_obj = vib_obj = vib_fft[(freq < fr) & (freq > fl)]
        rms = np.sqrt(np.mean(vib_obj ** 2))
        rms_list.append(rms)

rms_list = np.array(rms_list).reshape(-1, 2)
g
for i in range(2):
    plt.plot(time, rms_list[:,i], label='order{}'.format(order[i]))
plt.legend(loc=0)
plt.show()
'''

'''
# 读取qDAQ结果数据
json_file_name = 'ino1vib-ttl-Debug210326_017700944M600034_200628061633_210330012452.json'
with open(json_file_name, 'r') as f:
    data = json.load(f)

order_list = [x['yName'] for x in data['resultData'][0]['dataSection'][0]['twodOC']]
order = [o.replace('ord_2D', '').replace('ords_2D', '') for o in order_list]


# 读取阶次谱
twdos = data['resultData'][0]['dataSection'][0]['twodOS'][0]

#plt.plot(twdos['xValue'], twdos['yValue'], label=twdos['yName'])
#plt.plot(o[roi], np.mean(yy,axis=0), label='cal')
#plt.legend()
#plt.show()

# 读取40阶阶次切片
ord_40 = data['resultData'][0]['dataSection'][0]['twodOC'][7]

# 读取48阶阶次切片
ord_48 = data['resultData'][0]['dataSection'][0]['twodOC'][8]
'''

rms_list = []
speed_list = []
t = []
oa = []
n_frame = int(8192*2.5)
step = 10240 * 2

roi = (refx >= 0.2) & (refx <= 24)
roit = refx[roi]
roirpm = refy[roi]

res = stats.linregress(roit, roirpm)
k, b = res.slope, res.intercept
# get order spec
order = [4, 8, 16, 24, 32, 40, 48, 96]
for i in range(len(vib) // step):
    times = step * i / 102400
    if times > 24 or times < 0.2:
        continue
    t.append(times)

    rpm = k * times + b
    speed_list.append(rpm)
    frame = vib[i*step:i*step+n_frame]
    pad_width = 8192 * 2
    frame = np.pad(frame, pad_width=pad_width, mode='constant', constant_values=0)
    window = np.hanning(len(frame))
    vib_fft = abs(fft(frame * window))[1:len(frame)//2] / (len(frame)/2) / 1.414
    freq = fftfreq(len(frame), d=1/102400)[1:len(frame)//2]
    oa.append(vib_fft)
    pass
    for o in order:
        freq_left = (o - 0.4) * rpm / 60
        freq_right = (o + 0.4) * rpm / 60
        vib_obj = vib_fft[(freq <= freq_right) & (freq >= freq_left)]
        rms = np.sqrt(np.sum(vib_obj ** 2))
        rms_list.append(rms)

rms_list = np.array(rms_list).reshape(-1, 8)
oa = np.array(oa).reshape(-1, len(frame)//2-1)
end = time.time()

#plt.plot(twdos['xValue'], twdos['yValue'], label=twdos['yName'])
#plt.legend()
#plt.show()
'''
for i in range(8):
    plt.plot(speed_list, rms_list[:,i], label='order{}'.format(order[i]))
#    dic = data['resultData'][0]['dataSection'][0]['twodOC'][i+2]
#    x = dic['xValue']
#    y = dic['yValue']
#    plt.plot(x, y, label=dic['yName'])
    plt.legend()
    plt.show()
'''
'''
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))

ax[0].plot(speed_list, rms_list[:,0], label='order{}'.format(order[0]))
ax[0].plot(ord_40['xValue'], ord_40['yValue'], label='q')
ax[0].legend(loc=0)

ax[1].plot(speed_list, rms_list[:,1], label='order{}'.format(order[1]))
ax[1].plot(ord_48['xValue'], ord_48['yValue'], label='q')
ax[1].legend(loc=0)
plt.show()
'''
