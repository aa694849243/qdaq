from tools import *
import os
import pandas as pd
import wavio
from scipy.io import wavfile
import scipy.signal as signal

cali = 'D:/SpeedFromAmplitudeV2.0/data/21081806/'
file_cali = os.listdir(cali)

def float_to_db(data):
    p_ref = 20 * 10**-6
    frame = 1024
    db = []
    for i in range(len(data) // frame):
        d = data[i*frame:i*frame+frame]
        p_rms = np.sqrt(np.mean(d**2))
        db.append(20*np.log10(p_rms / p_ref))
    return db


# read frf data in cali file
frf = []
with open ('7081617.txt') as f:
    frf = f.readlines()

freq, res = [], []
for data in frf[1:]:
    if data.strip():
        f, r = data.strip().split('\t')
        freq.append(float(f))
        res.append(float(r))


data1 = TdmsFile.read(cali+'1000Hz1_std.tdms')
mic1 = data1['AIData']['Mic'].data

data2 = TdmsFile.read(cali+'1000Hz2_std.tdms')
mic2 = data2['AIData']['Mic'].data


fy1 = 20*np.log10(abs(fft(mic1))[1:len(mic1)//2] / len(mic1) * 2 * 0.707 / 20e-6)

fy2 = abs(fft(mic2))[1:len(mic2)//2] / len(mic2) * 2

f1 = fftfreq(len(mic1), d=1/102400)[1:len(mic1)//2]

f2 = fftfreq(len(mic2), d=1/102400)[1:len(mic2)//2]

target1 = (f1 > 1000-10) & (f1 < 1000+10)
target2 = (f2 > 1000-10) & (f2 < 1000+10)

amp1 = max(fy1[target1])
amp2 = max(fy2[target2])

plt.plot(f1, fy1)
#plt.title(amp1)
plt.show()

y = []
for i in range(len(mic1)//8192):
    frame = mic1[i*1024:i*1024+8192]
    if len(frame) < 8192:
        continue
    yy = (abs(fft(frame))[1:len(frame)//2] / len(frame) * 2)
    y.append(yy)

plt.plot(fftfreq(8192,d=1/102400)[1:4096], 20*np.log10(np.mean(y,axis=0) * 0.707 / 20e-6))
plt.show()

#plt.plot(f2, fy2)
#plt.title(amp2)
#plt.show()

sample0 = pd.read_csv('sample0-std.csv')
sample0 = sample0['0']
s0 = sample0[60000:130000]

sample1 = pd.read_csv('sample1-std.csv')
sample1 = sample1['0']
s1 = sample1[60000:130000]

plt.plot(s1)
plt.show()

sf0 = abs(fft(s0))[1:len(s0)//2] / len(s0) * 2
sq0 = fftfreq(len(s0), d=1/48000)[1:len(s0)//2]
t0 = (sq0 > 1000-10) & (sq0 < 1000+10)
a0 = max(sf0[t0]) 

#plt.plot(sq0, sf0)
#plt.title(a0)
#plt.show()

sf1 = abs(fft(s1))[1:len(s1)//2] / len(s1) * 2
sq1 = fftfreq(len(s1), d=1/48000)[1:len(s1)//2]
t1 = (sq1 > 1000-10) & (sq1 < 1000+10)
a1 = max(sf1[t1])

#plt.plot(sq1, sf1)
#plt.title(a1)
#plt.show()

coef = 201

s0_c = s0 / coef

plt.plot((np.arange(len(mic1)) / 102400)[:len(mic1)//2], mic1[:len(mic1)//2])
plt.plot(np.arange(len(s0_c)) / 48000, s0_c)
plt.show()