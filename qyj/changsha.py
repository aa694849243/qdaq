from tools import *
import os

'''
############ 3in1 data #############
path = 'D:/SpeedFromAmplitudeV2.0/data/长沙数据/'
file_ls = os.listdir(path)
file = []

for f in file_ls:
    if f.split('.')[1] == 'tdms':
        file.append(f)


file_path = path + file[3]
tdms = TdmsFile.read(file_path)
vib = tdms['AIData']['Vibration'].data


nfft = int(8192 * 2)
step = 1024

fftlist = []
timelist = []

for i in range(len(vib) // step):
    t = i * step / 102400
    if (92 > t) or (t > 104):
        pass#continue
    frame = vib[i*step:i*step+nfft]
    nadd = np.zeros(nfft - len(frame))
    frame =  np.concatenate((frame, nadd))
    #frame = np.pad(frame, pad_width=8192*2)
    ffty = np.log(abs(fft(frame)))[1:len(frame)//2]
    fq = fftfreq(len(frame), d=1/102400)[1:len(frame)//2]
    roif = fq[fq < 2000]
    roifft = ffty[fq < 2000]
    fftlist.append(roifft)
    timelist.append(t)


plt.pcolormesh(timelist, roif, np.array(fftlist).T)
plt.title(file_path)


pos = plt.ginput(n=2, timeout=-1)
t1, y1 = pos[0]
t2, y2 = pos[1]
k = (y1 - y2) / (t1 - t2)
b = y1 - k * t1
t = np.arange(t1, t2, 0.01)
rpm = (k * t + b) / 24
plt.close()

cnt = get_revolution_by_speed(t[0], t[-1], rpm[0], rpm[-1], 0.0001)

loc = (np.array(cnt) * 102400).astype(np.int64)
vib_spec = vib[loc]

get_order_spectrum(vib_spec, 0.0001, 10, max_order=50, title=file_path)
get_order_cut(vib, t, rpm * 60, target_order=25, title=file_path)
'''
###### DM data ######
path = 'D:/SpeedFromAmplitudeV2.0/data/长沙数据/DM_test2/'
file_ls = os.listdir(path)
file = []

for f in file_ls:
    if f.split('.')[1] == 'tdms':
        file.append(f)


file_path = path + file[-1]
tdms = TdmsFile.read(file_path)
vib = tdms['AIData']['Vibration'].data


nfft = int(8192 * 4)
step = 1024

fftlist = []
timelist = []

for i in range(len(vib) // step):
    t = i * step / 102400
    if (92 > t) or (t > 104):
        pass#continue
    frame = vib[i*step:i*step+nfft]
    nadd = np.zeros(nfft - len(frame))
    frame =  np.concatenate((frame, nadd))
    #frame = np.pad(frame, pad_width=8192*2)
    ffty = np.log(abs(fft(frame)))[1:len(frame)//2]
    fq = fftfreq(len(frame), d=1/102400)[1:len(frame)//2]
    roif = fq[fq < 2000]
    roifft = ffty[fq < 2000]
    fftlist.append(roifft)
    timelist.append(t)


plt.pcolormesh(timelist, roif, np.array(fftlist).T)
plt.title(file_path)


pos = plt.ginput(n=2, timeout=-1)
t1, y1 = pos[0]
t2, y2 = pos[1]
k = (y1 - y2) / (t1 - t2)
b = y1 - k * t1
t = np.arange(t1, t2, 0.01)
rpm = (k * t + b) / 1
plt.close()

cnt = get_revolution_by_speed(t[0], t[-1], rpm[0], rpm[-1], 0.0001)

loc = (np.array(cnt) * 102400).astype(np.int64)
vib_spec = vib[loc]

get_order_spectrum(vib_spec, 0.0001, 10, max_order=50, title=file_path)
get_order_cut(vib, t, rpm * 60, target_order=25, title=file_path)