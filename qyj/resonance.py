from tools import *

input_path = 'D:/SpeedFromAmplitudeV2.0/data/jinkang_Testprofile1/YS35X900S-D200C03-ON201909120015_191110022010.tdms'

vib, speed = read_tdms(input_path)
x, y = speed_cal2(speed, 64)

step = 1024 * 10
n_frame = 8192 * 4
fs = 102400
vibList = []

for i in range(len(vib) // step):
    t = i * step / fs
    frame = vib[i*step:i*step+n_frame]
    if len(frame) < n_frame:
        continue
    vib_fft = abs(fft(frame))[1:len(frame)//2] / len(frame) * 2
    vib_freq = fftfreq(n=len(frame), d=1/fs)[1:len(frame)//2]

    roi_vib = vib_fft[vib_freq <= 20000]
    roi_freq = vib_freq[vib_freq <= 20000]

    vibList.append(roi_vib)


plt.plot(roi_freq, np.mean(vibList, axis=0))
plt.show()