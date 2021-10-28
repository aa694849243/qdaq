import matplotlib.pyplot as plt
import numpy as np
from tools import *
from speedFromVib import *
from scipy import stats, signal


def KalmanFilter(z):        
    # intial parameters  
    
    n_iter = len(z)
    sz = (n_iter,) # size of array   

    Q = 1e-4 # process variance   
    # allocate space for arrays  
    xhat=np.zeros(sz)      # a posteri estimate of x  
    P=np.zeros(sz)         # a posteri error estimate  
    xhatminus=np.zeros(sz) # a priori estimate of x  
    Pminus=np.zeros(sz)    # a priori error estimate  
    K=np.zeros(sz)         # gain or blending factor  
      
    R = 0.1**3 # estimate of measurement variance, change to see effect  
      
    # intial guesses  
    xhat[0] = 0.0  
    P[0] = 1.0
    A = 1
    H = 1

    for k in range(1,n_iter):  
        # time update  
        xhatminus[k] = A * xhat[k-1]  #X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0  
        Pminus[k] = A * P[k-1]+Q      #P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1  
      
        # measurement update  
        K[k] = Pminus[k]/( Pminus[k]+R ) #Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1  
        xhat[k] = xhatminus[k]+K[k]*(z[k]-H * xhatminus[k]) #X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1  
        P[k] = (1-K[k] * H) * Pminus[k] #P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1  
    return xhat

def get_motor_noise(vib, center_frequency, order_interval, time, speed, n_line=5, fs=102400):
    step = 10240
    order_list = [order_interval * (n + 0.5) for n in range(-n_line, n_line)]
    n_frame = int(8192 * 2.5)
    pad_width = 8192 * 2
    rms_list = []
    speed_list = []

    for i in range(len(vib) // step):
        t = step / fs * i
        idx = np.argmin(abs(np.array(time) - t))
        rpm = speed[idx]
        frame = vib[i*step:i*step+n_frame]
        frame = np.pad(frame, pad_width=pad_width, mode='constant', constant_values=0)
        window = np.hanning(len(frame))
        vib_fft = abs(fft(frame * window))[1:len(frame)//2] / (len(frame)/2) / 1.414
        freq = fftfreq(len(frame), d=1/102400)[1:len(frame)//2]

        vib_fft = vib_fft[freq <= 20000]
        freq = freq[freq <= 20000]
        speed_list.append(rpm)

        for o in order_list:
            freq_left = o * rpm / 60 + center_frequency - 0.25 * rpm / 60
            freq_right = o * rpm / 60 + center_frequency + 0.25 * rpm / 60
            vib_obj = vib_fft[(freq <= freq_right) & (freq >= freq_left)]
            
            rms = np.sqrt(np.sum(vib_obj ** 2))
            rms_list.append(rms)

    rms_list = np.array(rms_list).reshape(-1, len(order_list))
    print(rms_list[-1,:])
    label = [x for x in range(-n_line, n_line+1) if x != 0]
    for i in range(n_line*2):
        rmsy = rms_list[:,i]
        plt.plot(speed_list, rmsy, label='line {}'.format(label[i]))
        plt.title('noise for target line')
        plt.ylabel('rms value')
        plt.xlabel('speed (rpm)')
        plt.legend(loc=0)
        plt.show()


input_path = 'D:/SpeedFromAmplitudeV2.0/data/order_cut/017700944M600034_200628061633.tdms'

vib, speed = read_tdms(input_path)
x, y = speed_cal2(speed, 64)

refx, refy = get_speed_from_vib(vib, 'type2')
refx, refy = np.array(refx), np.array(refy)

rms_list = []
speed_list = []
t = []
oa = []
n_frame = int(8192*2.5)
step = 1024 * 10

roi = (refx >= 0.2) & (refx <= 24)
roit = refx[roi]
roirpm = refy[roi]

res = stats.linregress(roit, roirpm)
k, b = res.slope, res.intercept
slope = [0.134*i for i in [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]]
vmx = []
maxpct = []
l1, l2 = [], []
rms96 = []
ll96, lr96 = [], []
rmst = []

get_motor_noise(vib, center_frequency=10000, order_interval=8,
                time=np.arange(0.2, 24, 0.1), speed=np.arange(0.2, 24, 0.1)*k+b)   

# get order spec
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

    vib_fft = vib_fft[freq <= 20000]
    freq = freq[freq <= 20000]

    l96 = rpm * (72 - 0.1) / 60
    r96 = rpm * (72 + 0.9) / 60
    vib96 = vib_fft[(freq <= r96) & (freq >= l96)]
    rms96.append(np.sqrt(np.sum(vib96 ** 2)))
    ll96.append(l96)
    lr96.append(r96)

    for o in slope:
        freq_left = o * rpm + 10000 - 0.2 * rpm / 60
        freq_right = o * rpm + 10000 + 0.2 * rpm / 60
        left_broad = o * rpm + 10000 - rpm / 60
        right_broad = o * rpm + 10000 + rpm / 60

        vib_obj = vib_fft[(freq <= freq_right) & (freq >= freq_left)]
        vib_broad = vib_fft[(freq <= right_broad) & (freq >= left_broad)]

        vmax = max(vib_obj)
        rms = np.sqrt(np.sum(vib_obj ** 2))
        rms_broad = np.sqrt(np.sum(vib_broad ** 2))

        mpct = rms_broad / rms
        rms_list.append(rms)
        vmx.append(vmax)
        maxpct.append(mpct)
        l1.append(freq_left)
        l2.append(freq_right)

rms_list = np.array(rms_list).reshape(-1, len(slope))
print(rms_list[-1,:])
v1 = vmx[1:] + [0]
ratio_v = [i/j for (i, j) in zip(v1, vmx)]

for i in range(len(slope)):
    plt.plot(speed_list, rms_list[:,i], label='rms')
    plt.plot(speed_list, KalmanFilter(rms_list[:,i]), label='filter')
    plt.legend(loc=0)
    plt.show()

    plt.plot(speed_list, KalmanFilter(rms_list[:,i]) / rms_list[:,i])
    plt.show()


lf = []

for i in range(len(slope)):
    s = KalmanFilter(rms_list[:,i])
    lf.append(sum(s))

plt.plot(np.sum(rms_list, axis=0), label='before')
plt.plot(lf, label='after')
plt.title('before v.s after kalman filter \n total:{:.2f}'.format(np.sum(rms_list)))
plt.ylabel('rms')
plt.show()

plt.plot(np.diff(lf - np.sum(rms_list, axis=0)))
plt.title('diff before and after filter')
plt.show()

plt.plot(lf/np.sum(rms_list, axis=0))
plt.title('ratio before and after filter')
plt.show()

plt.plot(speed_list, rms96, label='rms96')
plt.plot(speed_list, KalmanFilter(rms96), label='kalman')
plt.legend()
plt.show()

rms0 = list(rms_list[:,0])
rms1 = rms0[1:] + [0]
ratio_rms = [i/j for (i, j) in zip(rms1, rms0)]
plt.plot(speed_list, ratio_v, label='max ratio')
plt.plot(speed_list, ratio_rms, label='rms ratio')
plt.legend(loc=0)
plt.show()

rmss = []
for i in range(len(rms0)):
    x = np.mean(rms0[i:i+10])
    rmss.append(x)

plt.plot(speed_list, rms0, label='rms')
plt.plot(speed_list, KalmanFilter(rms0), label='kalman')
plt.title('rms avg')
plt.legend()
plt.show()