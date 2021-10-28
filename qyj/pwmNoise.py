from tools import *
import timeit
import time
import pandas as pd

def KalmanFilter(z):
    '''
    kalman filter
    *************************************************************
    parameters
             z: input array
    return
          xhat: array after filt
    '''        
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
      
    R = 0.1**4 # estimate of measurement variance, change to see effect  
      
    # intial guesses  
    xhat[0] = z[0]  
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

def calspeed(tacho):
    '''
    speed calculation for tacho signal
    *********************************************************
    parameters
         tacho: tacho signal
    return
      rev_time: time list for speed
           rpm: rpm list for speed
    '''
    
    cbp = tacho                             # cbp: counts between pulses
    ppr = 1024                              # ppr: pulses per revolution
    cps = 10**8                             # cps: counts per second
    cst = 75586                             # cst: counts at starting

    tbp = cbp / cps                         # tbp: time duration between pulses
    rpm = cps / cbp / ppr * 60              # rpm: revolutions per minute;
    rev = np.arange(1, len(cbp) + 1) / ppr  # rev: cumulative revolutions at pulses
    rev_time = cst / cps + np.cumsum(tbp)   # rev_time: cumulative time at pulses
    
    return rev_time, rpm

def controlNoiseCal(speed_time, speed_rpm, fs, micData, center_freq, lines, count=0, nfft=8192):
    '''
    calculate PWM rms value for motor control machine
    ************************************************************
    parameters
     speed_time: speed time
      speed_rpm: speed rpm value
             fs: sampling rate
           nfft: points for fft
           step: step between two ffts
        micData: microphone data
    center_freq: center frequency for PWM
          lines: orders for calculate of PWM
          count: for calculation by frame only
           nfft: points to do fft
    
    return
     speed_list: rpm value
      total_rms: sum of rms value for every speed
    '''

    speed_list, rms_list = [], []
    step = int(fs / 100 * 20)                                                                           # step between two fft, 0.2s

    for i in range(len(micData) // step):
        t = step * i / fs + count * step / fs                                                           # current time
        rpm = speed_rpm[np.argmin(abs(t - speed_time))]                                                 # find current speed
        speed_list.append(rpm)
        frame = micData[i*step:i*step+nfft]                                                             # extract data
        frame = np.concatenate([frame, np.zeros(nfft - len(frame))])                                    # pad zeros to same length
        window = np.hanning(len(frame))
        vib_fft = (abs(fft(frame * window))[1:len(frame)//2] / len(frame) * 1.414 * 1.633)              # 1.633 is energy coef for hanning window
        freq = fftfreq(len(frame), d=1/fs)[1:len(frame)//2]

        for o in lines:                                                                                 # calculate target orders of PWM
            center = center_freq + o * rpm / 60
            idx = np.argsort(abs(center - freq))[:8]                                                    # get 8 points around center frequency
            vib_obj = vib_fft[idx]
            rms = max(vib_obj ** 2)
            rms_list.append(rms)

    rms_list = np.array(rms_list).reshape(-1, len(lines))
    total_rms = np.sqrt(np.sum(rms_list, axis=1))
    
    return speed_list, total_rms

def NoiseByFrame(micData, fs, speedProfile_time, speedProfile_speed, center_freq, lines, count, nfft=8192):
    '''
    calculate PWM rms value for motor control machine
    ************************************************************
    parameters
       speedProfile_time: speed profile time
      speedProfile_speed: speed profile speed value
                      fs: sampling rate
                    nfft: points for fft
                    step: step between two ffts
                 micData: microphone data
             center_freq: center frequency for PWM
                   lines: orders for calculate of PWM
                   count: record time
                    nfft: fft points
    
    return
              speed_list: rpm value
               total_rms: sum of rms value for every speed
    '''
    if len(micData) < nfft:
        return [], [], count

    speed_list, rms_list = [], []
    step = int(fs / 100 * 20)                                                                           # step between two calculation, 0.2s
    micData = micData[count*step:]
    idx = list(range(count*step, count*step + len(micData)))
    speed_time, speed_rpm = getSpeedFromProfile(idx, fs, speedProfile_time, speedProfile_speed)
    
    if not speed_rpm.size:
        return [], [], count
    
    for i in range(len(micData) // step):
        t = step * i / fs + count * step / fs                                                           # current time
        rpm = speed_rpm[np.argmin(abs(t - speed_time))]                                                 # find current speed
        speed_list.append(rpm)
        frame = micData[i*step:i*step+nfft]                                                             # extract data
        frame = np.concatenate([frame, np.zeros(nfft - len(frame))])                                    # pad zeros to same length
        window = np.hanning(len(frame))
        vib_fft = (abs(fft(frame * window))[1:len(frame)//2] / len(frame) * 1.414 * 1.633)              # 1.633 is energy coef for hanning window
        freq = fftfreq(len(frame), d=1/fs)[1:len(frame)//2]

        for o in lines:                                                                                 # calculate target orders of PWM
            center = center_freq + o * rpm / 60
            idx = np.argsort(abs(center - freq))[:8]                                                    # get 8 points around center frequency
            vib_obj = vib_fft[idx]
            rms = max(vib_obj ** 2)
            rms_list.append(rms)

    rms_list = np.array(rms_list).reshape(-1, len(lines))
    total_rms = np.sqrt(np.sum(rms_list, axis=1))
    count += len(total_rms)

    return speed_list, total_rms, count

def calByStream(calFunc, **kwargs):
    '''
    calculation function stream by stream, could be noise, order cut, order spectrum or speed
    '''
    count = 0
    if calFunc == 'noise':
        speed, noise = controlNoiseCal(speed_time, speed_rpm, fs, micData, center_freq, lines, count, nfft)
        count += len(noise)
        return speed, noise, count
    if calFunc == 'order_cut':
        pass
    if calFunc == 'order_spectrum':
        pass
    if calFunc == 'speed':
        pass

def getSpeedFromProfile(idx, fs, speedProfile_time, speedProfile_speed):
    
    speedProfile_time = np.array(speedProfile_time)
    speedProfile_speed = np.array(speedProfile_speed)

    time = np.array(idx) / fs

    speed_time = speedProfile_time[(speedProfile_time >= min(time)) & (speedProfile_time <= max(time))]
    speed_rpm = speedProfile_speed[(speedProfile_time >= min(time)) & (speedProfile_time <= max(time))]

    return speed_time, speed_rpm

'''
#---------------------- test ---------------------#

############## read file #############
file_path = 'D:/SpeedFromAmplitudeV2.0/data/ZF/Data/4460061507/2004/20042302/S280120001_200423024601.tdms'
file = TdmsFile.read(file_path)
tick = file['Counters']['@AISampleTicks'].data
tacho = file['Counters']['Tacho'].data
tacho1 = file['Counters']['Tacho1'].data
vib = file['AIData']['Vibration'].data
mic = file['AIData']['Mikrofon'].data

############## speed calculation ################
t0, rpm0 = calspeed(tacho)
rpm0 = rpm0 * 9.7                               # 9.7 is speed coefficient, case for case

############## read reference data ###############
df = pd.read_csv('pwm.csv')
pwm1 = df['PWM1_Mic_2D']
speed = df['Tacho']

############## controlNoiseCal function test ###############
s, r = controlNoiseCal(t0[:230400], rpm0[:230400], fs=51200, micData=mic, center_freq=8000, lines=np.arange(-5, 6)*3)

############## time-consuming test ################
print(timeit.timeit('controlNoiseCal(t0[:230400], rpm0[:230400], fs=51200, micData=mic, center_freq=8000, lines=np.arange(-5, 6)*3)', globals=globals(), number=100))

############## comparison of calculation and reference data ###############
plt.plot(s, r, label='controlNoiseCal')
plt.plot(speed, pwm1, label='AQS')
plt.legend(loc=0)
plt.show()
'''

################# test calculation by stream ###################

# read file
file_path = 'D:/SpeedFromAmplitudeV2.0/data/ZF/Data/4460061507/2004/20042302/S280120001_200423024601.tdms'
file = TdmsFile.read(file_path)
tick = file['Counters']['@AISampleTicks'].data
tacho = file['Counters']['Tacho'].data
tacho1 = file['Counters']['Tacho1'].data
vib = file['AIData']['Vibration'].data
mic = file['AIData']['Mikrofon'].data

# speed
t, rpm = calspeed(tacho)
rpm = rpm * 9.7                               # 9.7 is speed coefficient, case for case
t, rpm = np.array(t), np.array(rpm)

# test for 11s to 29s
mic = mic[11*51200:29*51200]
rpm = rpm[(t >= 11) & (t <= 29)]
t = t[(t >= 11) & (t <= 29)]

# points each reading
n_frame = 15000

# generate simu data
mic_data = []
t_data = []
speed_data = []

for i in range(len(mic) // n_frame):
    time = (i+1) * n_frame / 51200
    mic_data.append(mic[:(i+1)*n_frame])

# result
x, y = [], []

# parameter setting
fs = 51200
count = 0
step = int(fs / 100 * 20)

# simulate stream data
for i in range(len(mic_data)):
    feed = mic_data[i]
    s, r, count = NoiseByFrame(micData=feed, fs=fs, speedProfile_time=t, speedProfile_speed=rpm, center_freq=8000, lines=np.arange(-5, 6) * 3, count=count)
    x.extend(s)
    y.extend(r)


s, r = controlNoiseCal(t, rpm, fs=fs, micData=mic, center_freq=8000, lines=np.arange(-5, 6)*3)

plt.plot(s, r, label='read total file')
plt.plot(x, y, label='frame by frame')
plt.legend(loc=0)
plt.show()
