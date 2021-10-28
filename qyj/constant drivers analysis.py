from pydub import AudioSegment
from pydub.playback import play
import matplotlib.pyplot as plt
import wavio
import numpy as np
from scipy import signal
from tools import *
import os
import pandas as pd
from scipy.signal import butter, filtfilt


def float_to_db(data):
    p_ref = 20e-6
    frame = 1024
    db = []
    data = np.array(data)
    for i in range(len(data) // frame):
        d = data[i*frame:i*frame+frame]
        p_rms = np.sqrt(np.mean(d**2))
        db.append(20*np.log10(p_rms / p_ref))
    return db

def butter_filter(signal, wn, fs, order=2, btype='lowpass'):
    """
    function: signal filter user butter
    :param
    signal(list): input signal(before filtering)
    Wn(list): normalized cutoff frequency （one for high/low pass filter，two for band pass/stop filter)
    sampleRate(int): sample rate of input signal
    order(int)：filter order（generally the order higher， transition band narrower）default as 5
    btyte(str): filter type（low pass：'low'；high pass：'high'；band pass：'bandpass'；band stop：'bandstop'）
    default as low
    # analog: digital or analog filter，cutoff of analog filter is angular frequency，for digital is relative
    frequency. default is digital filter
    :return
        signal after filter
    """
    nyq = fs / 2  # Nyquist's Law
    if btype == 'lowpass' or btype == 'highpass':
        cutoff = wn[0]/nyq
        b, a = butter(order, cutoff, btype, analog=False)
    else:
        lowcut = wn[0]/nyq
        highcut = wn[1]/nyq
        b, a = butter(order, [lowcut, highcut], btype, analog=False)
    return filtfilt(b, a, signal)

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
      
    R = 1e-2 # estimate of measurement variance, change to see effect  
      
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


'''
path = 'D:/SpeedFromAmplitudeV2.0/data/dongxingchang/'
bad_wav = path + 'bad2.wav'
good_wav = path + 'good.wav'

fbad = wavio.read(bad_wav)
left_wav, right_wav = fbad.data.T
stereo_wav = (left_wav + right_wav) / 2

n_revolution = 10
fs = 44100
rpm = 4000
dr = rpm / 60 / fs

o, lbad = get_order_spectrum(left_wav, dr, n_revolution, max_order=50, normalize=True, title='')
o, rbad = get_order_spectrum(right_wav, dr, n_revolution, max_order=50, normalize=True, title='')

fgood = wavio.read(good_wav)
left, right = fgood.data.T

plt.plot(np.arange(len(right_wav))/44100, right_wav/max(right_wav))
plt.title('right channel')
plt.xlabel('Time (s)')
plt.show()


o1, lgood = get_order_spectrum(left, dr, n_revolution, max_order=50, normalize=True, title='')
o1, rgood = get_order_spectrum(right, dr, n_revolution, max_order=50, normalize=True, title='')

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
ax[0].plot(o, rbad, label='right channel (big power)')
ax[1].plot(o1, rgood, label='right channel')
ax[0].legend()
ax[1].legend()
fig.suptitle('right channel comparison')
fig.supxlabel('order')
fig.supylabel('Relative RMS')
plt.show()


speed_x = np.arange(0, 63, 0.1)
speed_y = np.ones(len(speed_x)) * 4000
get_colormap(right, speed_x, speed_y, fig_type='time-frequency', n_frame=8192*2, pad_width=0, 
                 resolution_level=1, window='hanning', fs=44100,
                 roi_time=None, roi_freq=[0, 3000], title='right channel')


for d in ['left', 'right', 'stereo']:
    t, freq, sxx = signal.spectrogram(eval('{}_wav'.format(d)), fs=44100)
    plt.pcolormesh(freq, t, np.log(sxx))
    plt.title('{}_bad'.format(d))
    plt.show()


get_colormap(left, speed_x, speed_y, fig_type='time-frequency', n_frame=8192, pad_width=0, 
                 resolution_level=1, window='hanning', fs=44100,
                 roi_time=None, roi_freq=[0, 3000], title='left good')



for d in ['left', 'right']:
    t, freq, sxx = signal.spectrogram(eval(d), fs=44100)
    plt.pcolormesh(freq, t, np.log(sxx))
    plt.title('{}_good'.format(d))
    plt.show()

fgood = wavio.read(good_wav)
left, right = fgood.data.T

t, freq, sxx = signal.spectrogram(right, fs=44100)
plt.pcolormesh(freq, t, np.log(sxx))
plt.show()
'''

def get_rpm(file_name):
    if file_name[0] == '1':
        return 3150
    if file_name[0] == '2':
        return 3500
    if file_name[0] == '3':
        return 3850
    if file_name[0] == '4':
        return 4970

mic_path = 'data/'
ni_path = 'D:/SpeedFromAmplitudeV2.0/data/21082609/'
file_list = os.listdir(ni_path)

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


compare_file = ['85NG1-1_210826093625.tdms', '4NG2-1_210826081835.tdms', '4NG3-1_210826073957.tdms', '4OK-1_210826075343.tdms']
df = np.array(pd.read_csv('data/85ng1-1.csv')['0']) / (2**23 - 1) * 126.34 * 10**(-4.1155/20)
df = KalmanFilter(df)
df2 = np.array(pd.read_csv('data/new-65ng1-1.csv')['0'] * 20**(-4.1155/20))
df3 = np.array(pd.read_csv('ni white noise1.csv')['0'])

n_frame = 8192 * 4
n_frame_umik = 8192 * 2
step = int(0.5*n_frame)
fs = 102400
for f in compare_file:
    y = []
    y1 = []
    file = ni_path + f
    data = TdmsFile.read(file)
    vib = data['AIData']['Mic1'].data[:102400*10]
    vib = KalmanFilter(vib)
    db_vib = float_to_db(vib)
    rmsni = np.sqrt(np.mean(vib**2))
    rmsu = np.sqrt(np.mean(df**2))

    rpm = 85000#get_rpm(f)
    for i in range(len(vib)//step):
        frame = vib[i*step:i*step+n_frame]
        f1 = df[i*step:i*step+n_frame_umik]
        if len(frame) < n_frame:
            continue
        if len(f1) < n_frame_umik:
            continue
        yfft = (abs(fft(frame * signal.hann(n_frame)))[1:n_frame//2] * 2) / sum(signal.hann(n_frame))
        yf1 = abs(fft(f1 * signal.hann(n_frame_umik)))[1:n_frame_umik//2] * 2 / sum(signal.hann(n_frame_umik))
        y.append(yfft)
        y1.append(yf1)
    denoise = 100 / rpm * 60
    
    #o = fftfreq(n_frame, d=rpm/60/fs)[1:n_frame//2]
    o = fftfreq(n_frame, d=1/fs)[1:n_frame//2]
    #o1 = fftfreq(n_frame_umik, d=rpm/60/48000)[1:n_frame_umik//2]
    o1 = fftfreq(n_frame_umik, d=1/48000)[1:n_frame_umik//2]

    y_pa = np.sqrt(np.mean(np.array(y)**2, axis=0))
    y1_pa = np.sqrt(np.mean(np.array(y1)**2, axis=0))
    y_db = 20 * np.log10(y_pa / 20 / 10**-6)
    y1_db = 20 * np.log10(y1_pa / 20 / 10**-6) - 20 * np.log10(102400/48000)
    for i in range(len(o1)):
        if o1[i] * rpm / 60 < 20000:
            fq = o1[i] * rpm / 60
            idx = np.argmin(abs(fq - freq))
            #y1_db[i] += res[idx]
    plt.plot(o[(o > denoise) & (o < 20000)], y_pa[(o > denoise) & (o < 20000)], label='NI' + ' ' + str(rmsni)[:5], linewidth=0.5)
    plt.plot(o1[(o1 > denoise) & (o1 < 20000)], y1_pa[(o1 > denoise) & (o1 < 20000)], label='UMIK-1 (with frf)' + ' ' + str(rmsu)[:5], linewidth=0.5)
    plt.legend()
    plt.xlabel('Order')
    plt.ylabel('SPL (dB)')
    plt.title(f.split('_')[0])
    plt.show()
plt.legend()
plt.show()









