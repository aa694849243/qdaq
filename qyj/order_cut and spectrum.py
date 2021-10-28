from scipy import stats
import matplotlib.pyplot as plt
from speedFromVib import *
from tools import *
import timeit
import time

def order_spectrum(vib, speedt, speed, roit, max_order, fs):
    '''
    calculate order spectrum from vibration data
    ********************************************************
    parameters
           vib: vibration data
        speedt: speed time
         speed: speed in rpm
          roit: time interval to calculate
     max_order: desired max order for plot
            fs: sampling rate
    
    return
             o: order, x-axis for order spectrum
             y: amplitude, y-axis for order spectrum
    '''
    speedt = np.array(speedt)
    speed = np.array(speed)

    roi = (speedt >= min(roit)) & (speedt <= max(roit))                                     # determine region of interest
    troi = speedt[roi]
    rpmroi = speed[roi]

    res = stats.linregress(troi, rpmroi)                                                    # preprocess for speed
    k, b = res.slope, res.intercept
    rpm = k * troi + b                                                                      # speed after process

    nr = 1 / (2 * max_order) * 60                                                           # angle for angular resample
    cnt = get_revolution_by_speed(troi[0], troi[-1], rpmroi[0], rpmroi[-1], nr)             # angular resample

    loc = (np.array(cnt) * fs).astype(np.int64)
    revib = vib[loc]                                                                        # signal after angular resample

    y = []
    n = int(30 * 60 / nr)                                                                   # nfft
    o = fftfreq(n, d=nr/60)[1:n // 2]                                                     # order for result
    oroi = o < (max_order + 1)                                                              # adjust order by max_order
    step = int(n / 30)                                                  

    for i in range(len(revib)//step):                                                       # doing fft, get result
        frame = revib[i*step:i*step+n]
        if len(frame) < n:
            continue
        fy = abs(fft(frame * np.hanning(len(frame))))[1:len(frame)//2] / np.sqrt(len(frame))
        y.append(fy[oroi])
    return o[oroi], np.mean(y, axis=0)

def order_cut(vib, speedt, speed, roit, order_list, fs, count=0):
    '''
    calculate order cut without speed signal
    **************************************************************
    parameters
           vib: vibration data
        speedt: speed time
         speed: speed in rpm
          roit: time list for calculation
    order_list: orders to calculate
            fs: sampling rate
         count: counter for calculation by stream
    '''
    speedt = np.array(speedt)
    speed = np.array(speed)

    roi = (speedt >= min(roit)) & (speedt <= max(roit))                                     # determine region of interest
    troi = speedt[roi]
    rpmroi = speed[roi]

    res = stats.linregress(troi, rpmroi)                                                    # preprocess for speed
    k, b = res.slope, res.intercept
    rpm = k * troi + b                                                                      # speed after process

    rms_list = []
    speed_list = []
    n_frame = int(8192*2.5)                                                                 # points in one frame
    step = int(fs * 0.2)                                                                    # points between two frames

    vib = vib[count*step:]
    if len(vib) < n_frame:
        return [], [], count
    
    for i in range(len(vib) // step):
        times = step * i / fs + count * step / fs
        if times > max(troi) or times < min(troi):                                 
            continue
        rpm = k * times + b                                                                 # speed for current time
        speed_list.append(rpm)
        
        frame = vib[i*step:i*step+n_frame]
        pad_width = 8192 * 2
        frame = np.pad(frame, pad_width=pad_width, mode='constant', constant_values=0)
        window = np.hanning(len(frame))
        vib_fft = abs(fft(frame * window))[1:len(frame)//2] / (len(frame)/2) / 1.414
        freq = fftfreq(len(frame), d=1/fs)[1:len(frame)//2]

        for o in order_list:                                                                # calculate order cut
            freq_left = (o - 0.4) * rpm / 60                                                # choose cut width = 0.4 order
            freq_right = (o + 0.4) * rpm / 60
            vib_obj = vib_fft[(freq <= freq_right) & (freq >= freq_left)]
            rms = np.sqrt(np.sum(vib_obj ** 2))
            rms_list.append(rms)

    rms_list = np.array(rms_list).reshape(-1, len(order_list))                              # each column is an order cut
    count += rms_list.shape[0]
    return speed_list, rms_list, count

def order_spectrum_stream(vib, speedt, speed, roit, max_order, fs, count=0):
    '''
    calculate order spectrum from vibration data
    ********************************************************
    parameters
           vib: vibration data
        speedt: speed time
         speed: speed in rpm
          roit: time interval to calculate
     max_order: desired max order for plot
            fs: sampling rate
    
    return
             o: order, x-axis for order spectrum
             y: amplitude, y-axis for order spectrum
    '''
    speedt = np.array(speedt)
    speed = np.array(speed)

    roi = (speedt >= min(roit)) & (speedt <= max(roit))                                     # determine region of interest
    troi = speedt[roi]
    rpmroi = speed[roi]

    res = stats.linregress(troi, rpmroi)                                                    # preprocess for speed
    k, b = res.slope, res.intercept
    rpm = k * troi + b                                                                      # speed after process

    nr = 1 / (2 * max_order) * 60                                                           # angle for angular resample
    cnt = get_revolution_by_speed(troi[0], troi[-1], rpm[0], rpm[-1], nr)                   # angular resample

    loc = (np.array(cnt) * fs).astype(np.int64)
    revib = vib[loc]                                                                        # signal after angular resample
    y = []
    n = int(30 * 60 / nr)                                                                   # nfft
    o = fftfreq(n, d=nr/60)[1:n // 2]                                                       # order for result
    oroi = o < (max_order + 1)                                                              # adjust order by max_order
    step = int(n / 30)                                                  

    revib = revib[count*step:]
    if len(revib) < n:
        return o[oroi], [], count
    
    for i in range(len(revib)//step):                                                       # doing fft, get result
        frame = revib[i*step:i*step+n]
        if len(frame) < n:
            continue
        fy = abs(fft(frame * np.hanning(len(frame))))[1:len(frame)//2] / np.sqrt(len(frame))
        y.append(fy[oroi])
        count += 1
    return o[oroi], y, count

def resample_stream(data, count, n_read, fs, speedx, speedy, max_order, redata):
    '''
    angular resampling for data stream
    *************************************************
    parameters
          data: data for angular resampling
         count: count number for determining start point
        n_read: points per read
            fs: sampling rate
        speedx: time for speed curve
        speedy: speed for speed curve
     max_order: max order for order spectrum
        redata: data after resample
    
    return
        redata: data after resample
         count: count number for determining start point
    '''
    start = count * n_read / fs
    end = (count+1) * n_read / fs
    idx = (speedx >= start) * (speedx <= end)
    speed = speedy[idx]
    t = speedx[idx]
    res = stats.linregress(t, speed)
    k, b = res.slope, res.intercept
    rpm = k * t + b
    nr = 1 / (2 * max_order) * 60
    cnt = get_revolution_by_speed(count*n_read/fs, (count+1)*n_read/fs, rpm[0], rpm[-1], nr)
    cnt = (np.array(cnt) * fs).astype(np.int64)
    revib = data[cnt]
    redata.extend(revib)
    count += 1

    return redata, count

def order_spectrum_stream(data, redata, n_read, fs, speedx, speedy, max_order, n_revolution, count, count_fft):
    '''
    get order sepctrum for data stream
    **********************************************
    parameters
                data: origin vibration data
              redata: data after angular sampling
              n_read: points per read
                  fs: sampling rate
              speedx: time for speed curve
              speedy: speed for speed curve
           max_order: max order for order spectrum
        n_revolution: revolution number to do fft
               count: angular resampling count number
           count_fft: order spectrum count number
    
    return
    order[order_roi]: x-axis of order spectrum
            spectrum: y-aixs of order spectrum
               count: angular resampling count number
           count_fft: order spectrum count number
              redata: data after angular resampling 
    '''
    redata, spectrum = [], []
    nr = 1 / (2 * max_order) * 60
    nfft = int(n_revolution * 60 / nr)
    order = fftfreq(nfft, d=nr/60)[1:nfft//2]
    order_roi = order < (max_order + 1)
    step = int(nfft / 30)
    
    resample, count = resample_stream(data, count, n_read, fs, speedx, speedy, max_order, redata)
    redata.extend(resample)
    
    pstart = int(count_fft*step)
    if len(redata[pstart:]) < nfft:
        return order[order_roi], [], count, count_fft, redata
    else:
        revib = redata[pstart:]
        for i in range(len(revib)//nfft):
            frame = revib[i*step:i*step+nfft]
            spec = abs(fft(frame * np.hanning(len(frame))))[1:len(frame)//2] / np.sqrt(len(frame))
            spectrum.append(spec[order_roi])
            count_fft += 1
        return order[order_roi], spectrum, count, count_fft, redata


#----------------------- test ------------------------#
'''
########## read vibration data ##############
input_path = 'D:/SpeedFromAmplitudeV2.0/data/order_cut/017700944M600034_200628061633.tdms'
vib, speed = read_tdms(input_path)

########## speed from vibration data #############
refx, refy = get_speed_from_vib(vib, fs=102400, 'type2')
########## test for order_cut calculation efficency ##############
order = [4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 96]                                 # order for calculation
n = [4, 5, 8, 10]                                                                  # number of order to calculate
for i in n:
    o = order[:i]
    print('order number: %d' % i)
    print(timeit.timeit("order_cut(vib, refx, refy, [0.3, 24], [24, 40, 48, 72], fs=102400)", globals=globals(), number=100))

########## test for order_spectrum calculation efficency ###########
for x in [50, 100, 150, 200]:
    print('max order:%d' % x)
    print(timeit.timeit("order_spectrum(vib, refx, refy, [0.3, 24], x, fs=102400)", globals=globals(), number=100))
'''

############ test for calculation by stream ##############

# ************************* test for order cut function ************************** #
'''
# read data
input_path = 'D:/SpeedFromAmplitudeV2.0/data/order_cut/017700944M600034_200628061633.tdms'
vib, speed = read_tdms(input_path)

# calculate speed
refx, refy = get_speed_from_vib(vib, fs=102400, machine_type='type2')
refx = np.array(refx)
refy = np.array(refy)

refy = refy[(refx >= 0.3) & (refx <= 24)]
refx = refx[(refx >= 0.3) & (refx <= 24)]

# points per read
n_read = 50000

# generate simu data
simu_data = []
for i in range(len(vib) // n_read):
    simu_data.append(vib[:n_read*(i+1)])

# initialize
count = 0
speedx, rms = [], []

# calculation by stream
for data in simu_data:
    t, r, count = order_cut(data, refx, refy, [0.3, 24], [48], 102400, count)
    speedx.extend(t)
    rms.extend(r)
   
x, y, count = order_cut(vib, refx, refy, [0.3, 24], [48], fs=102400)

# comparison 
plt.plot(x, y, label='read whole file')
plt.plot(speedx, rms, label='stream calculation')
plt.legend()
plt.show()
'''


# ************************* test for order spectrum function ************************** #

# read data
input_path = 'D:/SpeedFromAmplitudeV2.0/data/order_cut/017700944M600034_200628061633.tdms'
vib, speed = read_tdms(input_path)

# calculate speed
refx, refy = get_speed_from_vib(vib, fs=102400, machine_type='type2')
refx = np.array(refx)
refy = np.array(refy)

n_read = 10000
fs = 102400

# generate simu data
simu_data = []
for i in range(len(vib) // n_read):
    simu_data.append(vib[:n_read*(i+1)])

# set initial parameters
count, count_fft = 0, 0
max_order = 150
n_revolution = 30
spec, redata = [], []

# get order spectrum
for data in simu_data:
    order, spectrum, count, count_fft, redata = order_spectrum_stream(data, redata, n_read, fs, refx, refy, max_order, n_revolution, count, count_fft)
    spec.extend(spectrum)

o, r = order_spectrum(vib, refx, refy, [0.3, 24], 150, fs=102400)
plt.plot(order, np.mean(spec, axis=0), label='stream data')
plt.plot(o, r, label='read all file')
plt.legend()
plt.show()
