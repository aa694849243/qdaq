import pyaudio
import wave
from scipy.io import wavfile
import numpy as np
import struct as st
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq
from nptdms import TdmsWriter, ChannelObject
import wavio
import time

def write_tdms(filename, group_name, channel_name, data, properties=None, mode='a'):
    channel_object = ChannelObject(group_name, channel_name, data, properties)
    with TdmsWriter(filename, mode) as tdms_writer:
        tdms_writer.write_segment([channel_object])

def get_device():
    p = pyaudio.PyAudio()
    n_device = p.get_device_count()
    for i in range(n_device):
        info = p.get_device_info_by_index(i)
        name, hostapi = info['name'], info['hostApi']
        if name == '麦克风 (Umik-1  Gain: 18dB  )' and hostapi == 0:
            return i

def bit24_to_int(byte_data):
    n = len(byte_data) // 3
    num = []
    for i in range(n):
        data = byte_data[i*3:i*3+3]
        number = int.from_bytes(data, byteorder='little', signed=True)
        num.append(number)
    return num

def float_to_db(data):
    p_ref = 20 * 10**-6
    frame = 1024
    db = []
    data = np.array(data)
    for i in range(len(data) // frame):
        d = data[i*frame:i*frame+frame]
        p_rms = np.sqrt(np.mean(d**2))
        db.append(20*np.log10(p_rms / p_ref))
    return db


# read frf file
frf = []
with open ('7081617.txt') as f:
    frf = f.readlines()

frf_f, res = [], []
for data in frf[1:]:
    if data.strip():
        f, r = data.strip().split('\t')
        frf_f.append(float(f))
        res.append(float(r))


chunk = 1024                         # Each chunk will consist of 1024 samples
sample_format = pyaudio.paInt24      # 24 bits per sample
channels = 1                         # Number of audio channels
fs = 48000                           # Record at 48000 samples per second
time_in_seconds = 5
record = True
max_record_time = 10
t = 0

p = pyaudio.PyAudio()  # Create an interface to PortAudio

print('-----Now Recording-----')
 
#Open a Stream with the values we just defined
stream = p.open(format=sample_format,
                channels = channels,
                rate = fs,
                frames_per_buffer = chunk,
                input_device_index = get_device(),
                input = True)
 
frames = []  # Initialize array to store frames (byte data)
number_data = [] # store number data from byte
x = []
# record data and save


while record:
    if t < max_record_time:
        data = stream.read(chunk)
        #decode_data = st.unpack('={}h'.format(chunk), data)
        decode_data = bit24_to_int(data)
        number_data.extend(decode_data)
        frames.append(data)
        t += chunk / fs
    else:
        break

# Stop and close the Stream and PyAudio
stream.stop_stream()
stream.close()
#p.terminate()

# write data to wav file
wf = wave.open('test.wav', 'wb')
wf.setnchannels(channels)
sample_width = p.get_sample_size(sample_format)
wf.setsampwidth(sample_width)
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()
p.terminate()

print('-----Finished Recording-----')

# data correction
number_data = np.array(number_data) / (2**23 - 1) * 126.34 * 10**(-4.1155/20)

plt.plot(number_data, label='after correction')
plt.show()

#wavio.write('audio2.wav', np.array(n1), fs, sampwidth=3)

p1 = 20*np.log10(abs(fft(number_data))[1:len(number_data)//2] / len(number_data) * 2 / 20e-6)
freq = fftfreq(len(number_data), d=1/fs)[1:len(number_data)//2]

plt.plot(freq, p1)
plt.show()

# convert number to db
db = float_to_db(number_data)
plt.plot(np.arange(len(db)) * 1024 / 48000, db)
plt.show()