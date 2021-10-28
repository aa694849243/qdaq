import os
import time
import wave

import pyaudio
import numpy as np
import matplotlib.pyplot as plt

from qdaq.utils import write_tdms


def get_device():
    p = pyaudio.PyAudio()
    n_device = p.get_device_count()
    info_list=list()
    for i in range(n_device):
        info = p.get_device_info_by_index(i)
        info_list.append(info)
        name, hostapi = info['name'], info['hostApi']
        print(name)
        if name == 'Microphone (UMIK-2)' and hostapi == 0:
            return i



def bit_to_int(byte_data,len_vib):
    byte_count=len_vib//8
    n = len(byte_data) // byte_count
    num = []
    for i in range(n):
        data = byte_data[i * byte_count:i * byte_count + byte_count]
        number = int.from_bytes(data, byteorder='little', signed=True)
        num.append(number)
    return num



chunk = 8192*100  # Each chunk will consist of 1024 samples
frame=8192
sample_format = pyaudio.paInt32  # 24 bits per sample
# sample_format = pyaudio.paInt24  # 24 bits per sample

# sample_format = pyaudio.paInt24  # 24 bits per sample
channels = 1  # Number of audio channels
fs = 102400 # Record at 48000 samples per second
record = True  # Record flag, True or False
max_record_time =3 # Max record time
t = 0  # record time
# corr = 78.67 / (2 ** 23 - 1)  # Coefficient to get Pascal data
corr=10**(114/20)*(2e-5)*np.sqrt(2)/(0.920)  #


p = pyaudio.PyAudio()  # Create an interface to PortAudio

print('-----Now Recording-----')

# Open a Stream with the values we just defined
print(get_device())

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input_device_index=get_device(),
                input=True)

print("already open")
frames = []  # Initialize array to store frames (byte data)
number_data = []  # store number data from byte


# record data and save
while record:
    if t < max_record_time:
        data = stream.read(frame)  # data is byte data
        decode_data = bit_to_int(data,32)  # decode 24 bit data
        # decode_data=np.array(decode_data)/(2**31-1)
        number_data.extend(decode_data)  # save number data
        frames.append(data)  # save original data

        t += frame / fs
    else:
        break




plt.plot(number_data)
raw_data_filename="D:/qdaq/rawdata/xiaodianji_umik2/210917"

write_tdms(os.path.join(raw_data_filename,"65Ok1-1-umic.tdms"), 'AIData',
           "Umic2",
           number_data,
           None)



# wf = wave.open('test.wav', 'wb')
# wf.setnchannels(channels)
# sample_width = p.get_sample_size(sample_format)
# wf.setsampwidth(sample_width)
# wf.setframerate(fs)
# wf.writeframes(b''.join(frames))
# wf.close()


# Stop and close the Stream and PyAudio

plt.show()
stream.stop_stream()
stream.close()
p.terminate()
print('-----Finished Recording-----')