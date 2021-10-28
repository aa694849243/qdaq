import pyaudio
import numpy as np

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

chunk = 1024                         # Each chunk will consist of 1024 samples
sample_format = pyaudio.paInt24      # 24 bits per sample
channels = 1                         # Number of audio channels
fs = 48000                           # Record at 48000 samples per second
record = True                        # Record flag, True or False
max_record_time = 10                 # Max record time
t = 0                                # record time
corr = 78.67 / (2**23 - 1)           # Coefficient to get Pascal data

p = pyaudio.PyAudio()                # Create an interface to PortAudio

print('-----Now Recording-----')
 
#Open a Stream with the values we just defined
stream = p.open(format=sample_format,
                channels = channels,
                rate = fs,
                frames_per_buffer = chunk,
                input_device_index = get_device(),
                input = True)
 
frames = []                          # Initialize array to store frames (byte data)
number_data = []                     # store number data from byte

# record data and save
while record:
    if t < max_record_time:
        data = stream.read(chunk)               # data is byte data
        decode_data = bit24_to_int(data)        # decode 24 bit data
        number_data.extend(decode_data)         # save number data
        frames.append(data)                     # save original data 
        t += chunk / fs
    else:
        break

# Stop and close the Stream and PyAudio
stream.stop_stream()
stream.close()
p.terminate()
print('-----Finished Recording-----')