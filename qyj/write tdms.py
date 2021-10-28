from nptdms import TdmsWriter, ChannelObject
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time

def write_tdms(filename, group_name, channel_name, data, properties=None, mode='a'):
    channel_object = ChannelObject(group_name, channel_name, data, properties)
    with TdmsWriter(filename, mode) as tdms_writer:
        tdms_writer.write_segment([channel_object])
    
path = 'data/'
out = 'datatdms/'

file_list = os.listdir(path)
for file in file_list:
    if file[:2] in ['65', '85', '10']:
        data = np.array(pd.read_csv(path+file)['0']) / (2**23 - 1) * 126.34 * 10**(-4.1155/20)
        write_tdms(out+file.split('.')[0]+'.tdms', 'AIData', 'Mic', data)
