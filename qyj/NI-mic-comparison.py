from tools import *
import pandas as pd
import os
import wavio
import requests

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


# compare ni and umik data

umik_path = 'data/'
ni_path = 'D:/SpeedFromAmplitudeV2.0/data/21082608/'
ni_file = os.listdir(ni_path)
umik_file = os.listdir(umik_path)
coef =  126.34 * 10**(-4.1155/20) / (2**23 - 1)

for f in ni_file:
    f_name = f.split('_')[0]
    ni_data = TdmsFile.read(ni_path+f)['AIData']['Mic1'].data
    umik_data = pd.read_csv(umik_path+f_name.lower()+'.csv')['0'] * coef

    get_colormap(ni_data, [0], [0], fig_type='time-frequency', n_frame=8192*2, pad_width=0, 
                 resolution_level=1, window='hanning', fs=102400,
                 roi_time=None, roi_freq=[200, 2000], title='NI'+' '+f_name)
    plt.show()
    
    get_colormap(umik_data, [0], [0], fig_type='time-frequency', n_frame=8192, pad_width=0, 
             resolution_level=1, window='hanning', fs=48000,
             roi_time=None, roi_freq=[200, 2000], title='UMIK-1'+' '+f_name)
    plt.show()





