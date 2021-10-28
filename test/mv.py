import os

import matplotlib.pyplot as plt
import numpy as np
from nptdms import TdmsFile

def mv(data,n):
    count=len(data)-n
    a=list(np.zeros(n-1))
    for i in range(count):
        a.append(np.mean(data[i:i+n]))

    return a

path = 'D:/qdaq/rawdata/test01'
file="test01_3mm_8k_210802064140.tdms"
data = TdmsFile.read(os.path.join(path, file))
vib=data["AIData"]["Mic"]
vib=vib[:10*102400]
plt.figure("vib")
plt.plot(vib)

mv_d=dict()

n_list=[2000,5000,10000]
n_list=[10000]
for n in n_list:
    print(n)
    mv_d[n]=mv(vib,n)
    plt.figure(n)
    plt.plot(mv_d[n])
    print(n)
plt.show()


