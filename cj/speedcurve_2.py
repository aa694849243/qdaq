import matplotlib.pyplot as plt
from auxiliary_tools import read_tdms, read_hdf5
import json

from matplotlib import rcParams
colors = [plt.cm.tab10(i/float(4)) for i in range(4)]
rcParams.update({'figure.autolayout': True})
file1 = r'D:\qdaq\temp\resolver-ttl_PM61191-jinkang-3vibs_211125085046.tdms'
file2 = r'D:\qdaq_\Data\resolver-ttl\2111\21112403\PM61191-jinkang-3vibs_211124033527.tdms'
file3 = r'D:\qdaq_\Data\resolver-ttl\2111\21112407\PM61191-jinkang-3vibs_211124070833.tdms'
file4 = r'D:\qdaq_\Data\resolver-ttl\2111\21112406\PM61191-jinkang-3vibs_211124064702.tdms'
new_x, _ = read_tdms(file1, 'speedData', 'speedLoc')
new_y, _ = read_tdms(file1, 'speedData', 'speedValue')
old_x, _ = read_tdms(file2, 'speedData', 'speedLoc')
old_y, _ = read_tdms(file2, 'speedData', 'speedValue')
oldr_x, _ = read_tdms(file3, 'speedData', 'speedLoc')
oldr_y, _ = read_tdms(file3, 'speedData', 'speedValue')
oldr2_x, _ = read_tdms(file4, 'speedData', 'speedLoc')
oldr2_y, _ = read_tdms(file4, 'speedData', 'speedValue')
plt.plot(new_x, new_y,label='new',c=colors[0])
plt.plot(old_x, old_y,label='old_ttl',c=colors[1])
plt.plot(oldr_x, oldr_y,label='old_r',c=colors[2])
plt.plot(oldr2_x,oldr2_y,label='old_r2',c=colors[3])
plt.legend()
plt.show()
