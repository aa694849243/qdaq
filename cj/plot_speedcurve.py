import matplotlib.pyplot as plt
from auxiliary_tools import read_tdms, read_hdf5
import json

from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
# file1 = r"D:\qdaq_new\temp\ktz999x_cj_7_byd-alltests-1mic_211118062844.tdms"
# file2 = r"D:\qdaq\temp\KTZ66X32S_7_byd-alltests-1mic_211118070248.h5"
# data1x, _ = read_tdms(file1, 'speedData', 'speedLoc')
# data1y, _ = read_tdms(file1, 'speedData', 'speedValue')
# data2x, _ = read_hdf5(file2, 'speedData', 'speedLoc')
# data2y, _ = read_hdf5(file2, 'speedData', 'speedValue')
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(data1x, data1y, label='new')
# ax1.set_xlabel('Time/s')
# ax1.set_ylabel('rpm')
# ax1.legend()
# ax2.plot(data2x, data2y, label='orgin')
# ax2.legend()
# ax2.set_xlabel('Time/s')
# plt.show()
file_old = r'D:\qdaq_\temp\KTZ66X32S-alltests-resolver_byd-alltests-1mic_211122065221.json'
file_new = r'D:\qdaq\temp\ktz999x_cj_byd-alltests-1mic_211123013618.json'
with open(file_old, 'r') as f:
    data_old = json.load(f)
with open(file_new, 'r', encoding='utf-8') as f:
    data_new = json.load(f)
data_old_s1 = data_old['resultData'][0]['dataSection'][0]['twodTD']
data_new_s1 = data_new['resultData'][0]['dataSection'][0]['twodTD']
# 时间域rms计算
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(data_new_s1[0]['xValue'], data_new_s1[0]['yValue'], label='new')
ax1.set_xlabel('Time/s')
ax1.set_ylabel(f'{data_new_s1[0]["yName"]}/{data_new_s1[0]["yUnit"]}')
ax2.plot(data_old_s1[0]['xValue'], data_old_s1[0]['yValue'], label='old')
ax2.set_xlabel('Time/s')
ax2.set_ylabel(f'{data_old_s1[0]["yName"]}/{data_old_s1[0]["yUnit"]}')
plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(data_new_s1[1]['xValue'], data_new_s1[1]['yValue'], label='new')
ax1.set_xlabel('Time/s')
ax1.set_ylabel(f'{data_new_s1[1]["yName"]}/{data_new_s1[1]["yUnit"]}')
ax2.plot(data_old_s1[1]['xValue'], data_old_s1[1]['yValue'], label='old')
ax2.set_xlabel('Time/s')
ax2.set_ylabel(f'{data_old_s1[1]["yName"]}/{data_old_s1[1]["yUnit"]}')
plt.show()
# # 按圈计算rms
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(data_new_s1[3]['xValue'], data_new_s1[3]['yValue'], label='new')
# ax1.set_xlabel('Time/s')
# ax1.set_ylabel(f'{data_new_s1[3]["yName"]}/{data_new_s1[3]["yUnit"]}')
# ax2.plot(data_old_s1[3]['xValue'], data_old_s1[3]['yValue'], label='old')
# ax2.set_xlabel('Time/s')
# ax2.set_ylabel(f'{data_old_s1[3]["yName"]}/{data_old_s1[3]["yUnit"]}')
# plt.show()
#
# data_old_s1 = data_old['resultData'][0]['dataSection'][0]['twodOC']
# data_new_s1 = data_new['resultData'][0]['dataSection'][0]['twodOC']
# # twodOc
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(data_new_s1[0]['xValue'], data_new_s1[0]['yValue'], label='new')
# ax1.set_xlabel('Time/s')
# ax1.set_ylabel(f'{data_new_s1[0]["yName"]}/{data_new_s1[0]["yUnit"]}')
# ax2.plot(data_old_s1[0]['xValue'], data_old_s1[0]['yValue'], label='old')
# ax2.set_xlabel('Time/s')
# ax2.set_ylabel(f'{data_old_s1[0]["yName"]}/{data_old_s1[0]["yUnit"]}')
# plt.show()
#
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(data_new_s1[1]['xValue'], data_new_s1[1]['yValue'], label='new')
# ax1.set_xlabel('Time/s')
# ax1.set_ylabel(f'{data_new_s1[1]["yName"]}/{data_new_s1[1]["yUnit"]}')
# ax2.plot(data_old_s1[1]['xValue'], data_old_s1[1]['yValue'], label='old')
# ax2.set_xlabel('Time/s')
# ax2.set_ylabel(f'{data_old_s1[1]["yName"]}/{data_old_s1[1]["yUnit"]}')
# plt.show()
#
# data_old_s1 = data_old['resultData'][0]['dataSection'][0]['twodOS']
# data_new_s1 = data_new['resultData'][0]['dataSection'][0]['twodOS']
# # twodOS
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(data_new_s1[0]['xValue'], data_new_s1[0]['yValue'], label='new')
# ax1.set_xlabel('Time/s')
# ax1.set_ylabel(f'{data_new_s1[0]["yName"]}/{data_new_s1[0]["yUnit"]}')
# ax2.plot(data_old_s1[0]['xValue'], data_old_s1[0]['yValue'], label='old')
# ax2.set_xlabel('Time/s')
# ax2.set_ylabel(f'{data_old_s1[0]["yName"]}/{data_old_s1[0]["yUnit"]}')
# plt.show()
#
# data_old_s1 = data_old['resultData'][0]['dataSection'][0]['twodCeps']
# data_new_s1 = data_new['resultData'][0]['dataSection'][0]['twodCeps']
# # twodCeps
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(data_new_s1[0]['xValue'], data_new_s1[0]['yValue'], label='new')
# ax1.set_xlabel('Time/s')
# ax1.set_ylabel(f'{data_new_s1[0]["yName"]}/{data_new_s1[0]["yUnit"]}')
# ax2.plot(data_old_s1[0]['xValue'], data_old_s1[0]['yValue'], label='old')
# ax2.set_xlabel('Time/s')
# ax2.set_ylabel(f'{data_old_s1[0]["yName"]}/{data_old_s1[0]["yUnit"]}')
# plt.show()
#
# # twod 4.8k sensor-1
data_old_s1 = data_old['resultData'][0]['dataSection'][1]['twodTD']
data_new_s1 = data_new['resultData'][0]['dataSection'][1]['twodTD']
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(data_new_s1[0]['xValue'], data_new_s1[0]['yValue'], label='new')
ax1.set_xlabel('Time/s')
ax1.set_ylabel(f'{data_new_s1[0]["yName"]}/{data_new_s1[0]["yUnit"]}')
ax2.plot(data_old_s1[0]['xValue'], data_old_s1[0]['yValue'], label='old')
ax2.set_xlabel('Time/s')
ax2.set_ylabel(f'{data_old_s1[0]["yName"]}/{data_old_s1[0]["yUnit"]}')
plt.show()
#
# twod 4.8k sensor-1 crest
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(data_new_s1[1]['xValue'], data_new_s1[1]['yValue'], label='new')
ax1.set_xlabel('Time/s')
ax1.set_ylabel(f'{data_new_s1[1]["yName"]}/{data_new_s1[1]["yUnit"]}')
ax2.plot(data_old_s1[1]['xValue'], data_old_s1[1]['yValue'], label='old')
ax2.set_xlabel('Time/s')
ax2.set_ylabel(f'{data_old_s1[1]["yName"]}/{data_old_s1[1]["yUnit"]}')
plt.show()

# # twod 4.8k sensor-1 按圈计算rms
# data_old_s1 = data_old['resultData'][0]['dataSection'][1]['twodTD']
# data_new_s1 = data_new['resultData'][0]['dataSection'][1]['twodTD']
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(data_new_s1[3]['xValue'], data_new_s1[3]['yValue'], label='new')
# ax1.set_xlabel('Time/s')
# ax1.set_ylabel(f'{data_new_s1[3]["yName"]}/{data_new_s1[3]["yUnit"]}')
# ax2.plot(data_old_s1[3]['xValue'], data_old_s1[3]['yValue'], label='old')
# ax2.set_xlabel('Time/s')
# ax2.set_ylabel(f'{data_old_s1[3]["yName"]}/{data_old_s1[3]["yUnit"]}')
# plt.show()
#
# # twodC 4.8k sensor-1
# data_old_s1 = data_old['resultData'][0]['dataSection'][1]['twodOC']
# data_new_s1 = data_new['resultData'][0]['dataSection'][1]['twodOC']
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(data_new_s1[0]['xValue'], data_new_s1[0]['yValue'], label='new')
# ax1.set_xlabel('Time/s')
# ax1.set_ylabel(f'{data_new_s1[0]["yName"]}/{data_new_s1[0]["yUnit"]}')
# ax2.plot(data_old_s1[0]['xValue'], data_old_s1[0]['yValue'], label='old')
# ax2.set_xlabel('Time/s')
# ax2.set_ylabel(f'{data_old_s1[0]["yName"]}/{data_old_s1[0]["yUnit"]}')
# plt.show()
#
# data_old_s1 = data_old['resultData'][0]['dataSection'][1]['twodOC']
# data_new_s1 = data_new['resultData'][0]['dataSection'][1]['twodOC']
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(data_new_s1[1]['xValue'], data_new_s1[1]['yValue'], label='new')
# ax1.set_xlabel('Time/s')
# ax1.set_ylabel(f'{data_new_s1[1]["yName"]}/{data_new_s1[1]["yUnit"]}')
# ax2.plot(data_old_s1[1]['xValue'], data_old_s1[1]['yValue'], label='old')
# ax2.set_xlabel('Time/s')
# ax2.set_ylabel(f'{data_old_s1[1]["yName"]}/{data_old_s1[1]["yUnit"]}')
# plt.show()
#
# # twodOS 4.8k sensor-1
# data_old_s1 = data_old['resultData'][0]['dataSection'][1]['twodOS']
# data_new_s1 = data_new['resultData'][0]['dataSection'][1]['twodOS']
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(data_new_s1[0]['xValue'], data_new_s1[0]['yValue'], label='new')
# ax1.set_xlabel('Time/s')
# ax1.set_ylabel(f'{data_new_s1[0]["yName"]}/{data_new_s1[0]["yUnit"]}')
# ax2.plot(data_old_s1[0]['xValue'], data_old_s1[0]['yValue'], label='old')
# ax2.set_xlabel('Time/s')
# ax2.set_ylabel(f'{data_old_s1[0]["yName"]}/{data_old_s1[0]["yUnit"]}')
# plt.show()
#
# # twodOS 4.8k sensor-1
# data_old_s1 = data_old['resultData'][0]['dataSection'][1]['twodCeps']
# data_new_s1 = data_new['resultData'][0]['dataSection'][1]['twodCeps']
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(data_new_s1[0]['xValue'], data_new_s1[0]['yValue'], label='new')
# ax1.set_xlabel('Time/s')
# ax1.set_ylabel(f'{data_new_s1[0]["yName"]}/{data_new_s1[0]["yUnit"]}')
# ax2.plot(data_old_s1[0]['xValue'], data_old_s1[0]['yValue'], label='old')
# ax2.set_xlabel('Time/s')
# ax2.set_ylabel(f'{data_old_s1[0]["yName"]}/{data_old_s1[0]["yUnit"]}')
# plt.show()
# 时间域rms计算 sensor-2
data_old_s1 = data_old['resultData'][1]['dataSection'][0]['twodTD']
data_new_s1 = data_new['resultData'][1]['dataSection'][0]['twodTD']
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(data_new_s1[0]['xValue'], data_new_s1[0]['yValue'], label='new')
ax1.set_xlabel('Time/s')
ax1.set_ylabel(f'{data_new_s1[0]["yName"]}/{data_new_s1[0]["yUnit"]}')
ax2.plot(data_old_s1[0]['xValue'], data_old_s1[0]['yValue'], label='old')
ax2.set_xlabel('Time/s')
ax2.set_ylabel(f'{data_old_s1[0]["yName"]}/{data_old_s1[0]["yUnit"]}')
plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(data_new_s1[1]['xValue'], data_new_s1[1]['yValue'], label='new')
ax1.set_xlabel('Time/s')
ax1.set_ylabel(f'{data_new_s1[1]["yName"]}/{data_new_s1[1]["yUnit"]}')
ax2.plot(data_old_s1[1]['xValue'], data_old_s1[1]['yValue'], label='old')
ax2.set_xlabel('Time/s')
ax2.set_ylabel(f'{data_old_s1[1]["yName"]}/{data_old_s1[1]["yUnit"]}')
plt.show()
#
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(data_new_s1[3]['xValue'], data_new_s1[3]['yValue'], label='new')
# ax1.set_xlabel('Time/s')
# ax1.set_ylabel(f'{data_new_s1[3]["yName"]}/{data_new_s1[3]["yUnit"]}')
# ax2.plot(data_old_s1[3]['xValue'], data_old_s1[3]['yValue'], label='old')
# ax2.set_xlabel('Time/s')
# ax2.set_ylabel(f'{data_old_s1[3]["yName"]}/{data_old_s1[3]["yUnit"]}')
# plt.show()
#
# # TDOC计算 sensor-2
# data_old_s1 = data_old['resultData'][1]['dataSection'][0]['twodOC']
# data_new_s1 = data_new['resultData'][1]['dataSection'][0]['twodOC']
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(data_new_s1[0]['xValue'], data_new_s1[0]['yValue'], label='new')
# ax1.set_xlabel('Time/s')
# ax1.set_ylabel(f'{data_new_s1[0]["yName"]}/{data_new_s1[0]["yUnit"]}')
# ax2.plot(data_old_s1[0]['xValue'], data_old_s1[0]['yValue'], label='old')
# ax2.set_xlabel('Time/s')
# ax2.set_ylabel(f'{data_old_s1[0]["yName"]}/{data_old_s1[0]["yUnit"]}')
# plt.show()

#
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(data_new_s1[1]['xValue'], data_new_s1[1]['yValue'], label='new')
# ax1.set_xlabel('Time/s')
# ax1.set_ylabel(f'{data_new_s1[1]["yName"]}/{data_new_s1[1]["yUnit"]}')
# ax2.plot(data_old_s1[1]['xValue'], data_old_s1[1]['yValue'], label='old')
# ax2.set_xlabel('Time/s')
# ax2.set_ylabel(f'{data_old_s1[1]["yName"]}/{data_old_s1[1]["yUnit"]}')
# plt.show()
#
# # tdOS计算 sensor-2
# data_old_s1 = data_old['resultData'][1]['dataSection'][0]['twodOS']
# data_new_s1 = data_new['resultData'][1]['dataSection'][0]['twodOS']
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(data_new_s1[0]['xValue'], data_new_s1[0]['yValue'], label='new')
# ax1.set_xlabel('Time/s')
# ax1.set_ylabel(f'{data_new_s1[0]["yName"]}/{data_new_s1[0]["yUnit"]}')
# ax2.plot(data_old_s1[0]['xValue'], data_old_s1[0]['yValue'], label='old')
# ax2.set_xlabel('Time/s')
# ax2.set_ylabel(f'{data_old_s1[0]["yName"]}/{data_old_s1[0]["yUnit"]}')
# plt.show()
#
# # tdCeps计算 sensor-2
# data_old_s1 = data_old['resultData'][1]['dataSection'][0]['twodCeps']
# data_new_s1 = data_new['resultData'][1]['dataSection'][0]['twodCeps']
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(data_new_s1[0]['xValue'], data_new_s1[0]['yValue'], label='new')
# ax1.set_xlabel('Time/s')
# ax1.set_ylabel(f'{data_new_s1[0]["yName"]}/{data_new_s1[0]["yUnit"]}')
# ax2.plot(data_old_s1[0]['xValue'], data_old_s1[0]['yValue'], label='old')
# ax2.set_xlabel('Time/s')
# ax2.set_ylabel(f'{data_old_s1[0]["yName"]}/{data_old_s1[0]["yUnit"]}')
# plt.show()
# 时间域rms计算 sensor-2
data_old_s1 = data_old['resultData'][1]['dataSection'][1]['twodTD']
data_new_s1 = data_new['resultData'][1]['dataSection'][1]['twodTD']
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(data_new_s1[0]['xValue'], data_new_s1[0]['yValue'], label='new')
ax1.set_xlabel('Time/s')
ax1.set_ylabel(f'{data_new_s1[0]["yName"]}/{data_new_s1[0]["yUnit"]}')
ax2.plot(data_old_s1[0]['xValue'], data_old_s1[0]['yValue'], label='old')
ax2.set_xlabel('Time/s')
ax2.set_ylabel(f'{data_old_s1[0]["yName"]}/{data_old_s1[0]["yUnit"]}')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(data_new_s1[1]['xValue'], data_new_s1[1]['yValue'], label='new')
ax1.set_xlabel('Time/s')
ax1.set_ylabel(f'{data_new_s1[1]["yName"]}/{data_new_s1[1]["yUnit"]}')
ax2.plot(data_old_s1[1]['xValue'], data_old_s1[1]['yValue'], label='old')
ax2.set_xlabel('Time/s')
ax2.set_ylabel(f'{data_old_s1[1]["yName"]}/{data_old_s1[1]["yUnit"]}')
plt.show()

# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(data_new_s1[3]['xValue'], data_new_s1[3]['yValue'], label='new')
# ax1.set_xlabel('Time/s')
# ax1.set_ylabel(f'{data_new_s1[3]["yName"]}/{data_new_s1[3]["yUnit"]}')
# ax2.plot(data_old_s1[3]['xValue'], data_old_s1[3]['yValue'], label='old')
# ax2.set_xlabel('Time/s')
# ax2.set_ylabel(f'{data_old_s1[3]["yName"]}/{data_old_s1[3]["yUnit"]}')
# plt.show()
#
# # TDOC计算 sensor-2
# data_old_s1 = data_old['resultData'][1]['dataSection'][1]['twodOC']
# data_new_s1 = data_new['resultData'][1]['dataSection'][1]['twodOC']
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(data_new_s1[0]['xValue'], data_new_s1[0]['yValue'], label='new')
# ax1.set_xlabel('Time/s')
# ax1.set_ylabel(f'{data_new_s1[0]["yName"]}/{data_new_s1[0]["yUnit"]}')
# ax2.plot(data_old_s1[0]['xValue'], data_old_s1[0]['yValue'], label='old')
# ax2.set_xlabel('Time/s')
# ax2.set_ylabel(f'{data_old_s1[0]["yName"]}/{data_old_s1[0]["yUnit"]}')
# plt.show()
#
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(data_new_s1[1]['xValue'], data_new_s1[1]['yValue'], label='new')
# ax1.set_xlabel('Time/s')
# ax1.set_ylabel(f'{data_new_s1[1]["yName"]}/{data_new_s1[1]["yUnit"]}')
# ax2.plot(data_old_s1[1]['xValue'], data_old_s1[1]['yValue'], label='old')
# ax2.set_xlabel('Time/s')
# ax2.set_ylabel(f'{data_old_s1[1]["yName"]}/{data_old_s1[1]["yUnit"]}')
# plt.show()

# # tdOS计算 sensor-2
# data_old_s1 = data_old['resultData'][1]['dataSection'][1]['twodOS']
# data_new_s1 = data_new['resultData'][1]['dataSection'][1]['twodOS']
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(data_new_s1[0]['xValue'], data_new_s1[0]['yValue'], label='new')
# ax1.set_xlabel('Time/s')
# ax1.set_ylabel(f'{data_new_s1[0]["yName"]}/{data_new_s1[0]["yUnit"]}')
# ax2.plot(data_old_s1[0]['xValue'], data_old_s1[0]['yValue'], label='old')
# ax2.set_xlabel('Time/s')
# ax2.set_ylabel(f'{data_old_s1[0]["yName"]}/{data_old_s1[0]["yUnit"]}')
# plt.show()
#
# # tdCeps计算 sensor-2
# data_old_s1 = data_old['resultData'][1]['dataSection'][1]['twodCeps']
# data_new_s1 = data_new['resultData'][1]['dataSection'][1]['twodCeps']
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(data_new_s1[0]['xValue'], data_new_s1[0]['yValue'], label='new')
# ax1.set_xlabel('Time/s')
# ax1.set_ylabel(f'{data_new_s1[0]["yName"]}/{data_new_s1[0]["yUnit"]}')
# ax2.plot(data_old_s1[0]['xValue'], data_old_s1[0]['yValue'], label='old')
# ax2.set_xlabel('Time/s')
# ax2.set_ylabel(f'{data_old_s1[0]["yName"]}/{data_old_s1[0]["yUnit"]}')
# plt.show()
