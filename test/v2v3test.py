# 用v2保存trigger,给v3计算，保证两者数据相同
import json
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from qdaq.indicator_tools import order_spectrum_for_share, order_spectrum, angular_resampling_old
from qdaq.initial import create_empty_threedos_for_share, create_empty_twodoc_for_share, \
    create_empty_threedos, create_empty_twodoc
from qdaq.parameters import Parameters
from nptdms import TdmsFile
from indicator_tools import twod_time_domain, oned_time_domain, \
    convert_time_speed_npinterp, twod_stat_factor, \
    convert_time_speed, average_oned_stat_factor, angular_resampling_for_share, \
    twod_time_domain_for_share, twod_stat_factor_for_share, \
    order_spectrum_for_share, convert_time_speed_for_share, \
    angular_resampling_for_share_scipy
from indicator_tools import angular_resampling, order_spectrum, \
    twod_order_spectrum, oned_order_spectrum, cepstrum
import matplotlib.pyplot as plt


# left_index_for_put:266082
# right_index_for_put:1887798
def readAllRawData(filename, channelName):
    allrawdata = dict()

    with TdmsFile.open(filename) as tdms_file:
        for channelName in channelName:
            allrawdata[channelName] = list(tdms_file['AIData'][channelName][:])

    return allrawdata


def read_hdf5(filename, groupname, channelnames, start=None, end=None):
    data = dict()
    with h5py.File(filename) as f:
        for channel in channelnames:
            data[channel] = np.array(f[groupname][channel][start:end])

    return data


if __name__ == '__main__':
    trigger_filename2 = "D:\\qdaq\\对比\\qdaqv2.0和v3.0对比\\trigger_v2.h5"
    trigger_right_index = 126596
    trigger_start_index = 7764
    trigger_filename = "D:\\qdaq\\对比\\qdaqv2.0和v3.0对比\\trigger_v3_ppr.h5"
    trigger_right_index = 126588
    trigger_start_index = 7796
    # trigger_right_index=126596
    # trigger_start_index=7814

    trigger_ndarray = read_hdf5(trigger_filename, "triggerV2", ["trigger"])['trigger'].astype('int64')
    trigger_ndarray2 = read_hdf5(trigger_filename2, "triggerV2", ["trigger"])['trigger'].astype('int64')[
                       :297569]

    # rawdata_filename="D:\qdaq\Simu\\PM61191-1vib.tdms"
    # vib_ndarray=readAllRawData(rawdata_filename,['Vibration'])['Vibration']
    vib_ndarray = \
        read_hdf5("D:\qdaq\Data\PM61191-1vib-1test\\2107\\21071909\\PM61191-1vib_210719091718.h5",
                  "AIData",
                  ["Vibration"])["Vibration"]

    # trigger_before=np.arange(405-35,0,-35)[::-1]
    # trigger_ndarray=np.concatenate([trigger_before,trigger_ndarray])
    # trigger_start_index+=12
    # trigger_right_index+=12

    plt.plot(trigger_ndarray, c='r', label="v3")
    plt.plot(trigger_ndarray2, c='b', label="v2")
    plt.legend()
    plt.show()

    # trigger_ndarray=trigger_ndarray2-8
    trigger_ndarray = trigger_ndarray
    # a=np.random.randint(4,len(trigger_ndarray))
    # a=np.where(a<4,0,1)
    # trigger_ndarray+=a

    vib_ndarray = np.array(vib_ndarray).astype('float32')
    # v2:trigger[7764]==266082 v2:trigger[126596]==1887798
    # v2:trigger[7808]==267579 v2:trigger[126596]==1887798
    # v2 np.searchsorted(trigger_ndarray,1887795)=126596
    # v2 np.searchsorted(trigger_ndarray,267751)=7814

    # v3 trigger[7796]=267571  v3:trigger[126588]==1887795

    param = Parameters("D:\\qdaq\\Config\\PM61191-2vibs-ppr_paramReceived.json")

    trigger_right_index_array = []

    full_rev = (np.arange(trigger_right_index - trigger_start_index)) / param.speedCalcInfo['ppr']
    full_rev_time = trigger_ndarray[trigger_start_index:trigger_right_index] / 102400 - trigger_ndarray[
        trigger_start_index] / 102400
    vib = vib_ndarray[trigger_ndarray[trigger_start_index]:trigger_ndarray[trigger_right_index - 1]]

    counter_ar = 0
    vib_rsp_index = 0
    vib_rsp = np.zeros(1000000)
    vib_rsp_time = np.zeros(1000000)
    while ((counter_ar + 1) * 8192 < len(vib)):
        vib_rsp_frame, vib_rsp_time_frame = angular_resampling_old(full_rev, full_rev_time,
                                                                   vib,
                                                                   np.arange(0, len(vib)) /
                                                                   102400,
                                                                   counter_ar,
                                                                   8192,
                                                                   41,
                                                                   param.speedRecogInfo[
                                                                       'minSpeed'][
                                                                       0] / 60 /
                                                                   102400,
                                                                   param.orderSpectrumCalcInfo[
                                                                       'dr_af']
                                                                   )
        counter_ar += 1
        vib_rsp[vib_rsp_index:vib_rsp_index + len(vib_rsp_frame)] = vib_rsp_frame
        vib_rsp_time[vib_rsp_index:vib_rsp_index + len(vib_rsp_frame)] = np.array(
            vib_rsp_time_frame) + trigger_ndarray[trigger_start_index] / 102400
        vib_rsp_index = vib_rsp_index + len(vib_rsp_frame)

    sensor_index = 0
    counter_or = 0
    maxT = param.speedRecogInfo["maxT"][0] * 1.5
    # maxSpeed是测试段内的最大速度
    maxSpeed = max(param.speedRecogInfo["startSpeed"][0], param.speedRecogInfo["endSpeed"][0])
    osMaxLen = int(maxT * maxSpeed / 60 // (param.orderSpectrumCalcInfo['revNum'] * (
            1 - param.orderSpectrumCalcInfo['overlapRatio']))) + 1
    threed_os = create_empty_threedos_for_share(param.orderSpectrumCalcInfo, param.taskInfo,
                                                sensor_index, osMaxLen,
                                                db_flag=param.basicInfo['dBFlag'])
    twod_oc = create_empty_twodoc_for_share(param.orderCutCalcInfo, param.taskInfo, sensor_index,
                                            osMaxLen, db_flag=param.basicInfo['dBFlag'])

    # order domain indicators
    group_name = 'sensor0test0'
    filename = 'v3-liner-1arpoints'
    counter_or = 0
    threed_os, twod_oc, counter_or = order_spectrum_for_share(threed_os,
                                                              twod_oc,
                                                              counter_or,
                                                              vib_rsp[:vib_rsp_index],
                                                              vib_rsp_time[:vib_rsp_index],
                                                              param.orderSpectrumCalcInfo,
                                                              param.orderCutCalcInfo,
                                                              sensor_index,
                                                              db_flag=
                                                              param.basicInfo[
                                                                  'dBFlag'])

    threed_os['xValue'] = threed_os['xValue'][:counter_or]
    threed_os['zValue'] = threed_os['zValue'][:counter_or]

    for result in twod_oc:
        result['xValue'] = result['xValue'][:counter_or]
        result['yValue'] = result['yValue'][:counter_or]

    twod_os = twod_order_spectrum(threed_os, param.taskInfo,sensor_index)

    plt.plot(twod_os[0]['xValue'], twod_os[0]['yValue'], c='b', label="newmethodoldar")

    filename3 = "D:\\qdaq\\对比\\qdaqv2v3-cubic\\PM61191-2vibs-resolver2-6tests-v2.2.5-1.json"
    with open(filename3, encoding="utf-8") as f:
        data3 = json.load(f)
    orderSpectrum3 = data3["resultData"][0]["dataSection"][0]["twodOS"][0]
    plt.plot(orderSpectrum3["xValue"], orderSpectrum3["yValue"], c="r", label="v2")
    plt.legend()
    plt.show()
    print("over")
    pass
