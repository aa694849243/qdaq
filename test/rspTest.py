import os
import h5py
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from qdaq.indicator_tools import order_spectrum_for_share, order_spectrum
from qdaq.initial import create_empty_threedos_for_share, create_empty_twodoc_for_share, \
    create_empty_threedos, create_empty_twodoc
from qdaq.parameters import Parameters

def read_hdf5(filename,groupname,channelnames,start=None,end=None):
    data=dict()
    with h5py.File(filename) as f:
        for channel in channelnames:
            data[channel]=np.array(f[groupname][channel][start:end])

    return data

if __name__ == '__main__':
    file1="D:\qdaq\对比\qdaqv2.0和v3.0对比\\v3-liner-1arpoints.h5"
    file2="D:\qdaq\对比\qdaqv2.0和v3.0对比\\v2-cubic-1arpoints.h5"
    t=os.path.splitext(file1)
    file_list=[file1,file2]
    data=dict()
    # filename_list=[]
    # for file in file_list:
    #     filename=os.path.splitext(file)[0].split('\\')[-1]
    #     data[filename]=dict()
    #     file_list.append(filename)

    sensor_list=range(1)
    test_list=range(1)
    channel_list=['rsp','rsp_time']

    for sensor_id in sensor_list:
        for test_id in test_list:
            group_name="sensor{}test{}".format(sensor_id,test_id)
            plt.figure(group_name)
            colors = list(mcolors.TABLEAU_COLORS.keys())  # 颜色变化
            data[group_name]=dict()
            for i,file in enumerate(file_list):
                filename = os.path.splitext(file)[0].split('\\')[-1]
                data[group_name][filename]=read_hdf5(file,group_name,channel_list)
                plt.plot(data[group_name][filename]['rsp_time'],data[group_name][filename]['rsp'],c=mcolors.TABLEAU_COLORS[colors[1-i]],label=filename)

    # plt.legend()
    # plt.show()

    param=Parameters("D:\\qdaq\\Config\\PM61191-2vibs_paramReceived.json")

    sensor_index=0
    counter_or=0
    maxT = param.speedRecogInfo["maxT"][0] * 1.5
    # maxSpeed是测试段内的最大速度
    maxSpeed = max(
        param.speedRecogInfo["startSpeed"][0],
        param.speedRecogInfo["endSpeed"][0])
    osMaxLen = int(maxT * maxSpeed / 60 // (
            param.orderSpectrumCalcInfo['revNum'] * (
            1 - param.orderSpectrumCalcInfo[
        'overlapRatio']))) + 1
    threed_os = create_empty_threedos_for_share(
        param.orderSpectrumCalcInfo,
        param.taskInfo, sensor_index, osMaxLen,
        db_flag=param.basicInfo[
            'dBFlag'])
    twod_oc = create_empty_twodoc_for_share(param.orderCutCalcInfo,
                                            param.taskInfo,
                                            sensor_index, osMaxLen,
                                            db_flag=param.basicInfo[
                                                'dBFlag'])

    # order domain indicators
    group_name='sensor0test0'
    filename='v3-liner-1arpoints'

    threed_os, twod_oc, counter_or = order_spectrum_for_share(threed_os,
                                                              twod_oc,
                                                              counter_or,
                                                              data[group_name][filename]['rsp'],
                                                              data[group_name][filename]['rsp_time'],
                                                              param.orderSpectrumCalcInfo,
                                                              param.orderCutCalcInfo,
                                                              sensor_index,
                                                              db_flag=
                                                              param.basicInfo[
                                                                  'dBFlag'])


    # threed_os = create_empty_threedos(param.orderSpectrumCalcInfo, param.taskInfo, sensor_index, db_flag=param.basicInfo['dBFlag'])
    # twod_oc = create_empty_twodoc(param.orderCutCalcInfo, param.taskInfo, sensor_index, db_flag=param.basicInfo['dBFlag'])
    # counter_or=0
    # threed_os, twod_oc, counter_or = order_spectrum(threed_os,
    #                                                           twod_oc,
    #                                                           counter_or,
    #                                                           data[group_name][
    #                                                               filename]['rsp'],
    #                                                           data[group_name][
    #                                                               filename]['rsp_time'],
    #                                                           param.orderSpectrumCalcInfo,
    #                                                           param.orderCutCalcInfo,
    #                                                           sensor_index,
    #                                                           db_flag=
    #                                                           param.basicInfo[
    #                                                               'dBFlag'])
    # filename = 'v2-cubic-1arpoints'
    threed_os_v2 = create_empty_threedos(param.orderSpectrumCalcInfo, param.taskInfo, sensor_index, db_flag=param.basicInfo['dBFlag'])
    twod_oc_v2 = create_empty_twodoc(param.orderCutCalcInfo, param.taskInfo, sensor_index, db_flag=param.basicInfo['dBFlag'])
    counter_or=0
    threed_os_v2, twod_oc_v2, counter_or = order_spectrum(threed_os_v2,
                                                              twod_oc_v2,
                                                              counter_or,
                                                              data[group_name][
                                                                  filename]['rsp'],
                                                              data[group_name][
                                                                  filename]['rsp_time'],
                                                              param.orderSpectrumCalcInfo,
                                                              param.orderCutCalcInfo,
                                                              sensor_index,
                                                              db_flag=
                                                              param.basicInfo[
                                                                  'dBFlag'])
    for i in range(len(twod_oc_v2[7]['xValue'])):
        twod_oc_v2[7]['xValue'][i]=np.mean(twod_oc_v2[7]['xValue'][i])
        twod_oc_v2[8]['xValue'][i]=np.mean(twod_oc_v2[8]['xValue'][i])
    for i in range(len(twod_oc[7]['xValue'])):
        twod_oc[7]['xValue'][i]=np.mean(twod_oc[7]['xValue'][i])
        twod_oc[8]['xValue'][i]=np.mean(twod_oc[8]['xValue'][i])

    plt.figure("40order")
    plt.plot(twod_oc[7]['xValue'],twod_oc[7]['yValue'],c='r',label='v3')
    plt.plot(twod_oc_v2[7]['xValue'],twod_oc_v2[7]['yValue'],c='b',label='v2')

    plt.figure("48order")
    plt.plot(twod_oc[8]['xValue'],twod_oc[8]['yValue'],c='r',label='v3')
    plt.plot(twod_oc_v2[8]['xValue'],twod_oc_v2[8]['yValue'],c='b',label='v2')


    plt.legend()
    plt.show()
    ic(1)

    # for sensor_id in sensor_list:
    #     for test_id in test_list:
    #         group_name = "sensor{}test{}".format(sensor_id, test_id)




