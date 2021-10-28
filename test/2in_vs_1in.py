from nptdms import TdmsFile
import numpy as np
import time

from qdaq.utils import read_raw_data, write_hdf5
from qdaq.speed_tools import rpm_calc, resolver_single_signal, resolver_single_signal_for_share, \
    rpm_calc_for_share, resolver_single_signal_for_share_middle, resolver
import matplotlib.pyplot as plt
from parameters import Parameters

if __name__ == '__main__':

    filename = 'D:/qdaq/Simu/resolvererror.tdms'
    channelName = ['Sin', 'Cos', 'Vib1','Vib2']
    allrawdata = read_raw_data(filename, channelName, "tdms")

    # filename = 'D:/qdaq/Simu/PM61191-2vibs.tdms'
    # allrawdata = read_raw_data(filename, channelName, "tdms")

    cdata = dict()
    counter = 0
    counter_angle = 0
    param = Parameters('D:/qdaq\\Config\\resolvererror_paramReceived.json')
    # param = Parameters('D:/qdaq\\Config\\PM61191-2vibs-resolver_paramReceived.json')
    channelNames = param.taskInfo['channelNames']

    plt.figure("allrawdata")
    plt.plot(allrawdata["Sin"][:2000], c='r', label="Sin")
    plt.plot(allrawdata["Cos"][:2000], c='b', label="Cos")
    plt.legend()

    sampleRate = param.taskInfo["sampleRate"]
    sampsPerChan = param.taskInfo["sampsPerChan"]


    length=len(allrawdata["Sin"])
    gv_dict_rawdata=dict()
    for channelName in channelNames:
        gv_dict_rawdata[channelName]=np.zeros(length)
    index_rawdata=0
    trigger_per_fish = np.int(param.speedCalcInfo['ppr'] / 2 / param.speedCalcInfo['coils'])
    trigger_array=np.zeros(length)
    index_trigger=0
    rpml_array=np.zeros(length)
    rpm_array=np.zeros(length)
    rpm_index=0

    # 上一帧转速计算用到的最后一个trigger是第几个trigger
    last_rpm_cal_index = 0

    print("start")

    # 双路旋变需要的数据
    last_angle_l1f, loc_l1f = 0, None
    last_angle_l2f, loc_l2f = 0, None


    while True:
        # print(counter)
        index_rawdata_backup=index_rawdata
        for channelName in channelNames:
            cdata[channelName] = allrawdata[channelName][
                                 counter * sampsPerChan:(counter + 1) * sampsPerChan]
            # cdata[channelName] = allrawdata[channelName]
            len_rawdata_frame = len(cdata[channelName])
            gv_dict_rawdata[channelName][index_rawdata:index_rawdata + len_rawdata_frame] = \
                cdata[channelName]
        index_rawdata += len_rawdata_frame

        if (len_rawdata_frame==0):
            plt.figure("rpm")
            plt.plot(rpml_array[:rpm_index], rpm_array[:rpm_index], c='r')
            plt.show()
            break

        if counter==0:
            last_frame_status=-1
            left_index=0

        # if counter==15:
        #     time.sleep(0.01)

        index_trigger_backup=index_trigger
        index_trigger, left_index, last_frame_status = resolver_single_signal_for_share_middle(
            gv_dict_rawdata["Sin"], left_index,
            index_rawdata, last_frame_status,
            trigger_per_fish, trigger_array, index_trigger,param.speedCalcInfo)
        print(index_trigger)

        # temptl, last_angle_l1f, loc_l1f, last_angle_l2f, loc_l2f, counter_angle = resolver(
        #     gv_dict_rawdata['Cos'][index_rawdata_backup:index_rawdata],
        #     gv_dict_rawdata['Sin'][index_rawdata_backup:index_rawdata],
        #     param.speedCalcInfo['triggerLevel'],
        #     param.speedCalcInfo['coils'],
        #     param.speedCalcInfo['ppr'],
        #     last_angle_l1f,
        #     loc_l1f,
        #     last_angle_l2f,
        #     loc_l2f,
        #     counter_angle,
        #     sampsPerChan)
        # trigger_array[index_trigger:index_trigger + len(temptl)] = \
        #     np.array(temptl) + index_rawdata_backup
        # index_trigger += len(temptl)


        # if np.where(np.diff(trigger_array[index_trigger_backup-1:index_trigger])>40)[0].size>0:
        #     print(counter)
        #     pass

        last_rpm_cal_index, rpm_index = rpm_calc_for_share(trigger_array,
                                                           last_rpm_cal_index,
                                                           index_trigger - 1,
                                                           sampleRate,
                                                           param.speedCalcInfo,
                                                           rpml_array,
                                                           rpm_array, rpm_index)
        # plt.figure("rpm")
        # plt.plot(rpml_array[:rpm_index], rpm_array[:rpm_index], c='r')
        # plt.show()
        # break
        counter+=1
    filename="D:\qdaq\对比\\v2v3\\BYD-test.h5"
    write_hdf5(filename,"speedData","speedLoc",rpml_array[:rpm_index])
    write_hdf5(filename,"speedData","speedValue",rpm_array[:rpm_index])
    print("over")
