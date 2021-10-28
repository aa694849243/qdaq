#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2021/6/30 9:58
@Author : Wall@Synovate
@Email  : qxu@sonustc.com
@Version: 1.0.0
"""

# 赋值与pack_into性能对比。
# 全部变长度与只有一个变长度
import struct,time
from multiprocessing import shared_memory


from nptdms import TdmsFile
import numpy as np

from qdaq.speed_tools import trigger_detect_for_share


def readAllRawData(filename, channelName):
    allrawdata = dict()

    with TdmsFile.open(filename) as tdms_file:
        for channelName in channelName:
            allrawdata[channelName] = list(tdms_file['AIData'][channelName][:])

    return allrawdata

# jinkang的数据共10956800个点，共2773730上升沿
if __name__ == '__main__':
    samplesPerChan=8192
    struct_samplesPerChan = struct.Struct('f' * samplesPerChan)
    speed = np.array(readAllRawData("D:\\qdaq\\Simu\\jinkang-1vib.tdms", ["Speed"])["Speed"])
    shm_speed=shared_memory.SharedMemory(name="speed",create=True,size=4*(len(speed)))
    shm_trigger=shared_memory.SharedMemory(name="trigger",create=True,size=4*(len(speed)))


    time_before_rawdata_pack=time.time()
    offset_rawdata=0
    for i in range(int(len(speed)/samplesPerChan)+1):
        data=speed[i*samplesPerChan:samplesPerChan*(i+1)]

        if len(data) == samplesPerChan:
            struct_temp_for_rawdata = struct_samplesPerChan
        else:
            struct_temp_for_rawdata = struct.Struct('f' * len(data))

        # 将读到的数据存入共享内存

        struct_temp_for_rawdata.pack_into(shm_speed.buf, offset_rawdata, *data)
        offset_rawdata += 4 * len(data)
        # pack后删除临时变量
        del struct_temp_for_rawdata
    time_after_rawdata_pack = time.time()

    time_before_rawdata_assignment=time.time()
    index_rawdata=0
    speed_ndarray=np.ndarray(shape=(len(speed),),dtype="f",buffer=shm_speed.buf,offset=0)
    for i in range(int(len(speed)/samplesPerChan)+1):
        data=speed[i*samplesPerChan:samplesPerChan*(i+1)]
        speed_ndarray[index_rawdata:index_rawdata+len(data)]=data


        index_rawdata += len(data)
    time_after_rawdata_assignment = time.time()
    print(np.sum(np.power(speed-speed_ndarray,2)))


    speed_cal_info = dict()
    speed_cal_info["triggerLevel"] = 5.0
    speed_cal_info["triggerMode"] = 'Rising'
    offset_trigger=0
    index_trigger=0
    time_before_trigger_pack = time.time()
    for i in range(int(len(speed) / samplesPerChan) + 1):
        # data = speed[i * samplesPerChan:samplesPerChan * (i + 1)]
        trigger=trigger_detect_for_share(speed,i*samplesPerChan,(i+1)*samplesPerChan,speed_cal_info)
        struct_for_trigger=struct.Struct('f' * len(trigger))
        struct_for_trigger.pack_into(shm_trigger.buf,offset_trigger,*trigger)
        offset_trigger+=4*len(trigger)
        del struct_for_trigger
    time_after_trigger_pack = time.time()

    time_before_trigger_assignment = time.time()
    trigger_ndarray = np.ndarray(shape=(len(speed),), dtype="f", buffer=shm_speed.buf, offset=0)
    for i in range(int(len(speed) / samplesPerChan) + 1):
        # data = speed[i * samplesPerChan:samplesPerChan * (i + 1)]
        trigger=trigger_detect_for_share(speed,i*samplesPerChan,(i+1)*samplesPerChan,speed_cal_info)
        trigger_ndarray[index_trigger:index_trigger+len(trigger)]=trigger
        index_trigger+=len(trigger)
    time_after_trigger_assignment = time.time()

    print("time_for_rawdata_pack:{}".format(time_after_rawdata_pack-time_before_rawdata_pack))
    print("time_for_rawdata_assignment:{}".format(time_after_rawdata_assignment-time_before_rawdata_assignment))
    print("time_for_trigger_pack:{}".format(time_after_trigger_pack-time_before_trigger_pack))
    print("time_for_trigger_assignment:{}".format(time_after_trigger_assignment-time_before_trigger_assignment))

