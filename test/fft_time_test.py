
import time

import h5py
import numpy as np
from nptdms import TdmsFile


def read_raw_data(filename, channel_names, file_type='tdms'):
    """
    功能：读取原始数据（全部通道）
    输入：
    1. 原始数据文件名（全路径）
    2. 通道名列表
    3. 原始文件类型
    返回：原始数据（按通道名存储）
    """
    raw_data = dict()
    if file_type == 'tdms':
        with TdmsFile.open(filename) as tdms_file:
            for channel_name in channel_names:
                raw_data[channel_name] = list(tdms_file['AIData'][channel_name][:])
    else:
        with h5py.File(filename, 'r') as h5pyFile:
            data_group = h5pyFile['AIData']
            for channel_name in channel_names:
                raw_data[channel_name] = list(
                    # np.array(data_group[channel_name][11000000:19000000], dtype='float'))
                    np.array(data_group[channel_name], dtype='float'))
    return raw_data


if __name__ == "__main__":
    filename=r"d:/qdaq/simu/byd-alltests.h5"
    allrawdata = read_raw_data(filename, ["Sin"], "h5")

    # len_list=[32768,24575,19660]
    len_list=[32767,32768,32769]
    # count_list=[30,40,50]
    count_list=[30,30,30]
    for i in range(len(count_list)):
        time_start=time.time()
        for j in range(count_list[i]):
            frame=allrawdata["Sin"][j*len_list[i]:(j+1)*len_list[i]]
            # print(len(frame))
            a=np.abs(np.fft.rfft(frame))

        print(time.time()-time_start)

