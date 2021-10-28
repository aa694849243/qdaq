# 给定阶次彩图的tdms画出阶次彩图


import os
import re
import sys
import getopt
import json
import datetime
import logging
import traceback
from nptdms import TdmsFile
import numpy as np
import matplotlib.pyplot as plt


sensitivity = 9.81
ref_accel = 1e-6
ref_value = 20 * np.log10(ref_accel * sensitivity)

# 读取TDMS文件指定group的内容
def Speed_Vibration_X(tdms_file, Colormap_group_name):
    colormap = list()
    x_axis = list()
    temp = 0

    def tryint(s):
        try:
            return int(s)
        except ValueError:
            return s

    def str2int(v_str):
        return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

    data_group = tdms_file[Colormap_group_name]
    channels_name_list = [channel.name for channel in data_group.channels()]
    # 排序以防止数据存储循序错乱
    channels_name_list.sort(key=str2int)
    # print(channels_name_list)

    step = data_group[channels_name_list[1]].properties["wf_increment"]
    # sort之后先是第一个传感器的数据，时候是第二个传感器的数据

    speed = list((data_group[channels_name_list[0]])[:])

    for channel_name in channels_name_list[1:]:
        # colormap.append(list(data_group[channel_name]))
        colormap.append(list(20 * np.log10(data_group[channel_name]) - ref_value))

    for i in range(0, len(colormap[0])):
        temp += step
        x_axis.append(temp)

    # return speed, colormap, x_axis
    return speed, colormap[:len(colormap)//2], x_axis


if __name__ == '__main__':
    path = "D:/ShirohaUmi/work_document/mcc/aqsrt/ordercolormap"
    files = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x)) and os.path.splitext(x)[1] == ".tdms"]

    # sourcefile =
    # Step3: read the TDMS file and write it in JSON format


    for file in files:

        with TdmsFile.open(os.path.join(path,file)) as tdms_file:

            result = list()
            for group in tdms_file.groups():
                if "ATEOLOSMAP_" in group.name:
                    Speed, Colormap, Order = Speed_Vibration_X(tdms_file, group.name)
                    plt.figure(file)
                    # Order=Order[:len(Order)//2]
                    # Speed=Speed[:len(Speed) // 2]
                    # Colormap=Colormap[:len(Colormap) // 2]



                    plt.pcolormesh(Order[:len(Order)//5],Speed[:len(Speed)],np.array(Colormap)[:,:len(Colormap[0])//5],cmap="jet")
                    # plt.pcolormesh(Order,Speed,np.array(Colormap),cmap="jet")


    plt.show()
