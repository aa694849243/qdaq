# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:33:14 2020

@author: Sonus Wall

the special tools, for the functions not very needed/necessary for the whole program
"""

import logging
import traceback
import os
import shutil
import errno
import stat
import numpy as np
import h5py
from nptdms import TdmsFile

def compact_speed_calc_info(speed_calc_info):
    """
    功能：整合转速计算的相关参数信息
    输入：转速计算参数
    输出：
    1. 转速计算参数的名称
    2. 转速计算参数的值
    function: transfer the speedCal parameters into the format that can write into TDMS(this function is not necessary
    :param
    speedCalInfo(dict): include all the info of speed calculation as json format,default as the value in parameters
    :return
    parameterName(list): name list of speed calculation parameters
    parameterValue(list): name list of speed calculation parameters
    """
    parameter_name = list()
    parameter_value = list()
    for key, value in speed_calc_info.items():
        parameter_name.append(key)
        parameter_value.append(str(value))
    return parameter_name, parameter_value


def read_hdf5(filename, group_name, channel_name):
    """
    功能：读取hdf5文件内指定的数据
    输入：
    1. 文件名（全路径）
    2. 数据组名称
    3. 数据集名称（也可以说是通道名）
    返回：
    1. 数据
    2. 对应的属性
    """
    pp_dict = dict()
    with h5py.File(filename, 'r') as h5pyFile:
        data_group = h5pyFile[group_name]
        data_set = data_group[channel_name]
        for pp_key, pp_value in data_set.attrs.items():
            pp_dict[pp_key] = pp_value
        return np.array(data_set, dtype='float'), pp_dict


def read_tdms(filename, group_name, channel_name):
    """
    功能：读取tdms文件内指定的数据
    输入：
    1. 文件名（全路径）
    2. 数据组名称
    3. 数据集名称（也可以说是通道名）
    返回：
    1. 数据
    2. 对应的属性
    """
    pp_dict = dict()
    with TdmsFile.open(filename) as tdms_file:
        data_set = tdms_file[group_name][channel_name]
        for pp_key, pp_value in data_set.properties.items():
            pp_dict[pp_key] = pp_value
        return data_set[:], pp_dict


def move_file(src_file, dst_folder):
    """
    功能：移动文件到目标文件夹
    输入：
    1. 待移动到文件
    2. 目标文件夹
    返回：
    1. True：表示移动成功
    2. False：表示移动失败（出错）
    function: move the target file into another folder
    :param
    src_file(string): full path of source file(target file)
    dst_folder(string): target folder to save the target file
    :return
    True or False
    """
    try:
        fpath, fname = os.path.split(src_file)
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        shutil.move(src_file, os.path.join(dst_folder, fname))
        logging.info("succeed to move {} from {} -> {}".format(fname, fpath, dst_folder))
        return True
    except Exception:
        logging.warning("failed to move the file, failed msg:" + traceback.format_exc())
        return False


def copy_file(src_file, dst_folder):
    """
    功能：复制文件到目标文件夹
    输入：
    1. 待复制到文件
    2. 目标文件夹
    返回：
    1. True：表示复制成功
    2. False：表示复制失败（出错）
    function: copy the target file into another folder
    :param
    src_file(string): full path of source file(target file)
    dst_folder(string): target folder to save the target file
    :return
    True or False
    """
    try:
        fpath, fname = os.path.split(src_file)
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        shutil.copyfile(src_file, os.path.join(dst_folder, fname))
        logging.info("succeed to copy {} from {} -> {}".format(fname, fpath, dst_folder))
    except Exception:
        logging.warning("failed to copy the file, failed msg:" + traceback.format_exc())
        return False


def handle_remove_read_only(func, path, exc):
    # 解决使用shutil来删除文件夹时报PermissionError, 配合empty_folder使用
    excvalue = exc[1]
    if func in (os.rmdir, os.remove, os.unlink) and excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777
        func(path)
    else:
        raise


def empty_folder(folder):
    """
    功能：清空目标文件夹
    输入：目标文件夹
    返回：
    1. True：表示清空成功
    2. False：表示清空失败（出错）
    function: mandatory remove all the files inside the target folder
    :param
    folder(list): the target folder need to be empty
    :return
    no
    """
    try:
        if os.path.exists(folder):
            shutil.rmtree(folder, onerror=handle_remove_read_only)
        os.makedirs(folder)
        return True
    except Exception:
        logging.warning("failed to empty the folder, failed msg:" + traceback.format_exc())
        return False


def db_revertion(data, ref_value):
    """
    功能：dB转换为初始值
    输入：
    1. 结果数据，可以是一维或者二维数据
    2. 参考值，用于db的计算
    返回：转换后的结果
    """
    return np.power(10, data/20) * ref_value


# ==============下面的这些函数目前已暂停使用=================
def find_proximal_element(element, data):
    """
    暂停使用
    功能：在目标序列里寻找最接近目标元素的数据点，返回其索引，目前已暂停使用，序列较大时耗时过长
    输入：
    1. 目标元素
    2. 目标序列
    返回：与目标元素最接近的数据点的索引
    function: find the proximal element,update at 2020/8/3, use numpy
    :param
    element(float): the target element
    data(list): the data list needed to find the element
    :return
    index: the index of the proximal element
    """
    idx = (np.abs(np.asarray(data) - element)).argmin()
    return idx


def find_proximal_elements(elements, data):
    """
    暂停使用
    功能：find_proximal_element的升级版，可寻找多个元素
    function: find the proximal element,update at 2020/8/3, use numpy
    :param
    elements(list): the target element
    data(list): the data list needed to find the element
    :return
    index(list): the index of the proximal element
    """
    idx = list()
    for element in elements:
        idx.append((np.abs(np.asarray(data) - element)).argmin())
    return idx


def get_value(target_list1, list1, list2):
    """
    暂停使用
    功能：匹配目标序列的值，主要用于通过时间信息和转速曲线来匹配出转速
    输入：
    1. 目标序列，这里主要指的是时间序列
    2. 完整序列1，目标序列的完整序列，这里通常指的是转速曲线的X轴，即时间
    2. 完整序列2，返回序列的完整序列，这里通常指的是转速曲线的Y轴，即转速
    function: to get the target time or speed based on the speed curve
    :param
    targetList1(list): speed or time, if this one is speed, the return is time 目标时间点
    list1(list): full info of targetList1, e.g. if targetList1 is speed, this is also speed 时间
    list2(list): corresponding list of list1, e.g. if list1 is speed, list2 is time 与时间的转速
    :return
    targetList2(list): speed or time, if targetList1 is speed, this one is time
    """
    target_list2 = list()
    for value in target_list1:
        idxs = find_proximal_elements(value, list1)
        # target_list2.append(np.mean(list2[idxs[0]: idxs[1]+1]))
        # 修改为float64类型，Object of type float32 is not JSON serializable
        target_list2.append(np.mean(list2[idxs[0]: idxs[1]+1]).astype('float64'))
    return target_list2


def get_time(target_list1):
    """
    功能：当记录的时间为开始时间和结束时间时，合并为一个时间，即求平均值
    输入：时间序列
    返回：时间序列
    function: to get the target time or speed based on the speed curve
    :param
    targetList1(list): speed or time, if this one is speed, the return is time
    :return
    targetList2(list): speed or time, if targetList1 is speed, this one is time
    """
    target_list2 = list()
    for value in target_list1:
        target_list2.append(np.mean(value))
    return target_list2


if __name__ == "__main__":
    # 测试该模块相关函数
    import time
    import matplotlib.pyplot as plt
    # 测试HDF5文件读取
    s_time_hdf5 = time.time()
    hdf5_filename = 'test512.h5'
    data, pp = read_hdf5(hdf5_filename, 'AIData', 'Vib')
    print(time.time() - s_time_hdf5)
    print(data[0], pp)
    print(len(data)/51200)
    print(type(data))
    plt.plot(data)
    plt.show()

    f = np.fft.rfftfreq(len(data), 1 / 51200)
    fvib = np.fft.rfft(data) / len(data) * 2
    plt.figure()
    plt.plot(f, np.abs(fvib))
    plt.show()
    # 测试TDMS文件读取
    # s_time_tdms = time.time()
    # tdms_filename = r'D:\qdaq\Simu\aiwayTest\20210514\TEST0515-9.tdms'
    # data, pp = read_tdms(tdms_filename, 'AIData', 'Speed')
    # print(time.time() - s_time_tdms)
    # print(data[0], pp)
    # print(type(data))
    pass
