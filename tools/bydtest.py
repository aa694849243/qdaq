import h5py
import numpy as np
from nptdms import TdmsFile

def write_hdf5(filename, group_name, channel_name, channel_data, properties=None):
    """
    功能：往指定的HDF5文件中写入数据
    输入：
    1. HDF5文件
    2. 数据组名称
    3. 数据通道名称
    4. 数据
    5. 属性
    返回：本地hdf5文件
    :param
    h5file(fp): HDF5格式文件句柄
    group_name(str)：数据组名称
    channel_name(str)：数据集名称
    channel_data(array)：待写入的数据集数据值
    properties(dict)：数据集属性信息
    :return
    local file
    """
    with h5py.File(filename, 'a') as h5file:
        group_list = [h5file[key].name.split('/')[1] for key in h5file.keys()]
        if group_name in group_list:
            # 数据组已存在则直接打开
            data_group = h5file[group_name]
        else:
            # 数据组不存在则新建
            data_group = h5file.create_group(group_name)
        # 写入数据集数据
        data_set = data_group.create_dataset(channel_name, data=np.array(channel_data,
                                                                         dtype='float32'))
        if properties:
            # 写入属性信息（若存在）
            for key, value in properties.items():
                try:
                    data_set.attrs[key] = value
                except Exception:
                    print(key)
                    print(value)


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
rawdata=read_raw_data("d:/qdaq/simu/byd-alltests.h5",["Sin","Cos","Vib1","Vib2"],file_type="h5")

raw_data_filename="d:/qdaq/simu/byd-alltests-2.h5"
write_hdf5(raw_data_filename, 'AIData',"Sin",rawdata["Sin"])
write_hdf5(raw_data_filename, 'AIData',"Cos",rawdata["Cos"])
write_hdf5(raw_data_filename, 'AIData',"Vib1",rawdata["Vib2"])
write_hdf5(raw_data_filename, 'AIData',"Vib2",rawdata["Vib1"])

