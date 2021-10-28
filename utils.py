# -*- coding: utf-8 -*-
"""
Created on Thu Mar 5 10:33:14 2020

@author: Sonus Wall

this module is for common functions, add resultId in the request of create report
说明：该模块主要提供一些通用的函数方法，比如，写入数据到tdms，hdf5，
"""
import shutil
import time
import traceback

import h5py
import numpy as np
from nptdms import TdmsWriter, ChannelObject, TdmsFile
import json
import os
import requests
import base64
import pickle
from common_info import qDAQ_logger, server_ip, basic_folder, ftp_flag
from ftp_tools import file_upload, folder_check, ftp_connect, file_check


def create_properities(wf_starttime, wf_increment, wf_samples):
    """
    功能：创建原始数据和阶次彩图数据写入所需要的通道属性信息
    输入：
    1. 开始时间
    2. 数据采样间隔（可根据该参数得到采样率，如果是彩图可推出X轴，如阶次序列）
    3. 帧长
    返回：属性数据（字典类型）
    function: create the formatted properties for TDMS write
    :param
    wf_starttime(datetime.datetime):the start time of writing the TDMS(also the time of getting the data)
    wf_increment(float): 1.0/sample rate
    wf_samples(int): the num of each time write into TDMS file
    :return
    properties(dict): include the info of wf_increment, wf_samples, wf_start_offset, wf_start_time, Start Index
    """
    properties = dict()
    # wf_increment为数据对应的x轴间隔
    properties['wf_increment'] = wf_increment
    # wf_samples帧长
    properties['wf_samples'] = wf_samples
    # wf_start_offset开始点索引偏置，使用tdms的time_track时能用到，意义不大，但还是写上吧
    properties['wf_start_offset'] = 0
    # wf_start_time，数据记录开始时间（一般都是这次测试开始的时间），供参考，实际意义不大
    properties['wf_start_time'] = str(wf_starttime)
    # properties['Start Index'] = 1
    return properties


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


def hdf5_data_confirm(filename, group_name, channel_name):
    """
    功能：确认hdf5文件是否已写入目标通道的数据
    输入：
    1. 目标文件名（全路径）
    2. 目标数据组名
    3. 目标数据集名
    返回：
    1. True：数据集不存在，需要写入
    2. False：数据集已存在， 不需要写入
    """
    # 确认文件是否存在
    if os.path.exists(filename):
        # 存在则继续确认数据组
        with h5py.File(filename, 'a') as h5file:
            group_name_list = [h5file[key].name.split('/')[1] for key in h5file.keys()]
            if group_name in group_name_list:
                # 如果数据组存在则继续确认数据集
                channel_name_list = list(h5file[group_name].keys())
                if channel_name in channel_name_list:
                    # 数据集已存在
                    return False
                else:
                    # 数据集不存在
                    return True
            else:
                # 数据组不存在
                return True
    else:
        # 文件不存在
        return True


def tdms_data_confirm(filename, group_name, channel_name):
    """
    功能：确认tdms文件是否已写入目标通道的数据
    输入：
    1. 目标文件名（全路径）
    2. 目标数据组名
    3. 目标数据集名
    返回：
    1. True：数据集不存在，需要写入
    2. False：数据集已存在，不需要写入
    """
    # 确认文件是否存在
    if os.path.exists(filename):
        # 存在则继续确认数据组
        with TdmsFile.open(filename) as tdms_file:
            group_name_list = [group.name for group in tdms_file.groups()]
            if group_name in group_name_list:
                # 如果数据组存在则继续确认数据集
                channel_name_list = [channel.name for channel in
                                     tdms_file[group_name].channels()]
                if channel_name in channel_name_list:
                    # 数据集已存在
                    return False
                else:
                    # 数据集不存在
                    return True
            else:
                # 数据组不存在
                return True
    else:
        # 文件不存在
        return True


def write_tdms(filename, group_name, channel_name, data, properties=None, mode='a'):
    """
    功能：按通道写入数据到TDMS文件中（tdms数据结构，文件（file）->数据组（group）->数据通道（channel））
    输入:
    1. 文件名（全路径）
    2. tdms的数据组名称
    3. tdms的数据通道
    4. 要写入指定通道数据
    5. 属性信息，默认为空
    6. 写入模式，默认为添加模式
    返回：本地tdms文件
    function: write the data into TDMS file
    :param
    filename(string): the full path of target TDMS file
    groupname(string): the group name for TDMS write
    channelname(string): the channel name for TDMS write
    data(list): data need to write into TDMS
    properties(dict): the properties of channel data, default as {}
    mode(char): 'w' or 'a', 'w' means it will remove all the existed data and write new data,
    'a' means it just append new data, hold the existed data, default as 'a'
    :return:
    existed TDMS file
    """
    # 创建数据通道对象
    channel_object = ChannelObject(group_name, channel_name, data, properties)
    # 写入数据到文件
    with TdmsWriter(filename, mode) as tdms_writer:
        tdms_writer.write_segment([channel_object])


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

def bit24_to_int(byte_data):
    n = len(byte_data) // 3
    num = []
    for i in range(n):
        data = byte_data[i * 3:i * 3 + 3]
        number = int.from_bytes(data, byteorder='little', signed=True)
        num.append(number)
    return num

def bit_to_int(byte_data,len_vib):
    byte_count=len_vib//8
    n = len(byte_data) // byte_count
    num = []
    for i in range(n):
        data = byte_data[i * byte_count:i * byte_count + byte_count]
        number = int.from_bytes(data, byteorder='little', signed=True)
        num.append(number)
    return num


def timestamp_to_time(timestamp):
    """
    功能：将时间戳转化为毫秒
    输入：
    1. 时间戳
    返回：
    1. 秒级时间（长整型）
    """
    return int(time.mktime(timestamp.timetuple()) * 1000)
def copy_file(src_file, dst_folder):
    """
    功能：复制文件到目标文件夹
    输入：
    1. 待复制到文件
    2. 目标文件夹
    返回：无
    function: copy the target file into another folder
    :param
    src_file(string): full path of source file(target file)
    dst_folder(string): target folder to save the target file
    :return
    True or False
    """
    fpath, fname = os.path.split(src_file)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    shutil.copyfile(src_file, os.path.join(dst_folder, fname))

def send_raw_data(url_info, folder_info, data_info, format_time):
    """
    功能：发送原始数据(通过ftp服务）
    输入：
    1. 原始数据状态确认接口，如http://192.168.2.109:8081/api/storage/originData/status
    2. 数据解析接口，数据上传完成后调该接口进行数据解析（彩图计算等），如http://192.168.2.109:8081/api/storage/originData/save
    3. 文件目录信息，包含原始数据的目录结构，本地文件目录和远程的目录结构
    4. 接口请求需要传递的参数，包含文件名（如test210913_210913085303.h5），系统名，类型名，序列号，时间戳
    返回：数据状态（再次请求只要请求status即可确认数据在服务器端的状态）
    1. True，上传成功
    2. False，上传失败
    """
    # 确认数据状态
    try:
        # 创建请求头
        headers = {'Content-Type': 'application/json;charset=UTF-8'}
        # 发送请求，并获取返回信息，可以设置超时时间以避免长时间无响应导致一直等待，timeout = 3
        # 这里用的是get请求，参数是通过param传递的
        response = requests.get(url_info['dataStatus'], headers=headers, params=data_info, timeout=3)
        qDAQ_logger.info('raw data status: {}'.format(response.text))
        print(response.url)
    except Exception:
        # 避免请求失败,若失败需要记录信息（意味着无法上传，直接记录相关信息）
        qDAQ_logger.error('raw data status query error')
        record_info = dict()
        record_info["localFolder"] = folder_info['localFolder']
        record_info["dataInfo"] = data_info
        record_info["ip"] = server_ip
        record_info['formatTime'] = format_time
        record_info['msg'] = "raw data status query failed"
        record_filename = os.path.join(folder_info["uploadError"],
                                       os.path.splitext(data_info["fileName"])[0] + '.json')
        with open(record_filename, 'w') as f:
            json.dump(record_info, f)
        return False
    else:
        # 原始数据状态查询成功, 根据返回的状态执行对应的操作
        try:
            response_flag = response.json()['code']
            if response_flag == 200:
                # 查询成功，开始判断原始数据的状态
                status_flag = response.json()['data']
                local_filepath = os.path.join(folder_info['localFolder'], data_info['fileName'])
                if status_flag == 4:
                    # 表示数据不存在，执行上传操作（区分是否为单机版）
                    try:
                        if ftp_flag:
                            # 上传数据至服务器
                            sv_ftp = ftp_connect(server_ip)
                            print(sv_ftp.getwelcome())
                            remote_folder = folder_check(sv_ftp, basic_folder, data_info['system'], data_info['type'], data_info['serial'], format_time)
                            remote_filepath = remote_folder + '/' + data_info['fileName']
                            if file_check(sv_ftp, remote_filepath):
                                qDAQ_logger.info("remote file {} already existed".format(remote_filepath))
                                # 若文件已存在则删除
                                sv_ftp.delete(remote_filepath)
                            # 开始上传
                            file_upload(sv_ftp, local_filepath, remote_filepath)
                            # 上传完成，退出ftp
                            sv_ftp.quit()
                            pass
                        else:
                            remote_folder = os.path.join(os.path.join(os.path.join(os.path.join(basic_folder, data_info['system']), data_info['type']), data_info['serial']), format_time)
                            if not os.path.exists(remote_folder):
                                os.makedirs(remote_folder)
                            remote_filepath = os.path.join(remote_folder, data_info['fileName'])
                            if os.path.exists(remote_filepath):
                                qDAQ_logger.info("remote file {} already existed".format(remote_filepath))
                                # 若文件已存在则删除
                                os.remove(remote_filepath)
                            # 复制文件至目标路径
                            copy_file(local_filepath, remote_folder)
                    except Exception:
                        # 上传或拷贝出错，记录信息，之后重新上传
                        qDAQ_logger.error("raw data upload or move error, msg:" + traceback.format_exc())
                        record_info = dict()
                        record_info["localFolder"] = folder_info['localFolder']
                        record_info["dataInfo"] = data_info
                        record_info["ip"] = server_ip
                        record_info['formatTime'] = format_time
                        record_info['status'] = 1
                        record_info['msg'] = "raw data upload or move failed"
                        record_filename = os.path.join(folder_info["uploadError"], os.path.splitext(data_info["fileName"])[0] + '.json')
                        with open(record_filename, 'w') as f:
                            json.dump(record_info, f)
                        return False
                    else:
                        # 完成上传或者移动后开始请求解析
                        try:
                            qDAQ_logger.info("data upload or move finished, start to analysis")
                            # 未发现错误则删除本地文件（即源文件）
                            os.remove(local_filepath)
                            # 创建请求头
                            headers = {'Content-Type': 'application/json;charset=UTF-8', 'from': 'Y'}
                            # 发送请求，并获取返回信息，可以设置超时时间以避免长时间无响应导致一直等待，timeout = 3
                            # 这里用的是post请求，参数是通过data传入的
                            response = requests.post(url_info['dataAnalysis'], headers=headers, data=json.dumps(data_info), timeout=3)
                            qDAQ_logger.info('raw data analysis: {}'.format(response.text))
                            print(response.url)
                            return True
                        except Exception:
                            # 避免请求失败
                            qDAQ_logger.error('raw data analysis error' + traceback.format_exc())
                            record_info = dict()
                            record_info["localFolder"] = folder_info['localFolder']
                            record_info["dataInfo"] = data_info
                            record_info["ip"] = server_ip
                            record_info['formatTime'] = format_time
                            record_info['msg'] = "raw data status query failed"
                            record_filename = os.path.join(folder_info["analysisError"],
                                                           os.path.splitext(data_info["fileName"])[
                                                               0] + '.json')
                            with open(record_filename, 'w') as f:
                                json.dump(record_info, f)
                            return False
                elif status_flag == 1:
                    # 已完成数据解析（意味着该数据已存在即上传成功），可以直接删除数据
                    os.remove(local_filepath)
                    qDAQ_logger.info('data already upload and analysis done')
                    return False
                elif status_flag == 2:
                    # 数据已进入redis， 但是未开始解析（原始数据已存在），可以记录下信息然后然后再次确认
                    os.remove(local_filepath)
                    qDAQ_logger.info('data already upload but waiting for analysis')
                    return False
                elif status_flag == 3:
                    # 解析失败，表示原始数据已存在
                    os.remove(local_filepath)
                    qDAQ_logger.info('data already upload but waiting for analysis')
                    return False
                else:
                    # 返回的状态异常，没必要再次上传
                    qDAQ_logger.info('data status not found')
                    return False
            else:
                # 查询失败，可能是服务器状态异常
                qDAQ_logger.error("service error, please check SigMA service")
                record_info = dict()
                record_info["localFolder"] = folder_info['localFolder']
                record_info["dataInfo"] = data_info
                record_info["ip"] = server_ip
                record_info['formatTime'] = format_time
                record_info['msg'] = "raw data status query error(service error)"
                record_filename = os.path.join(folder_info["uploadError"],
                                               os.path.splitext(data_info["fileName"])[
                                                   0] + '.json')
                with open(record_filename, 'w') as f:
                    json.dump(record_info, f)
                return False

        except Exception:
            # 避免SigMA返回的状态解析出错（服务器状态异常）
            qDAQ_logger.error("response error, please check SigMA service")
            record_info = dict()
            record_info["localFolder"] = folder_info['localFolder']
            record_info["dataInfo"] = data_info
            record_info["ip"] = server_ip
            record_info['formatTime'] = format_time
            record_info['msg'] = "raw data status query error(SigMA error)"
            record_filename = os.path.join(folder_info["uploadError"],
                                           os.path.splitext(data_info["fileName"])[
                                               0] + '.json')
            with open(record_filename, 'w') as f:
                json.dump(record_info, f)
            return False


def write_json(filename, data, flag=0, indent=4):
    """
    功能：json格式数据写入文件，其中包括其他编码格式（为了防止用户在本地修改文件）。默认不进行其他方式编码
    输入：
    1. filename：文件名，包含完整的路径信息
    2. data：json格式的数据
    3. flag：编码标志，默认为0，表示直接写入json字符串，标志及其意义如下：
        3.1 flag=1：b64encode
        3.2 flag=2：b32encode
        3.3 flag=3：b16encode
        3.4 flag=4：pickle进行序列化
        3.5 其他，直接写入json字符串
    返回：写入数据到文件中
    function: write the data into json file
    :param
    filename: the full path of target JSON file
    data(dict): the data need to write into the JSON file
    :return
    existed json file
    """
    if flag == 1:
        # b64编码
        with open(filename, 'wb') as f:
            f.write(base64.b64encode(json.dumps(data).encode('utf-8')))
    elif flag == 2:
        # b32编码
        with open(filename, 'wb') as f:
            f.write(base64.b32encode(json.dumps(data).encode('utf-8')))
    elif flag == 3:
        # b16编码
        with open(filename, 'wb') as f:
            f.write(base64.b16encode(json.dumps(data).encode('utf-8')))
    elif flag == 4:
        # pickle进行序列化
        open(filename, "wb").write(pickle.dumps(data))
    else:
        # 直接写入
        with open(filename, 'w') as f:
            json.dump(data, f, indent=indent)


def read_json(filename, flag=0):
    """
    功能：读取json格式的数据（里面是json字符串），可指定按那种编码格式服务（需根据写入时的编码格式来定）
    输入：
    1. 文件名（包含完整路径）
    2. 编码格式包括：
        2.1 flag=1：b64decode
        2.2 flag=2：b32decode
        2.3 flag=3：b16decode
        2.4 flag=4：pickle进行反序列化
        2.5 其他，直接读取json字符串
    返回：字典类型的json格式数据
    function: read out the data saved in the target JSON file
    :param
    filename(str): read the target JSON file into the data
    :return
    data(dict): the data read from the JSON file
    """
    if flag == 1:
        # b64解码
        with open(filename, 'rb') as f:
            data = json.loads(base64.b64decode(f.read()).decode('utf-8'))
    elif flag == 2:
        # b32解码
        with open(filename, 'rb') as f:
            data = json.loads(base64.b32decode(f.read()).decode('utf-8'))
    elif flag == 3:
        # b16解码
        with open(filename, 'rb') as f:
            data = json.loads(base64.b16decode(f.read()).decode('utf-8'))
    elif flag == 4:
        # pickle进行反序列化
        data = pickle.loads(open(filename, "rb").read())
    else:
        # 直接读取
        with open(filename, 'r') as f:
            data = json.load(f)
            f.close()
    return data


def file_namer(serial_number, record_time):
    """
    功能：生成文件名，用于，原始数据文件，彩图文件，转速曲线文件，
    输入：
    1. 序列号
    2. 时间戳
    返回：以下划线分割的文件名
    """
    return '_'.join([serial_number, record_time])


def send_result_data(url, data, filename, folder_info):
    """
    功能：发送结果数据
    输入：
    1. 结果数据发送接口
    2. 结果数据
    3. 文件名称，文件名称，主要用于发送失败或错误进行数据保存
    4. 文件目录信息，包括网络错误和数据内容异常等错误时数据保存目录信息
    返回：数据发送状态，True表示发送成功，False表示发送失败
    function: to send the result data to the server/test bench, update time out of request
    :param
    url(str): the url for the requests to send data to server and test bench
    data(json str): the json str include the info need to send(json.dumps(dict))
    filename: the fileName of the data need to send, in case it can not be send out
    folder_info: folder path to save the result file in case send error
    :return
    status: True means succeed, and False means failed
    """
    try:
        # 创建请求头
        headers = {'Content-Type': 'application/json;charset=UTF-8'}
        # 发送请求，并获取返回信息，可以设置超时时间以避免长时间无响应导致一直等待，timeout = 3
        response = requests.post(url, data=json.dumps(data), headers=headers, timeout=3)
        # 记录反馈信息
        qDAQ_logger.info(response.text)
        # print(response.text)
    except Exception:
        qDAQ_logger.info("can not send out the data, please check the internet!")
        # save data to folder1(internet error)
        write_json(os.path.join(folder_info['reportNetError'], filename), data)
        return False
    else:
        # 若发送成功并获得反馈，则确认返回的信息
        try:
            # check response status_code to confirm if request result is correct
            if response.json()['code'] == 200:
                # 返回信息的代码为200表示发送成功
                # if send data succeed, create a report for that
                qDAQ_logger.info("result data send successful!")
                return True
            elif response.json()['code'] == 3000:
                # 返回信息的代码为3000表示数据异常，需要确认数据内容
                # save data to folder2(information missing)
                write_json(os.path.join(folder_info['reportInfoMissing'], filename), data)
                qDAQ_logger.info(response.text)
                return False
            else:
                # 返回信息的代码暂认定为其他错误（比如服务器状态不对）之后重新发送
                write_json(os.path.join(folder_info['reportNetError'], filename), data)
                qDAQ_logger.info(response.text)
                return False
        except Exception:
            # 避免收到的反馈不包含代码信息，比如服务未正常启动
            qDAQ_logger.warning('data send error, please check sigma service!')
            write_json(os.path.join(folder_info['reportNetError'], filename), data)
            return False


# 最大阶次与最小速度之间要满足如下关系，该功能尚未使用
def max_order_confirm(min_speed, max_order, sample_rate):
    """
    function: check if max order and minimum speed can be used for angular resampling
    :param
        min_speed: 转速识别的最小转速（所有测试段）
        sample_rate(int):
    :return:
        bool: true means the setting is correct, false means it's wrong
    """
    max_order_available = (60 * sample_rate) / (min_speed * 2 * 1.6384)
    if max_order <= max_order_available:
        return True
    else:
        return False


def decrypt_data(aes_obj, filename):
    """
    功能：读取参数配置或者界限值并进行AES解密
    输入：
    1. AES对象，内含加密和解密的方法
    2. 参数配置或者界限值文件名（全路径）
    """
    # 读取待解密数据
    with open(filename, 'rb') as f:
        encrypted_data_str = f.read()
    # 解密
    decrypted_data_str = aes_obj.decrypt(encrypted_data_str)
    return json.loads(decrypted_data_str)


if __name__ == '__main__':
    # 测试读取和写入数据函数
    raw_filename = r'D:\qdaq\Config\AP4000_paramReceived.json'
    raw_config_data = read_json(raw_filename)
    # 测试base64和pickle
    new_filename = r'D:\qdaq\test\AP4000_paramReceived.json'
    write_json(new_filename, raw_config_data, flag=4)
    new_config_data = read_json(new_filename, flag=4)

    # 测试原始数据读取
    rawdata_filename = r'D:\qdaq\Simu\simu.tdms'
    rawdata_filename = r'D:\qdaq\Data\Ino-TTL-Type3\2107\21071508\1000000000000005_200626114444_210715080307.h5'
    channel_names = ['Speed', 'Vibration', 'Torque']
    raw_data = read_raw_data(rawdata_filename, channel_names, 'hdf5')
    pass
