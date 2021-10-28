# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 18:22:59 2020
updated on 20210617, 更新了自定义换算方法
@author: Wall@Synovate

function: 1. handle NI DAQ Tasks(setup, start, stop, clear, save)
          2. define function to save raw data as TDMS
          3. get some info of NI DAQmx task/device
功能：1. 定义个NI DAQmx的方法类
     2. 定义了NI采集设备基本操作方法
说明：使用该模块的方法需要提前安装nidaqmx的相关驱动
"""


import nidaqmx
import sys
import logging
import traceback


def get_daq_tasks():
    """
    功能：获取在NI MAX中保存的任务
    输入：无
    返回：NI DAXmx任务列表
    :return
    task list defined in the NI-DAQmx(task_list:task names)
    """
    system = nidaqmx.system.system.System()
    task_list = system.tasks.task_names
    return task_list


def get_taskinfo(taskname):
    """
    # 获取指定采集任务的相关信息
    输入：任务名称
    返回：任务的设置信息
    :return
    the info of target task stored in NI-DAQmx(taskinfo: info dict)
    """
    taskinfo = dict()
    task = nidaqmx.system.storage.persisted_task.PersistedTask(taskname).load()
    taskinfo['sample_rate'] = int(task.timing.samp_clk_rate.real)
    taskinfo['sample_mode'] = task.timing.samp_quant_samp_mode.name
    taskinfo['chan_names'] = task.channel_names
    for i in range(len(task.channel_names)):
        taskinfo[task.ai_channels[i].name] = dict()
        taskinfo[task.ai_channels[i].name]['chanType'] = task.ai_channels[i].ai_meas_type.name
        taskinfo[task.ai_channels[i].name]['couplingMode'] = task.ai_channels[i].ai_coupling
        taskinfo[task.ai_channels[i].name]['minVal'] = task.ai_channels[i].ai_min
        taskinfo[task.ai_channels[i].name]['maxVal'] = task.ai_channels[i].ai_max
    task.close()
    return taskinfo


def get_ni_devices():
    """
    功能：获取ni采集设备
    输入：无
    返回：设备列表
    :return
    device list of available NI devices(dev_list:device names)
    """
    system = nidaqmx.system.system.System()
    dev_list = system.devices.device_names
    return dev_list


def reset_ni_device(dev='Dev1'):
    """
    功能：重置指定的NI采集设备
    输入：设备名称，默认是Dev1
    返回：无
    """
    # device = nidaqmx.system.System.local().devices[dev]
    device = nidaqmx.system.system.System().devices[dev]
    device.reset_device()


def get_channels(dev='Dev1'):
    """
    功能：获取指定设备的通道名列表
    输入：设备名称
    返回：通道名称列表
    return: channel list of target NI device(chan_list:device names)
    """
    chan_list = nidaqmx.system._collections.physical_channel_collection.AIPhysicalChannelCollection(
        dev).channel_names
    return chan_list


def create_lin_scale(name, slope, y_intercept, pre_unit=nidaqmx.constants.UnitsPreScaled.VOLTS, scale_unit="Nm"):
    """
    功能：自定义换算标尺
    输入：
    1. 名称
    2. 斜率
    3. 截距
    4. 换算前单位，需要参考nidaqmx中的UnitsPreScaled，默认是伏特（Volts）
    5. 换算后单位，来源于采集任务的参数设置
    返回：标尺
    """
    # 自定义换算标尺
    scale = nidaqmx.scale.Scale(name)
    return scale.create_lin_scale(name, slope, y_intercept, pre_scaled_units=pre_unit, scaled_units=scale_unit)


# 参考振动值
ref_accel = 1e-6  # unit: m/s^2
# 参考声压
ref_sound_pressure = 20e-6  # unit: Pa


def task_info_update(task_info):
    """
    update the taskInfo based on raw information from config file
    1. add ni device info for physicalChannels
    2. 更新单位索引信息
    功能：完善数据采集信息，主要是结合本地配置文件（config.ini）完善采集设备信息
    返回：更新后的NI DAQmx数据采集任务配置信息
    """
    for i in range(len(task_info["physicalChannels"])):
        task_info["physicalChannels"][i] = "/".join([ni_device, task_info["physicalChannels"][i]])
    # 根据通道名称更新单位索引信息(目前只支持振动信号（Vib开头）和声压信号（Mic开头）),索引信息的数量最终会与基础信息里的传感器数量一致
    task_info['indicatorsUnitChanIndex'] = list()
    task_info['refValue'] = list()
    for channel_index, channel_name in enumerate(task_info['channelNames']):
        if channel_name.startswith('Vib'):
            task_info['indicatorsUnitChanIndex'].append(channel_index)
            task_info['refValue'].append(ref_accel / task_info['sensitivity'][channel_index])
        elif channel_name.startswith('Mic'):
            task_info['indicatorsUnitChanIndex'].append(channel_index)
            task_info['refValue'].append(ref_sound_pressure)
    # 设置线性换算参数
    if "lineScale" not in task_info.keys():
        # 如果不存在的话
        task_info["lineScale"] = dict()
        task_info["lineScale"]["flag"] = list()
        task_info["lineScale"]["slope"] = list()
        task_info["lineScale"]["intercept"] = list()
        task_info["lineScale"]["scaleUnits"] = list()
        for unit in task_info["units"]:
            task_info["lineScale"]["scaleUnits"].append(unit)
            task_info["lineScale"]["flag"].append(0)
            task_info["lineScale"]["slope"].append(1)
            task_info["lineScale"]["intercept"].append(0)
    # 更新通道单位
    for i, unit in enumerate(task_info["lineScale"]["scaleUnits"]):
        if task_info["lineScale"]["flag"][i]:
            # 如果需要换算则将通道的单位更新（根据换算后的单位）
            task_info["units"][i] = unit
    return task_info

class DAQTask:
    """
    功能：NI数据采集任务的类
    输入：数据采集任务设置信息
    1. setup the task (sample rate, sample mode, samp_per_channel...)
    2. task operation (start, stop, close, save)
    """

    def __init__(self, taskInfo):
        # 初始化，输入数据采集任务参数信息
        self.taskinfo = taskInfo
        self.task = None

    def createtask(self):
        """
        功能：新建一个数据采集任务
        :based on the settings to create new task, and update task
        """
        # 新建一个空任务
        self.task = nidaqmx.Task()
        for i in range(len(self.taskinfo['physicalChannels'])):
            # 根据通道列表依次增加并设置该采集任务
            if self.taskinfo['channelType'][i] == 'Accel':
                # 采集加速度信号
                if self.taskinfo['lineScale']['flag'][i]:
                    # 需要进行单位换算
                    s_accel = create_lin_scale('acc', self.taskinfo['lineScale']['slope'][i],
                                               self.taskinfo['lineScale']['intercept'][i],
                                               pre_unit=nidaqmx.constants.UnitsPreScaled.G,
                                               scale_unit=self.taskinfo['lineScale']['scaleUnits'][i])
                    self.task.ai_channels.add_ai_accel_chan(self.taskinfo['physicalChannels'][i],
                                                            self.taskinfo['channelNames'][i],
                                                            min_val=self.taskinfo['minVal'][i],
                                                            max_val=self.taskinfo['maxVal'][i],
                                                            units=nidaqmx.constants.AccelUnits.FROM_CUSTOM_SCALE,
                                                            sensitivity=self.taskinfo['sensitivity'][i],
                                                            sensitivity_units=nidaqmx.constants.AccelSensitivityUnits.M_VOLTS_PER_G,
                                                            current_excit_val=self.taskinfo['current_excit_val'][i],
                                                            custom_scale_name=s_accel.name)
                else:
                    # 不需要进行换算
                    # to convert the acceleration units
                    # 设置单位（根据设置参数转换为驱动可以识别的单位）
                    if self.taskinfo['units'][i] == 'm/s^2':
                        # Meters per second per second
                        accel_units = nidaqmx.constants.AccelUnits.METERS_PER_SECOND_SQUARED
                    elif self.taskinfo['units'][i] == 'in/s^2':
                        # Inches per second per second
                        accel_units = nidaqmx.constants.AccelUnits.INCHES_PER_SECOND_SQUARED
                    else:
                        accel_units = nidaqmx.constants.AccelUnits.G  # 1g is approximately equal to 9.81 m/s^2.
                    # 设置物理通道，通道名，最大最小值，灵敏度，激励电流值等参数配置
                    # 注意usb板卡的激励电流值只有2.1ma，pci的板卡可以有4ma，只要不超过就没有问题
                    self.task.ai_channels.add_ai_accel_chan(self.taskinfo['physicalChannels'][i],
                                                            self.taskinfo['channelNames'][i],
                                                            min_val=self.taskinfo['minVal'][i],
                                                            max_val=self.taskinfo['maxVal'][i], units=accel_units,
                                                            sensitivity=self.taskinfo['sensitivity'][i],
                                                            sensitivity_units
                                                            =nidaqmx.constants.AccelSensitivityUnits.M_VOLTS_PER_G,
                                                            current_excit_val=self.taskinfo['current_excit_val'][i])
                # 设置耦合方式（这里是AC耦合）
                self.task.ai_channels[i].ai_coupling = nidaqmx.constants.Coupling.AC
            elif self.taskinfo['channelType'][i] == 'Sound':
                # 采集声音信号（采集声压）
                if self.taskinfo['lineScale']['flag'][i]:
                    # 需要进行换算
                    s_sound = create_lin_scale('sound', self.taskinfo['lineScale']['slope'][i],
                                               self.taskinfo['lineScale']['intercept'][i],
                                               pre_unit=nidaqmx.constants.UnitsPreScaled.PA,
                                               scale_unit=self.taskinfo['lineScale']['scaleUnits'][i])
                    self.task.ai_channels.add_ai_microphone_chan(self.taskinfo['physicalChannels'][i],
                                                                 self.taskinfo['channelNames'][i],
                                                                 max_snd_press_level=self.taskinfo['maxVal'][i],
                                                                 units=nidaqmx.constants.SoundPressureUnits.FROM_CUSTOM_SCALE,
                                                                 mic_sensitivity=self.taskinfo['sensitivity'][i],
                                                                 current_excit_val=self.taskinfo['current_excit_val'][
                                                                     i], custom_scale_name=s_sound.name)
                else:
                    # 不需要进行换算
                    # 设置物理通道，通道名，最大最小值，灵敏度，激励电流值等参数配置
                    self.task.ai_channels.add_ai_microphone_chan(self.taskinfo['physicalChannels'][i],
                                                                 self.taskinfo['channelNames'][i],
                                                                 max_snd_press_level=self.taskinfo['maxVal'][i],
                                                                 units=nidaqmx.constants.SoundPressureUnits.PA,
                                                                 mic_sensitivity=self.taskinfo['sensitivity'][i],
                                                                 current_excit_val=self.taskinfo['current_excit_val'][i])
                # 设置耦合方式（这里是AC耦合）
                self.task.ai_channels[i].ai_coupling = nidaqmx.constants.Coupling.AC
            else:
                # 目前其他的一概认为是采集电压
                # default as voltage
                if self.taskinfo['lineScale']['flag'][i]:
                    # 需要进行换算则根据参数设置线性换算
                    s_volt = create_lin_scale('volt', self.taskinfo['lineScale']['slope'][i],
                                              self.taskinfo['lineScale']['intercept'][i],
                                              scale_unit=self.taskinfo['lineScale']['scaleUnits'][i])
                    # 设置物理通道，通道名，最大最小值等参数配置
                    self.task.ai_channels.add_ai_voltage_chan(self.taskinfo['physicalChannels'][i],
                                                              self.taskinfo['channelNames'][i],
                                                              min_val=self.taskinfo['minVal'][i],
                                                              max_val=self.taskinfo['maxVal'][i],
                                                              units=nidaqmx.constants.TorqueUnits.FROM_CUSTOM_SCALE,
                                                              custom_scale_name=s_volt.name)
                else:
                    # 设置物理通道，通道名，最大最小值等参数配置
                    self.task.ai_channels.add_ai_voltage_chan(self.taskinfo['physicalChannels'][i],
                                                              self.taskinfo['channelNames'][i],
                                                              min_val=self.taskinfo['minVal'][i],
                                                              max_val=self.taskinfo['maxVal'][i])
                # 设置耦合方式（电压信号是DC耦合）
                self.task.ai_channels[i].ai_coupling = nidaqmx.constants.Coupling.DC
        # 设置采样率（根据参数信息），采集方式为连续采样，待读取采样数（根据参数信息）
        # set the sample mode of NI DAQmx task
        self.task.timing.cfg_samp_clk_timing(self.taskinfo['sampleRate'],
                                             sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                                             samps_per_chan=self.taskinfo['sampsPerChan'])
        # 设置缓冲区大小
        # samps_per_chan is to make the buffer
        self.task.in_stream.input_buf_size = self.taskinfo['bufferSize']

    def start_task(self):
        # 开始采集任务
        self.task.start()

    def read_data(self):
        # 读取数据（可同时返回多个通道的数据，每个通道都是帧长）
        return self.task.read(self.taskinfo['sampsPerChan'])

    def stop_task(self):
        # 停止采集任务
        self.task.stop()

    def clear_task(self):
        # 清空采集任务
        self.task.close()

    def save_task(self):
        # 保存采集任务，这样就在NI MAX的任务列表就能看到
        del self.task.in_stream.input_buf_size  # NI-DAQmx task can not support the setting of input buffer size
        self.task.save(save_as="qdaq".encode(), author="Synovate".encode(), overwrite_existing_task=True)


if __name__ == '__main__':
    # 下面的代码为该模块测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(pathname)s %(funcName)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='[%Y-%m-%d %H:%M:%S]',
        filename='D:/qdaq/Log/temp.log',
        filemode='a'
    )
    try:

        task_info_before={
        "bufferSize": 1000000,
        "channelNames": [
            "Speed",
            "Vib1",
            "Vib2"
        ],
        "channelType": [
            "Volt",
            "Accel",
            "Accel"
        ],
        "current_excit_val": [
            1.0,
            0.0021,
            0.0021
        ],
        "maxVal": [
            10,
            500,
            500
        ],
        "minVal": [
            -10,
            -500,
            -500
        ],
        "physicalChannels": [
            "ai0",
            "ai1",
            "ai2"
        ],
        "sampleRate": 102400,
        "sampsPerChan": 16384,
        "sensitivity": [
            1.0,
            9.85,
            9.85
        ],
        "units": [
            "Volts",
            "g",
            "g"
        ],
		"coeff": [
			1.0,
			1.0
		],
		"timeout": 200,
		"indicatorsUnitChanIndex": 1
    }

        import os
        import time
        import numpy as np
        import matplotlib.pyplot as plt
        type_info = "Test210615"

        ni_device='Dev2'

        # test get_daq_tasks
        print(get_daq_tasks())

        # test get_taskinfo(taskname)
        # print(get_taskinfo('torque'))

        # test get_NI_Devices
        print(get_ni_devices())

        # test get_channels(dev='Dev1')
        print(get_channels(dev='Dev2'))

        # test create scale
        s_test = create_lin_scale('test', 10, 0)
        print(type(s_test))
        print(s_test.lin_slope, s_test.lin_y_intercept)

        # test data acquisition by NI-DAQmx task
        taskInfo=task_info_update(task_info_before)

        try:
            dtask = DAQTask(taskInfo)
            dtask.createtask()
            dtask.start_task()
        except nidaqmx.errors.DaqError:
            logging.info('Daq error')
            dtask.stop_task()
            dtask.clear_task()
            reset_ni_device(ni_device)
            dtask = DAQTask(taskInfo)
            dtask.createtask()
            dtask.start_task()

        for i in range(2):
            data = dtask.read_data()
            print(len(data), len(data[0]))
            print(data[0][0], data[1][0])
            print(np.max(data, axis=1))
            print(np.min(data, axis=1))
        dtask.save_task()
        dtask.stop_task()
        dtask.clear_task()
        reset_ni_device(ni_device)
        plt.plot(data[3])
        plt.show()
        # 从ni max中导入保存的采集任务并开始
        # task = nidaqmx.system.storage.persisted_task.PersistedTask('torque').load()
        # task.start()
        # for i in range(10):
        #     data = task.read(8192)
        #     print(max(data[0]))
        # task.stop()
        # task.close()
    except Exception:
        traceback.print_exc()
        logging.warning("exec failed, failed msg:" + traceback.format_exc())
        sys.exit()
