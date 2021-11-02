#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2020/9/24 23:42
@Author : Wall@Synovate
@Email  : qxu@sonustc.com
@Function: real time speed recognition and NVH calculation
"""

import atexit
import numpy as np
from uldaq import (get_daq_device_inventory, DaqDevice, AInScanFlag,
                   ScanOption, ScanStatus, create_float_buffer,
                   InterfaceType, AiInputMode, IepeMode, CouplingMode)
import traceback
import sys
import os


def get_daq_devices(interface_type=InterfaceType.USB):
    """
    function: get the DT devices available
    input:
    1. interface type, default as USB
    output:
    1. devices list
    """
    # Get descriptors for all of the available DAQ devices.
    device_list = get_daq_device_inventory(interface_type)

    number_of_device = len(device_list)
    if number_of_device > 0:
        return device_list
    else:
        raise Exception('Error: No DAQ devices found')


def device_confirm(daq_device):
    if daq_device:
        if daq_device.is_connected():
            daq_device.disconnect()
            daq_device.release()
            daq_device.reset()
        else:
            daq_device.connect(connection_code=0)


def device_reset(daq_device):
    if daq_device:
        if daq_device.is_connected():
            daq_device.disconnect()
        daq_device.release()


def get_ai_device_info(device):
    ai_device = device.get_ai_device()
    if ai_device is None:
        raise Exception('Error: The DAQ device does not support analog input')
    else:
        ai_info = ai_device.get_info()
        return ai_device, ai_info


def ai_info_confirm(ai_info):
    if not ai_info.has_pacer():
        raise Exception('\nError: The specified DAQ device does not support hardware paced analog input')

    # Verify the device supports IEPE
    if not ai_info.supports_iepe():
        raise RuntimeError('Error: The DAQ device does not support IEPE')


def get_channels_by_mode(ai_info, input_mode=AiInputMode.SINGLE_ENDED):
    channels = ai_info.get_num_chans_by_mode(input_mode)
    return channels


def channel_confirm(high_channel_index, low_channel_index, number_of_channels):
    # take big one as high channel and small one as low channel
    high_channel, low_channel = max(high_channel_index, low_channel_index), min(high_channel_index, low_channel_index)
    if high_channel >= number_of_channels:
        high_channel = number_of_channels - 1
    channel_count = high_channel - low_channel + 1
    return high_channel, low_channel, channel_count


def get_ranges(ai_info, range_index=0, input_mode=AiInputMode.SINGLE_ENDED):
    # Get a list of supported ranges and validate the range index.
    ranges = ai_info.get_ranges(input_mode)
    if range_index >= len(ranges):
        range_index = len(ranges) - 1
    return ranges, range_index


def set_ai_config(ai_device, task_info, iepe_mode=IepeMode.ENABLED):
    # Set IEPE mode, AC coupling and sensor sensitivity for each channel
    ai_config = ai_device.get_config()
    for chan, chan_type in enumerate(task_info['channelType']):
        if chan_type == 'Volt':
            coupling = CouplingMode.DC
            ai_config.set_chan_coupling_mode(chan, coupling)

        elif chan_type == 'Accel':
            coupling = CouplingMode.AC
            ai_config.set_chan_iepe_mode(chan, iepe_mode)
            ai_config.set_chan_coupling_mode(chan, coupling)
            # NI板卡的灵敏度单位为mV/g，DT板卡的灵敏度单位为V/g
            ai_config.set_chan_sensor_sensitivity(chan, task_info['sensitivity'][chan]/1000)

        elif chan_type == 'Sound':
            coupling = CouplingMode.AC
            ai_config.set_chan_iepe_mode(chan, iepe_mode)
            ai_config.set_chan_coupling_mode(chan, coupling)
            # NI板卡的灵敏度单位为mV/Pa，DT板卡的灵敏度单位为V/Pa
            ai_config.set_chan_sensor_sensitivity(chan, task_info['sensitivity'][chan]/1000)

        # 是否进行线性转化
        if task_info['lineScale']['flag'][chan]:
            # 设置偏置
            ai_config.set_chan_offset(chan, task_info['lineScale']['intercept'][chan])
            # 设置斜率
            ai_config.set_chan_slope(chan, task_info['lineScale']['slope'][chan])


def buffer_creator(task_info):
    channel_num = len(task_info['channelNames'])
    samples_per_chan = task_info['sampsPerChan']
    buffer_factor_value = int(round(task_info['bufferSize'] // channel_num // samples_per_chan))
    data_per_frame = channel_num * samples_per_chan
    data_buffer = create_float_buffer(channel_num, samples_per_chan * buffer_factor_value)
    return buffer_factor_value, data_per_frame, data_per_frame * buffer_factor_value, data_buffer


def set_daq_task(task_info, ai_info, ai_device, buffer_factor_value, buffer_data, target_mode_channels, target_input_mode):
    samples_per_chan = task_info['sampsPerChan']
    high_channel = len(task_info['channelNames']) - 1
    low_channel = 0
    high_channel, low_channel, channel_count = channel_confirm(high_channel, low_channel, target_mode_channels)
    ranges, range_index = get_ranges(ai_info)
    set_ai_config(ai_device, task_info)
    ai_device.a_in_scan(low_channel, high_channel, target_input_mode, ranges[range_index],
               samples_per_chan * buffer_factor_value, task_info['sampleRate'],
               ScanOption.CONTINUOUS, AInScanFlag.DEFAULT, buffer_data)
    return ai_device


@atexit.register
def dt_disconnect():
    global status
    global target_device
    global target_ai_device

    print('At exit function')
    if target_device:
        # Stop the acquisition if it is still running.
        if status == ScanStatus.RUNNING:
            target_ai_device.scan_stop()

        if target_device.is_connected():
            target_device.disconnect()
        target_device.release()
        print('At exit DT9837B disconnected')


if __name__ == '__main__':
    try:
        import json
        import time
        from utils import create_properities, write_hdf5

        devices = get_daq_devices()
        target_device = DaqDevice(devices[0])
        device_confirm(target_device)
        target_ai_device, target_ai_info = get_ai_device_info(target_device)
        ai_info_confirm(target_ai_info)

        target_input_mode = AiInputMode.SINGLE_ENDED
        # If SINGLE_ENDED input mode is not supported, set to DIFFERENTIAL.
        target_mode_channels = get_channels_by_mode(target_ai_info, target_input_mode)
        if target_mode_channels <= 0:
            target_input_mode = AiInputMode.DIFFERENTIAL
        target_mode_channels = get_channels_by_mode(target_ai_info, target_input_mode)

        param_filename = r'config.json'
        with open(param_filename, 'r') as f:
            param = json.load(f)

        buffer_data_extender = dict()
        for chan_name in param['channelNames']:
            buffer_data_extender[chan_name] = list()
        channel_count = len(param['channelNames'])
        frame_counter = 0
        data_reading_counter = 0
        buffer_counter = 0
        s_time = time.time()
        buffer_factor_value, frame_data_len, buffer_len, buffer_data = buffer_creator(param)
        target_ai_device = set_daq_task(param, target_ai_info, target_ai_device, buffer_factor_value, buffer_data, target_mode_channels, target_input_mode)
        while True:
            status, transfer_status = target_ai_device.get_scan_status()
            next_data_length = (frame_counter + 1) * frame_data_len
            frame_num = (transfer_status.current_total_count - frame_counter * frame_data_len) // frame_data_len
            print("frame num:", frame_num, frame_counter)
            for j in range(frame_num):
                print("frame ref:", j)
                # enough data to read out
                if transfer_status.current_total_count - frame_counter * frame_data_len <= buffer_len:
                    # no over write
                    rest_part = next_data_length % buffer_len
                    if rest_part < frame_data_len:
                        # take part of data from start and end
                        temp_data = np.concatenate((buffer_data[0:rest_part], buffer_data[-(frame_data_len - rest_part):]))
                        for i in range(channel_count):
                            buffer_data_extender[param['channelNames'][i]].extend(temp_data[i:: channel_count])
                        buffer_counter += 1
                    else:
                        # do not need to take data from start and end
                        start_index = (frame_counter * frame_data_len) % buffer_len
                        end_index = start_index + frame_data_len
                        for i in range(channel_count):
                            buffer_data_extender[param['channelNames'][i]].extend(
                                buffer_data[start_index + i: end_index: channel_count])
                    frame_counter += 1
                else:

                    raise ValueError("data overflow")
            data_reading_counter += 1

            index = transfer_status.current_index

            if time.time() - s_time > 10:
                break
            if data_reading_counter == 0:
                print('initialIndex = ', index, '\n')

            print('currentTotalCount = ',
                  transfer_status.current_total_count)
            print('currentScanCount = ',
                  transfer_status.current_scan_count)
            print('currentIndex = ', index, '\n')

            data_reading_counter += 1
            time.sleep(0.1)
        print(buffer_factor_value, frame_data_len, buffer_len)
        print(data_reading_counter, frame_counter, buffer_counter, buffer_len)
        print(len(buffer_data_extender[param['channelNames'][0]]))
        # save data
        # 保存原始数据
        raw_data_save_properties = create_properities("2021 09 11 12:00:12", 1 / param['sampleRate'], param['sampsPerChan'])
        raw_data_filename = 'test.h5'
        if os.path.exists(raw_data_filename):
            os.remove(raw_data_filename)
        for i in range(channel_count):
            print(len(buffer_data_extender[param['channelNames'][i]]))
            raw_data_save_properties['NI_ChannelName'] = param['channelNames'][i]
            raw_data_save_properties['NI_UnitDescription'] = param['units'][i]
            write_hdf5(raw_data_filename, 'AIData',
                       param['channelNames'][i],
                       buffer_data_extender[param['channelNames'][i]],
                       raw_data_save_properties)
    except Exception:
        traceback.print_exc()
        os.system("Pause")
        sys.exit()


