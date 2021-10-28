#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2021/7/21 17:09
@Author : Wall@Synovate
@Email  : qxu@sonustc.com
@Version: 1.0.0
"""
import h5py
from nptdms import TdmsFile
import os

with h5py.File('test.h5', 'a') as h5file:
    group_list = [h5file[key].name.split('/')[1] for key in h5file.keys()]
    print(group_list)
    if len(group_list):
        channel_list = list(h5file[group_list[0]].keys())
        print(group_list, channel_list)

test_hdf5_filename = r'D:\dataBackup\Customer\Aiways\Type1\Data\TZ220XS004M20210001M021427001_210427045306.h5'
with h5py.File(test_hdf5_filename, 'r') as h5file:
    group_list = [h5file[key].name.split('/')[1] for key in h5file.keys()]
    if len(group_list):
        channel_list = list(h5file[group_list[0]].keys())
        print(group_list, channel_list)

if os.path.exists('test.tdms'):
    with TdmsFile.open('test.tdms') as tdms_file:
        group_list = [group.name for group in tdms_file.groups()]
        print(group_list)
        if len(group_list):
            channels_name_list = [channel.name for channel in
                                  tdms_file[group_list[0]].channels()]
            print(group_list, channels_name_list)

test_tdms_filename = r'D:\dataBackup\Customer\Aiways\Type2\Data\SN0459.tdms'
with TdmsFile.open(test_tdms_filename) as tdms_file:
    group_list = [group.name for group in tdms_file.groups()]
    if len(group_list):
        channels_name_list = [channel.name for channel in tdms_file[group_list[-1]].channels()]
        print(group_list, channels_name_list)