import collections
import ctypes
import datetime
import logging
import math
import multiprocessing
import platform
import queue
import sys
import threading
import time
import traceback
import os
from utils import write_tdms

import numpy as np

from parameters import Parameters, ni_device
from common_info import config_file, config_folder
from collections import deque
import nidaqmx
from nptdms import TdmsWriter, ChannelObject
import queue
from DAQTask import DAQTask, reset_ni_device

switch = True
startflag = False
endflag = False
raw_data = collections.deque()
q = collections.deque()


def cal_sensity(p, array) -> float:
    """
    p: 声压
    array: 传入的数组
    """
    return math.sqrt(sum([array[i]**2 for i in range(len(array))]) / len(array))/p


def cal_task(channel, ampl=114, sample_freq=102400):
    """
    db:标定器分贝数
    time:取时间范围内的平均值
    sample_freq:采样频率
    """
    global switch,startflag
    startflag=True
    p = 2 * 10 ** (ampl / 20 - 5)
    type_info = "MicTest"
    config_filename = os.path.join(config_folder, "_".join([type_info, config_file]))
    cnt = 0

    if os.path.exists(config_filename):
        param = Parameters(config_filename)
        try:
            dtask = DAQTask(param.taskInfo)
            dtask.createtask()
            dtask.start_task()
        except nidaqmx.errors.DaqError:
            logging.info('Daq error')
            dtask.stop_task()
            dtask.clear_task()
            reset_ni_device(ni_device)
            dtask = DAQTask(param.taskInfo)
            dtask.createtask()
            dtask.start_task()
        arr = []
        leng = sample_freq * 2  # 2秒钟的长度
        print('start')
        while switch and cnt <= 750:
            cnt += 1
            data = np.array(dtask.read_data())
            for val in data:
                raw_data.append(val)
                arr.append(val)
                if len(arr) >= leng:
                    q.append('{:.2f}'.format((cal_sensity(p, arr))))
                    arr.clear()
                if len(raw_data) > 200:
                    # write_tdms(r'D:\qdaq\Data\MicTest\caojie.tdms', 'AIdata', 'sound', raw_data)
                    raw_data.popleft()
        print('end')
        print(raw_data)
        dtask.save_task()
        dtask.stop_task()
        dtask.clear_task()
        reset_ni_device(ni_device)
    print('结束')


def end_task():
    global switch, startflag
    switch = False
    startflag = False
    return raw_data


# def start(access, channel, ampl, sample_freq=102400):
#     if access == 'Mic':
#         channel = ['ai0', 'ai1', 'ai2', 'ai3'].index(channel)
#         cal_task(channel, ampl, sample_freq)
#         end_task()
#         return raw_data


def acquire():
    if q:
        return q.popleft()
    else:
        return -1


if __name__ == '__main__':
    t=threading.Thread(target=cal_task,args=(0,))
    t.start()
    while True:
        time.sleep(0.5)
        print(q)