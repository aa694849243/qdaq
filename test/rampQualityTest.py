#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2021/7/2 14:39
@Author : Wall@Synovate
@Email  : qxu@sonustc.com
@Version: 1.0.0
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from qdaq.speed_tools import ramp_quality
from qdaq.parameters import Parameters, config_folder, config_file


def create_speed_curve(start_time, end_time, start_speed, end_speed, points):
    # 构造转速曲线
    speed_curve = dict()
    speed_curve['time'] = np.linspace(start_time, end_time, points)
    if start_speed == end_speed:
        speed_curve['speed'] = np.ones(points) * start_speed
    else:
        speed_curve['speed'] = np.linspace(start_speed, end_speed, points)
    return speed_curve


if __name__ == '__main__':
    # 测试转速质量的正确性
    type_info = "Test210702-1"
    # 该配置包含三种测试段，包括恒速段，升速段和降速段，会依次进行验证
    config_filename = os.path.join(config_folder, "_".join([type_info, config_file]))
    print(config_filename)
    if os.path.exists(config_filename):
        # 如果配置参数存在则继续执行
        param = Parameters(config_filename)
        print(param.speedRecogInfo)

        # 生成转速的设置
        speed_index = 2
        start_time, end_time = 0, 34
        start_speed, end_speed = 910, 80
        speed_points = 1000
        speed_curve = create_speed_curve(start_time, end_time, start_speed, end_speed, speed_points)

        # 比较加噪声前和加噪声后的爬坡质量

        # 生成指定长度的随机数（相当于噪声）
        noise = np.random.randint(0, 10, speed_points)
        noise_speed_curve_y = speed_curve['speed'] + noise
        print("noise ramp quality:",
              ramp_quality(speed_curve['time'], noise_speed_curve_y, param.speedRecogInfo, speed_index))
        plt.figure()
        plt.plot(speed_curve['time'], noise_speed_curve_y)
        speed_curve = create_speed_curve(0, 26, start_speed, end_speed, speed_points)
        print("raw ramp quality:",
              ramp_quality(speed_curve['time'], speed_curve['speed'], param.speedRecogInfo, speed_index))
        plt.plot(speed_curve['time'], speed_curve['speed'])
        plt.xlabel("Time/s")
        plt.ylabel("Speed/rpm")
        plt.legend(["real", "target"])

        plt.show()
