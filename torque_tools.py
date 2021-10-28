#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2021/6/21 9:29
@Author : Wall@Synovate
@Email  : qxu@sonustc.com
@Version: 1.0.0
功能说明：扭矩信号只用于辅助
"""


from common_info import qDAQ_logger
import numpy as np


def find_torque_in_range(torque_data, lower_value, upper_value):
    """
    功能：确认并寻找是否存在界限值内的点，以确定结束点
    输入：
    1. 扭矩数据
    2. 扭矩范围下限值
    3. 扭矩范围上限值
    返回：扭矩识别结果，开始点索引
    """
    if type(torque_data) != np.ndarray:
        # 确认是否为array，若不是则强制转换，主要为了能直接使用numpy的函数进行比较
        torque_data = np.array(torque_data)
    return np.where(((torque_data >= lower_value) & (torque_data <= upper_value)))[0]


def find_torque_out_range(torque_data, lower_value, upper_value):
    """
    功能：确认并寻找是否存在界限值外的点，以确定结束点
    输入：
    1. 扭矩数据
    2. 扭矩范围下限值
    3. 扭矩范围上限值
    返回：扭矩识别结果，开始点索引
    """
    if type(torque_data) != np.ndarray:
        # 确认是否为array，若不是则强制转换，主要为了能直接使用numpy的函数进行比较
        torque_data = np.array(torque_data)
    return np.where(((torque_data < lower_value) | (torque_data > upper_value)))[0]


def speed_detect_start_with_torque(present_speed_loc, present_speed, speed, present_torque, recog_index, gv_dict,
                                   speed_recog_info, sample_rate, counter, frame):
    """
    功能：重新定义一个新的方法，通过转速和扭矩信号来识别开始点，注意：主要转速值在上下界内（即下限值<=转速值<=上限值），扭矩值在上下限内
    （最小值<=扭矩值<=最大值），目前主要应用于协助恒速段的识别
    输入：
    1. 当前帧转速X轴
    2. 当前帧转速Y轴
    3. 累积转速值
    4. 当前帧的扭矩值
    5. 转速识别的索引（第几个测试段）
    5. 工况识别状态信息
    6. 工况识别参数信息
    7. 采样率
    返回：转速识别状态
    """
    # 恒速段识别（一般扭矩识别主要用于该类型测试段）
    # constant pattern(default range is 100), update start point detection
    # 恒速段可以不用探测到进入点即可确认开始点，只要在范围内且在规定时间不跳出去即可
    # lower为转速下界限，小于该值意味着超出边界，重新开始识别，取的是开始转速和结束转速的最小值
    lower = min(speed_recog_info['startSpeed'][recog_index], speed_recog_info['endSpeed'][recog_index])
    # upper为转速上界限，大于该值意味着超出边界，重新开始识别，取的是开始转速和结束转速的最大值
    upper = max(speed_recog_info['startSpeed'][recog_index], speed_recog_info['endSpeed'][recog_index])
    if not gv_dict['tempStartFlag']:
        # 如果未检测到开始点，则开始检测
        for i in range(len(present_speed)):
            if lower <= present_speed[i] <= upper:
                # 逐点确认是否在范围内，一旦识别到在范围内则跳出进入确认逻辑里面（即tempStartFlag=True）
                gv_dict['startpoint_loc'] = present_speed_loc[i]
                gv_dict['startpoint_speed'] = present_speed[i]
                gv_dict['startpoint_index'] = len(speed) - len(present_speed) + i
                gv_dict['tempStartFlag'] = True
                break
    if gv_dict['tempStartFlag']:
        # 如果检测到开始，需要根据波动时间进行确认，包括同一帧和不同帧的情况
        if len(speed) > len(present_speed) > 0:
            # 为了确保帧内第一个点可以判断，需要确认帧内转速曲线的长度，有数据后才去确认第一个点是否超出上下界
            if (present_speed[0] <= upper < speed[len(speed)-len(present_speed)-1]) or \
                    (present_speed[0] >= lower > speed[len(speed)-len(present_speed)-1]):
                # 如果该数据点重新进入上下界则更新开始点信息
                gv_dict['startpoint_loc'] = present_speed_loc[0]
                gv_dict['startpoint_speed'] = present_speed[0]
                gv_dict['startpoint_index'] = len(speed) - len(present_speed)
            elif (present_speed[0] > upper >= speed[len(speed)-len(present_speed)-1]) or \
                    (present_speed[0] <= lower < speed[len(speed)-len(present_speed)-1]):
                # 如果跳出上下界则清空开始点信息（重置）
                gv_dict['startpoint_loc'] = None
                gv_dict['startpoint_speed'] = None
                gv_dict['startpoint_index'] = None
            else:
                if not gv_dict['startFlag']:
                    # to confirm if no point out of range in fluctuationT
                    # 确认帧内的第一个点是否为开始点（主要考虑可能要往后推几帧才能确认开始点）
                    if gv_dict['startpoint_loc']:
                        if lower <= present_speed[0] <= upper:
                            # 确保当前点的转速在上下限内
                            if present_speed_loc[0] - gv_dict['startpoint_loc'] >= \
                                    speed_recog_info['fluctuationT'][recog_index]:
                                # 开始基于扭矩判断
                                torque_recog_result = find_torque_in_range(present_torque,
                                                                           speed_recog_info['minTorque'][
                                                                               recog_index],
                                                                           speed_recog_info['maxTorque'][
                                                                               recog_index])
                                if len(torque_recog_result):
                                    # 若扭矩识别结果不为空
                                    if round(present_speed_loc[0] * sample_rate - counter * frame) in torque_recog_result:
                                        # 如果转速开始点所对应的扭矩在范围内
                                        # 确认数据是否满足波动时长，满足则将startFlag标定为True，恒速段需要切掉波动时长内的数据
                                        gv_dict['startpoint_loc'] = present_speed_loc[0]
                                        gv_dict['startpoint_speed'] = present_speed[0]
                                        gv_dict['startpoint_index'] = len(speed) - len(present_speed)
                                        qDAQ_logger.info(speed_recog_info['testName'][recog_index] + ", index: " + str(
                                            recog_index) + " start point detected")
                                        gv_dict['startFlag'] = True
        # 只要开始点没有确认，则会持续确认直到可以确认了开始点
        if not gv_dict['startFlag']:
            for i in range(1, len(present_speed)):
                # 第一个点已经确认过了，从第二个点开始判断
                if (present_speed[i] > upper >= present_speed[i - 1]) or \
                        (present_speed[i] < lower <= present_speed[i - 1]):
                    # 如果跳出上下限则重置开始点的信息（清空原来记录的信息）
                    gv_dict['startpoint_loc'] = None
                    gv_dict['startpoint_speed'] = None
                    gv_dict['startpoint_index'] = None
                elif (present_speed[i] <= upper < present_speed[i - 1]) or \
                        (present_speed[i] >= lower > present_speed[i - 1]):
                    # 如果重新进入上下限（符合条件），则更新开始点信息（重新记录）
                    gv_dict['startpoint_loc'] = present_speed_loc[i]
                    gv_dict['startpoint_speed'] = present_speed[i]
                    gv_dict['startpoint_index'] = len(speed) - len(present_speed) + i

                if gv_dict['startpoint_loc']:
                    # 如果有开始点信息则进行确认，是否波动时长内均符合条件，一旦确认是开始点则跳出循环
                    if lower <= present_speed[i] <= upper:
                        # 保证转速仍然在范围内，再进行判断
                        if present_speed_loc[i] - gv_dict['startpoint_loc'] >= \
                                speed_recog_info['fluctuationT'][recog_index]:
                            # 开始基于扭矩判断
                            torque_recog_result = find_torque_in_range(present_torque,
                                                                       speed_recog_info['minTorque'][recog_index],
                                                                       speed_recog_info['maxTorque'][recog_index])
                            if len(torque_recog_result):
                                # 若扭矩识别结果不为空
                                if round(present_speed_loc[i] * sample_rate - counter * frame) in torque_recog_result:
                                    # 如果转速开始点所对应的扭矩在范围内
                                    # 确认是否满足时长，则以满足时长的该点置为开始点（恒速段需要切掉波动时长内的数据）
                                    gv_dict['startpoint_loc'] = present_speed_loc[i]
                                    gv_dict['startpoint_speed'] = present_speed[i]
                                    gv_dict['startpoint_index'] = len(speed) - len(present_speed) + i
                                    # 确认开始点，将标志置为帧并记录到日志中
                                    gv_dict['startFlag'] = True
                                    qDAQ_logger.info(speed_recog_info['testName'][recog_index] + ", index: " + str(
                                        recog_index) + " start point detected")
                                    break
    return gv_dict


def speed_detect_end_with_torque(present_speed, speed_loc, speed, overall_torque, recog_index, gv_dict,
                                 speed_recog_info, sample_rate):
    """
    功能：在转速识别的基础上结合扭矩识别并确认目标测试段的结束点
    输入：
    1. 当前帧转速值
    2. 累积转速曲线的X轴
    3. 累积转速曲线的Y值
    4. 目标测试段的索引
    5. 识别状态
    6. 转速识别参数信息
    返回：识别状态
    """
    if gv_dict['firstinFlag']:
        # 如果是第一次进入目标测试段，确定开始点的位置（因为又可能往后跳了几帧才确定开始点）根据开始点的索引往后开始判断
        # for the first time should compare the point form detected start point to the end of overall speed
        starti = gv_dict['startpoint_index'] + 1
        # 一旦判断过了第一段则之后的数据均不用再往前回溯，直接判断帧内的数据点即可
        gv_dict['firstinFlag'] = False
    else:
        # 还是基于整个转速曲线来确定索引
        starti = len(speed) - len(present_speed)
    # 确定结束点索引
    endi = len(speed)
    # 确定开始点的转速XY值，之后需要通过开始点的转速值进行判断
    startx = speed_loc[gv_dict['startpoint_index']]
    starty = speed[gv_dict['startpoint_index']]
    # 以上部分适用于所有转速段，以下部分是恒速，升速，降速分开的
    # 确认扭矩识别结果
    if endi > starti:
        target_torque = overall_torque[round(speed_loc[starti] * sample_rate):]
        torque_recog_end_result = find_torque_out_range(target_torque, speed_recog_info['minTorque'][recog_index],
                                                        speed_recog_info['maxTorque'][recog_index])
        if len(torque_recog_end_result):
            torque_recog_end_result += round(speed_loc[starti] * sample_rate)

    # 恒速段结束点判断
    # constant pattern(default range is 100)
    # 恒速段上下限值
    lower = min(speed_recog_info['startSpeed'][recog_index], speed_recog_info['endSpeed'][recog_index])
    upper = max(speed_recog_info['startSpeed'][recog_index], speed_recog_info['endSpeed'][recog_index])
    for i in range(starti, endi):
        # 遍历转速点
        if speed[i] < lower:
            # 如果超出下限，继续判断是否符合允许的异常点限制
            if gv_dict['exceptionsCounter'] < speed_recog_info['permittedExceptions'][recog_index]:
                # 如果仍在允许异常点限制范围内则异常点个数+1
                gv_dict['exceptionsCounter'] = gv_dict['exceptionsCounter'] + 1
            else:
                # 如果不在允许异常点限制范围内了则记录该点的转速信息并跳出循环停止转速识别
                qDAQ_logger.info("Speed Recognition Error1: under the lower limit, test name: " +
                             speed_recog_info['testName'][recog_index] + ", index: " + str(recog_index))
                # 记录当前点转速值和当前的转速识别下限
                qDAQ_logger.info("present speed:" + str(speed[i]) + ", lower limit:" + str(lower))
                # 停止转速识别并跳出（并不会停止转速计算）
                gv_dict['speedRecogFinish'] = True  # stop the speed recognition and continue speed calculation
                break
        else:
            # 如果没超出下限则是否上限条件
            if speed[i] > upper:
                # 如果超出上限，继续判断是否符合允许的异常点限制
                if gv_dict['exceptionsCounter'] < speed_recog_info['permittedExceptions'][recog_index]:
                    # 如果仍在允许异常点限制范围内则异常点个数+1
                    gv_dict['exceptionsCounter'] = gv_dict['exceptionsCounter'] + 1
                else:
                    # 如果不在允许异常点限制范围内了则记录该点的转速信息并跳出循环停止转速识别
                    qDAQ_logger.info("Speed Recognition Error2: over the upper limit, test name: " +
                                 speed_recog_info['testName'][recog_index] + ", index: " + str(recog_index))
                    # 记录当前点转速值和当前的转速识别上限
                    qDAQ_logger.info("present speed:" + str(speed[i]) + ", upper limit:" + str(upper))
                    # 停止转速识别并跳出（并不会停止转速计算）
                    gv_dict['speedRecogFinish'] = True  # stop the speed recognition and continue speed calculation
                    break
            else:
                # 上下限均符合条件则确认是否满足期望时长要求或者扭矩
                if speed_loc[i] - startx >= speed_recog_info['maxT'][recog_index]:
                    # 记录结束点的信息
                    gv_dict['endpoint_loc'] = speed_loc[i]
                    gv_dict['endpoint_speed'] = speed[i]
                    gv_dict['endpoint_index'] = i
                    # 重置标志位用于下一个测试段识别，并记录该信息以判断转速识别已完成
                    gv_dict['firstinFlag'] = True
                    qDAQ_logger.info(speed_recog_info['testName'][recog_index] + ", index: " + str(recog_index) +
                                 " end point detected")
                    break
                elif speed_loc[i] - startx >= speed_recog_info['minT'][recog_index]:
                    # 一旦确认已满足最小时长则开始确认扭矩是否满足要求
                    if len(torque_recog_end_result):
                        if round(speed_loc[i]*sample_rate) in torque_recog_end_result:
                            # 记录结束点的信息
                            gv_dict['endpoint_loc'] = speed_loc[i]
                            gv_dict['endpoint_speed'] = speed[i]
                            gv_dict['endpoint_index'] = i
                            # 重置标志位用于下一个测试段识别，并记录该信息以判断转速识别已完成
                            gv_dict['firstinFlag'] = True
                            qDAQ_logger.info(speed_recog_info['testName'][recog_index] + ", index: " + str(recog_index) +
                                         " end point detected")
                            break
    return gv_dict


if __name__ == '__main__':
    # 测试该模块
    import logging
    import traceback
    import os
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(pathname)s %(funcName)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='[%Y-%m-%d %H:%M:%S]',
        filename='D:/qdaq/Log/temp.log',
        filemode='a'
    )
    try:
        from parameters import Parameters, config_folder, config_file
        from auxiliary_tools import read_tdms, read_hdf5
        from utils import write_json
        from global_var import set_default_speedRecog
        from speed_tools import trigger_detect, rpm_calc, speed_detect_start, speed_detect_end, ramp_quality
        import matplotlib.pyplot as plt
        import time

        type_info = "AP4000_05170"
        config_filename = os.path.join(config_folder, "_".join([type_info, config_file]))
        print(config_filename)
        if os.path.exists(config_filename):
            param = Parameters(config_filename)

            # 读取原始数据
            # # 数据1：配置文件为Test210627_paramReceived.json
            # tdms_filename = r'D:\qdaq\Simu\aiwayTest\20210514\TEST0515-9.tdms'
            # torque_data, pp_torque = read_tdms(tdms_filename, 'AIData', 'Torque')
            # time_array = np.arange(len(torque_data)) * pp_torque['wf_increment']
            # speed_data, pp = read_tdms(tdms_filename, 'AIData', 'Speed')
            #
            # # 读取转速曲线
            # speed_curve_filename = r'D:\qdaq\Simu\aiwayTest\20210514\speedCurve.h5'
            # speed_curve_x, _ = read_hdf5(speed_curve_filename, 'speedData', 'speedLoc')
            # speed_curve_y, _ = read_hdf5(speed_curve_filename, 'speedData', 'speedValue')

            # # 数据2,3,4：配置文件为Test210628_paramReceived.json，为2021年5月28日给爱驰远程恢复的数据
            # hdf5_filename = r'D:\Wall_Work\3_Project\308_Customer\Aiways\debug210528\TZ220XS004M20210001M021427003_210428012112\TZ220XS004M20210001M021427003_210428012112.tdms'
            # torque_data, pp_torque = read_hdf5(hdf5_filename, 'AIData', 'Torque')
            # pp_torque['wf_increment'] = 1/51200  # 数据写入时写错了采样率和每帧长度
            # pp_torque['wf_samples'] = 8192
            # time_array = np.arange(len(torque_data)) * pp_torque['wf_increment']
            # speed_data, pp = read_hdf5(hdf5_filename, 'AIData', 'Speed')
            # pp['wf_increment'] = 1 / 51200  # 数据写入时写错了采样率和每帧长度
            # pp['wf_samples'] = 8192
            #
            # # 读取转速曲线
            # speed_curve_filename = r'D:\Wall_Work\3_Project\308_Customer\Aiways\debug210528\TZ220XS004M20210001M021427003_210428012112\speedCurve.h5'
            # speed_curve_x, _ = read_hdf5(speed_curve_filename, 'speedData', 'speedLoc')
            # speed_curve_y, _ = read_hdf5(speed_curve_filename, 'speedData', 'speedValue')

            # # 数据5：配置文件为Test210628_paramReceived.json
            # hdf5_filename = r'D:\Wall_Work\3_Project\308_Customer\Aiways\debug210517\TZ220XS004M20210001M021514002_210517005116.h5'
            # torque_data, pp_torque = read_hdf5(hdf5_filename, 'AIData', 'Torque')
            # time_array = np.arange(len(torque_data)) * pp_torque['wf_increment']
            # speed_data, pp = read_hdf5(hdf5_filename, 'AIData', 'Speed')

            # # 数据6：配置文件为Test210628_paramReceived.json
            # tdms_filename = r'D:\dataBackup\Customer\Aiways\NewTestprofile_20210421_AP4000_1_13.066\Data\12333_210421083712.tdms'
            # torque_data, pp_torque = read_tdms(tdms_filename, 'AIData', 'Torque')
            # time_array = np.arange(len(torque_data)) * pp_torque['wf_increment']
            # speed_data, pp = read_tdms(tdms_filename, 'AIData', 'Speed')

            # 数据7：配置文件为AP4000_05170_paramReceived.json
            tdms_filename = r'D:\dataBackup\Customer\Aiways\AP4000\Data\TZ220XS004M20210001M021512002_210518025321.tdms'
            torque_data, pp_torque = read_tdms(tdms_filename, 'AIData', 'Torque')
            time_array = np.arange(len(torque_data)) * pp_torque['wf_increment']
            speed_data, pp = read_tdms(tdms_filename, 'AIData', 'Speed')

            # # 数据8：配置文件为Ino-TTL-TypeX_paramReceived.json
            # tdms_filename = r'D:\dataBackup\Customer\Inovance\TTL\TypeX\Data\20191217testn3018_191218090347.tdms'
            # torque_data, pp_torque = read_tdms(tdms_filename, 'AIData', 'Torque')
            # time_array = np.arange(len(torque_data)) * pp_torque['wf_increment']
            # speed_data, pp = read_tdms(tdms_filename, 'AIData', 'Speed')

            # # 数据9：配置文件为Ino-TTL-TypeY_paramReceived.json
            # tdms_filename = r'D:\dataBackup\Customer\Inovance\TTL\TypeY\Data\simu.tdms'
            # torque_data, pp_torque = read_tdms(tdms_filename, 'AIData', 'Torque')
            # time_array = np.arange(len(torque_data)) * pp_torque['wf_increment']
            # speed_data, pp = read_tdms(tdms_filename, 'AIData', 'Speed')

            print(pp)
            # 获取采样率
            sampleRate = round(1 / pp['wf_increment'])
            # 获取帧长
            frame = pp['wf_samples']
            # 计算帧数
            frame_num = len(speed_data) // frame
            # 初始化参数
            # 计数器
            counter = 0
            icounter = 0
            # 工况识别索引
            recog_index = 0
            # 脉冲位置信息
            triggerLocation = list()
            # 工况识别结果
            recog_result = list()
            # 初始化转速曲线
            rpml = list()
            rpm = list()
            # 初始化转速识别标志
            gv_dict_speedRecog = set_default_speedRecog()

            # 初始化扭矩信号
            torque_list = list()

            t1 = time.time()
            for counter_i in range(frame_num):
                # print(counter, gv_dict_speedRecog)
                # 每个工况识别完需要往前倒退一帧
                if counter == icounter:
                    # 计算转速
                    if counter == 0:
                        lfp = speed_data[0]
                    # speed calculation
                    temptl, lfp = trigger_detect(speed_data[counter * frame: (counter + 1) * frame], lfp,
                                                 param.speedCalcInfo)
                    temptl = list(np.array(temptl) + counter * frame)
                    triggerLocation.extend(temptl)
                    temp_rpml, temp_rpm = rpm_calc(temptl, triggerLocation, sampleRate, param.speedCalcInfo)
                    rpml.extend(temp_rpml)
                    rpm.extend(temp_rpm)
                    torque_frame = list(torque_data[counter * frame: (counter + 1) * frame])
                    torque_list.extend(torque_frame)
                else:
                    icounter = counter
                if not gv_dict_speedRecog['speedRecogFinish']:
                    # speed recognition
                    if not gv_dict_speedRecog['startFlag']:
                        # detect the start point of target operating mode
                        if param.speedRecogInfo["torqueRecogFlag"][recog_index]:
                            # 如果需要扭矩识别
                            gv_dict_speedRecog = speed_detect_start_with_torque(temp_rpml, temp_rpm, rpm, torque_frame,
                                                                                recog_index,
                                                                                gv_dict_speedRecog,
                                                                                param.speedRecogInfo, sampleRate, counter, frame)
                        else:
                            # 不需要扭矩识别
                            gv_dict_speedRecog = speed_detect_start(temp_rpml, temp_rpm, rpm, recog_index,
                                                                    gv_dict_speedRecog, param.speedRecogInfo)
                        # special case: start point and end point in same frame
                        if gv_dict_speedRecog['startFlag']:
                            # 第一帧识别到了开始点
                            if param.speedRecogInfo["torqueRecogFlag"][recog_index]:
                                gv_dict_speedRecog = speed_detect_end_with_torque(temp_rpm, rpml, rpm, torque_list,
                                                                      recog_index, gv_dict_speedRecog,
                                                                      param.speedRecogInfo, sampleRate)
                            else:
                                gv_dict_speedRecog = speed_detect_end(temp_rpm, rpml, rpm,
                                                                  recog_index, gv_dict_speedRecog, param.speedRecogInfo)
                            if gv_dict_speedRecog['endpoint_loc']:
                                # 识别到结束点
                                temp_speed_x = np.array(
                                    rpml[gv_dict_speedRecog['startpoint_index']: gv_dict_speedRecog['endpoint_index']])
                                temp_speed_y = np.array(
                                    rpm[gv_dict_speedRecog['startpoint_index']: gv_dict_speedRecog['endpoint_index']])
                                # 转速曲线添加噪声
                                # temp_speed_y += np.random.randint(90, 100, len(temp_speed_y))
                                # recog index is for test name list of speed recognition
                                recog_result.append({'testName': param.speedRecogInfo['testName'][recog_index],
                                                     'startTime': gv_dict_speedRecog['startpoint_loc'],
                                                     'endTime': gv_dict_speedRecog['endpoint_loc'],
                                                     'RampQuality': ramp_quality(temp_speed_x, temp_speed_y,
                                                                                 param.speedRecogInfo, recog_index)})
                                recog_index = recog_index + 1
                                if recog_index < len(param.speedRecogInfo['testName']):
                                    # set back the parameters for speed recognition if test not finished
                                    gv_dict_speedRecog = set_default_speedRecog()
                                    counter = counter - 1
                                else:
                                    gv_dict_speedRecog['speedRecogFinish'] = True
                            else:
                                # 未识别到结束点
                                pass
                        else:
                            # 第一帧为识别到开始点
                            pass
                    else:
                        # 如果识别到了开始点
                        # detect the start point of target operating mode(not the first frame)
                        if param.speedRecogInfo["torqueRecogFlag"][recog_index]:
                            gv_dict_speedRecog = speed_detect_end_with_torque(temp_rpm, rpml, rpm, torque_list,
                                                                              recog_index, gv_dict_speedRecog,
                                                                              param.speedRecogInfo, sampleRate)
                        else:
                            gv_dict_speedRecog = speed_detect_end(temp_rpm, rpml, rpm,
                                                                  recog_index, gv_dict_speedRecog, param.speedRecogInfo)
                        if gv_dict_speedRecog['endpoint_loc']:
                            temp_speed_x = np.array(
                                rpml[gv_dict_speedRecog['startpoint_index']: gv_dict_speedRecog['endpoint_index']])
                            temp_speed_y = np.array(
                                rpm[gv_dict_speedRecog['startpoint_index']: gv_dict_speedRecog['endpoint_index']])
                            # 转速曲线添加噪声
                            # temp_speed_y += np.random.randint(90, 100, len(temp_speed_y))
                            # recog index is for test name list of speed recognition
                            recog_result.append({'testName': param.speedRecogInfo['testName'][recog_index],
                                                 'startTime': gv_dict_speedRecog['startpoint_loc'],
                                                 'endTime': gv_dict_speedRecog['endpoint_loc'],
                                                 'RampQuality': ramp_quality(temp_speed_x, temp_speed_y,
                                                                             param.speedRecogInfo, recog_index)})
                            recog_index = recog_index + 1
                            if recog_index < len(param.speedRecogInfo['testName']):
                                gv_dict_speedRecog = set_default_speedRecog()
                                counter = counter - 1
                            else:
                                gv_dict_speedRecog['speedRecogFinish'] = True
                        else:
                            # 未识别到结束点
                            pass
                else:
                    # 未停止识别
                    pass
                counter = counter + 1
                icounter = icounter + 1
            print(time.time() - t1)
            # 将识别结果标注在转速曲线上
            print(recog_result)
            recog_result_filename = r'D:\qdaq\temp\TEST0515-9.json'
            write_json(recog_result_filename, recog_result)
            speed_loc, speed_value = np.array(rpml), np.array(rpm)
            # 画转速识别结果
            fig1, ax1 = plt.subplots(1, 1, figsize=(12, 7))
            ax1.plot(speed_loc, speed_value, '-k*', label='Raw Speed Curve')
            ax1.set_title('Speed Recognition result with Torque')
            ax1.set_xlabel('Time/s')
            ax1.set_ylabel('Speed/rpm')
            for result in recog_result:
                condition = (speed_loc >= result['startTime']) & (speed_loc <= result['endTime'])
                ax1.plot(speed_loc[condition], speed_value[condition], label=result['testName'])
                # ax1.axvline(result['startTime'], color='g')
                # ax1.axvline(result['endTime'], color='r')
            ax1.legend(loc=2)
            # 画扭矩识别结果
            ax2 = ax1.twinx()
            ax2.plot(time_array[::100], torque_data[::100], 'y', label='Torque Curve')
            ax2.set_ylabel('Torque/Nm')
            ax2.legend(loc=1)
            # plt.figure()
            # plt.plot(time_array, torque_data)
            # plt.plot(speed_curve_x, speed_curve_y)
            plt.show()
    except Exception:
        traceback.print_exc()
        logging.warning("exec failed, failed msg:" + traceback.format_exc())
        sys.exit()