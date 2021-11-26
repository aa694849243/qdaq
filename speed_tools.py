# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:14:44 2020

update: 2021-06-03

@author: Wall@Sonus

speed processing module(speed calculation, speed recognition, and resolver signal processing)
说明：该模块提供了转速计算及工况识别相关函数方法
"""

import traceback
import numpy as np
import os
import sys
import time
from scipy import stats, fftpack,signal
from scipy.interpolate import interp1d
from common_info import qDAQ_logger
from indicator_tools import butter_filter


def trigger_detect(present_speed_pulse, last_frame_point, speed_calc_info):
    """
    功能：识别脉冲，主要用于探测转速脉冲信号的脉冲位置并记录脉冲位置信息，之后用于计算转速和角度域重采样
    输入：
    1. 当前帧的转速脉冲信号
    2. 上一帧的最后一个点
    3. 转速计算参数信息（包括脉冲触发电平大小，上升沿还是下降沿触发等信息）
    返回：
    1. 脉冲位置信息
    2. 当前帧的最后一个点
    function:to detect the trigger(this function only need about half of the time by the trigger_detect)
    update: 2020/7/27, to speed up, change into numpy and list.extend
    :param
    present_speed_pulse(list): the raw speed signal in pulse format, in list format
    last_frame_point(float): the last point of last frame data, initial value is 0,
    this point to decide if the first point of present frame is trigger
    speed_calc_info(dict): parameters for speed calculation, for this function need these two info below:
        triggerLevel(float): default as 2, the point cross this trigger_level set as teh trigger location
        triggerMode(string): default as 'Rising', 'Rising' to detect the rising edge, 'Falling' to detect
        the trailing edge
    :returns
    trigger_location(list): the trigger location of this frame 当前帧的索引
    frame_last_point(float): return the last point of present frame for next frame
    """
    trigger_location = list()
    # 寻找脉冲电平大于触发电平的位置并赋值为1求导，这样等于1的位置即为上升沿触发位置，等于-1的即为下降沿位置
    triggermarker = np.diff(1 * (np.array(present_speed_pulse) >= speed_calc_info['triggerLevel']))
    if speed_calc_info['triggerMode'] == 'Rising':
        # 上升沿检测
        if last_frame_point < speed_calc_info['triggerLevel'] <= present_speed_pulse[0]:
            # 判断第一个点是否为脉冲触发点
            trigger_location.append(0)
        # 判断帧内的数据点
        trigger_location.extend(list(np.where(triggermarker == 1)[0] + 1))
    elif speed_calc_info['triggerMode'] == 'Falling':
        # 下降沿检测
        if last_frame_point >= speed_calc_info['triggerLevel'] > present_speed_pulse[0]:
            # 判断第一个点是否为脉冲触发点
            trigger_location.append(0)
        # 判断帧内的数据点
        trigger_location.extend(list(np.where(triggermarker == -1)[0] + 1))
    return trigger_location, present_speed_pulse[-1]


def trigger_detect_for_share(overall_speed, start_index, end_index, speed_calc_info):
    """

    Args:
        overall_speed: 储存整个测试的速度点
        start_index: 对每一帧数据均进行转速计算，为该帧数据speed的起始位置
        end_index: 为该帧数据speed的终止位置，overall_speed[end_index]处尚未赋值
        speed_calc_info: 转速计算信息

    Returns:
        上升沿/或下降沿所在的位置，以start_index为0索引位置
    """
    if start_index <= 0:
        left = 0
    else:
        # 不是第一帧则向左取一个点
        left = start_index - 1
    triggermaker = np.diff(1 * (overall_speed[left:end_index] >= speed_calc_info['triggerLevel']))
    if start_index != 0:
        triggerLocation = np.where(triggermaker == 1)[0]
    else:
        triggerLocation = np.where(triggermaker == 1)[0] + 1
    return triggerLocation


def rpm_calc(present_trigger_location, overall_trigger, fs, speed_calc_info):
    """
    功能：基于脉冲位置信息计算转速，得到转速曲线
    输入：
    1. 当前帧的脉冲位置信息
    2. 累积脉冲位置信息
    3. 采样率
    4. 转速计算参数信息，主要包括ppr，平均点数和重叠点数等信息
    返回：
    1. 转速曲线X轴
    2. 转速曲线Y轴
    function: to get the RPM(revolutions per minute) Value based on the trigger location
    :param:
    present_trigger_location(list): the trigger location record calculated by the speed pulse 数据是里面上升沿的索引
    overall_trigger(list):the overall trigger location list
    fs(int): with this info to transfer point index to time
    speed_calc_info(dict): include info below for this function:
        ppr(int): pulses per revolution, int type
        averageNum(int): the number of pulses to do average, default as 1(means each trigger will calculate 1 rpm value)
        overlap(int): the moving step, should less than average_num(e.g average_num=1, overlap must be 0), default as 0
    :returns:
    location(list): the x for rpm curve, this based on the trigger location  上升沿所在的时刻，从检测开始开始计时
    speed(list): the RPM value
    """
    # 根据脉冲到达时间计算转速（脉冲到达时间，采样率，多少个求平均，每转脉冲数，重叠数（例：3个求平均，重叠1个，也就是跳两个数）
    # 根据当前的trigger location的点数来决定要获取多长的历史点去做运算
    # RPM(转/分):60*Fs/(点数间隔/平均次数)/ppr
    speed_loc = list()
    speed_value = list()
    # 例如：10个值求平均，重叠数为5，则表示每5个脉冲得到一个转速值，采样率变成原来的1/5
    if len(overall_trigger) > speed_calc_info['averageNum']:
        loc_factor = 1 / fs
        val_factor = speed_calc_info['rpmFactor'] * fs
        for i in range(len(present_trigger_location)):
            # 确认
            if i + len(overall_trigger) - len(present_trigger_location) == speed_calc_info['averageNum']:
                # 计算均值（根据平均点数得设置）
                speed_loc.append(overall_trigger[speed_calc_info['averageNum']] * loc_factor)
                speed_value.append(
                    val_factor / (overall_trigger[speed_calc_info['averageNum']] - overall_trigger[0]))
            elif (i + len(overall_trigger) - len(present_trigger_location) - speed_calc_info[
                'averageNum']) > 0:
                if (i + len(overall_trigger) - len(present_trigger_location) - speed_calc_info[
                    'averageNum']) % \
                        speed_calc_info['resampleFactor'] == 0:
                    speed_loc.append(present_trigger_location[i] * loc_factor)
                    speed_value.append(val_factor / (present_trigger_location[i] -
                                                     (overall_trigger[i + len(overall_trigger) -
                                                                      len(present_trigger_location) -
                                                                      speed_calc_info['averageNum']])))
    return speed_loc, speed_value


def rpm_calc_for_share(trigger_array, start, end, sampleRate, average, step, rpmFactor,
                               rpml_array, rpm_array,
                               rpm_index, is_first_calc):
    """
    byd专版转速计算，由于转速信号的切换，所以不能用start==0来判断是否是第一次计算
    Args:
        trigger_array: 上升沿/下降沿位置
        start: 该帧计算出了第7，8，9，10，11，12（从0计数）个trigger,start必定小于7，可以为6，可以为5，
        因为上一帧计算到的最后的trigger不确定，特例是start为0
        end: 该帧计算出了第7，8，9，10，11，12（从0计数）个trigger,则end传递进来的是12
        sampleRate: 采样率
        speed_calc_info: 转速计算信息
        rpml_array: 整个测试的转速曲线数组 时间
        rpm_array: 整个测试的转速曲线数组  转速值
        rpm_index: 下一个rpm点要写入的索引

    Returns:

    """

    # 由于转速信号的切换，所以不能用start==0来判断是否是第一次计算
    # if start == 0:
    #     # 第一次算
    #     if end < average:
    #         # 算不出来
    #         return start, rpm_index
    #     else:
    #         # 能算出值
    #         start = average - step

    if is_first_calc:
        if (end - start) < average:
            # 算不出来
            return start, rpm_index, True
        else:
            # 能算出来
            start += average - step

    # 本次计算能计算出来多少个
    len = (end - start) // step
    for i in range(len):
        # 中间是speed_calc_info["averageNum"]个间隔
        delta_t = (trigger_array[start + step] - trigger_array[start + step - average]) / sampleRate
        rpm_array[rpm_index] = rpmFactor / delta_t
        # 取中间时刻作为speed_time
        # rpml_array[rpm_index]=(trigger_array[start+step]+
        #          trigger_array[start+step-average])/2/sampleRate
        # 取averager最后一个上升沿为时刻点,将来乘以采样率得到的值取int即为上升沿/下降沿的位置
        rpml_array[rpm_index] = (trigger_array[start + step]) / sampleRate
        rpm_index += 1
        start = start + step
    return start, rpm_index, False


def speed_detect_start(present_speed_loc, present_speed, speed, recog_index, gv_dict, speed_recog_info):
    """
    功能：通过转速识别开始点，注意：主要转速值在上下界内（即下限值<=转速值<=上限值）
    输入：
    1. 当前帧转速X轴
    2. 当前帧转速Y轴
    3. 累积转速值
    4. 转速识别的索引（第几个测试段）
    5. 转速识别状态信息
    6. 转速识别参数信息
    返回：转速识别状态
    function: just to detect the start point of the target speed period, once detect this start point, start to cal
                the parameter(RMS, Crest...)
    :param
    present_speed_loc(list):as the data is read in by frame, this is the speed location record of present frame
    present_speed(list):same as the present speed location, this is the value of speed(match the location)
    speed(list):the overall speed value
    recog_index(int): index of speed_recog_info, target speed pattern
    gv_dict(dict): global value a dict to store the information need to recode the recognition result
    speed_recog_info(dict):
        start_speed(float): start speed for target test conditions
        end_speed(float): end speed for target test conditions
        fluctuationT(float): this is to avoid the fluctuate point
    :returns(just update into global variable):
    gv_dict(dict): update the speed calculation status
        startflag(bool): flag to show if already detect the point pass through the target speed, init value is False
        startpoint_loc(int): ini value is None, once detected, update
        startpoint_speed(float): speed value match the location of startpoint
        startpoint_index(int): the startpoint index in the overall speed array
    """

    if speed_recog_info['speedPattern'][recog_index] == 1:
        # 恒速段识别
        # constant pattern(default range is 100), update start point detection
        # 恒速段可以不用探测到进入点即可确认开始点，只要在范围内且在规定时间不跳出去即可
        # lower为转速下界限，小于该值意味着超出边界，重新开始识别，取的是开始转速和结束转速的最小值
        lower = min(speed_recog_info['startSpeed'][recog_index],
                    speed_recog_info['endSpeed'][recog_index])
        # upper为转速上界限，大于该值意味着超出边界，重新开始识别，取的是开始转速和结束转速的最大值
        upper = max(speed_recog_info['startSpeed'][recog_index],
                    speed_recog_info['endSpeed'][recog_index])
        if not gv_dict['tempStartFlag']:
            # 如果未检测到开始点，则开始检测
            for i in range(len(present_speed)):
                if lower <= present_speed[0] <= upper:
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
                if (present_speed[0] <= upper < speed[len(speed) - len(present_speed) - 1]) or \
                        (present_speed[0] >= lower > speed[len(speed) - len(present_speed) - 1]):
                    # 如果该数据点重新进入上下界则更新开始点信息
                    gv_dict['startpoint_loc'] = present_speed_loc[0]
                    gv_dict['startpoint_speed'] = present_speed[0]
                    gv_dict['startpoint_index'] = len(speed) - len(present_speed)
                elif (present_speed[0] > upper >= speed[len(speed) - len(present_speed) - 1]) or \
                        (present_speed[0] <= lower < speed[len(speed) - len(present_speed) - 1]):
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
                                    # 确认数据是否满足波动时长，满足则将startFlag标定为True，恒速段需要切掉波动时长内的数据
                                    gv_dict['startpoint_loc'] = present_speed_loc[0]
                                    gv_dict['startpoint_speed'] = present_speed[0]
                                    gv_dict['startpoint_index'] = len(speed) - len(present_speed)
                                    qDAQ_logger.info(
                                        speed_recog_info['testName'][recog_index] + ", index: " + str(
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
                                # 确认是否满足时长，则以满足时长的该点置为开始点（恒速段需要切掉波动时长内的数据）
                                gv_dict['startpoint_loc'] = present_speed_loc[i]
                                gv_dict['startpoint_speed'] = present_speed[i]
                                gv_dict['startpoint_index'] = len(speed) - len(present_speed) + i
                                # 确认开始点，将标志置为帧并记录到日志中
                                gv_dict['startFlag'] = True
                                qDAQ_logger.info(
                                    speed_recog_info['testName'][recog_index] + ", index: " + str(
                                        recog_index) + " start point detected")
                                break

    elif speed_recog_info['speedPattern'][recog_index] == 2:
        # 识别升速段的开始点
        # detect the start point of drive pattern
        if len(speed) > len(present_speed) > 0:
            # 确认第一个点是否为开始点
            if present_speed[0] >= speed_recog_info['startSpeed'][recog_index] > speed[
                len(speed) - len(present_speed) - 1]:
                # 如果第一个点越过开始转速（即该测试段起始转速小于等于当前点转速且大于前一个点的转速值），则更新开始点信息
                gv_dict['startpoint_loc'] = present_speed_loc[0]
                gv_dict['startpoint_speed'] = present_speed[0]
                gv_dict['startpoint_index'] = len(speed) - len(present_speed)
            elif present_speed[0] < speed_recog_info['startSpeed'][recog_index] <= speed[
                len(speed) - len(present_speed) - 1]:
                # 如果跳出起始转速（即该测试段起始转速大于当前点转速且小于等于上一个转速点的值），则重置开始点信息（即清空）
                gv_dict['startpoint_loc'] = None
                gv_dict['startpoint_speed'] = None
                gv_dict['startpoint_index'] = None
            else:
                if not gv_dict['startFlag']:
                    # 确定是否满足波动时长的条件（及已确认的开始点指定时长内未出现任何点掉出限制范围）
                    # to confirm if no point out of range in fluctuationT
                    if gv_dict['startpoint_loc']:
                        if (speed_recog_info['startSpeed'][recog_index] <= present_speed[0] <=
                                speed_recog_info['endSpeed'][recog_index]):
                            # 确保该点速度值在上下限速度值内
                            if present_speed_loc[0] - gv_dict['startpoint_loc'] >= \
                                    speed_recog_info['fluctuationT'][recog_index]:
                                # 确保已满足波动时长的条件，并记录到日志中
                                qDAQ_logger.info(
                                    speed_recog_info['testName'][recog_index] + ", index: " + str(
                                        recog_index) + " start point detected")
                                # 将开始点标志设为真表示开始点已确认
                                gv_dict['startFlag'] = True
        # 只要未确认开始点则持续进行确认（主要针对帧内第二个转速点之后的数据点，第一个点刚才已经判断过了）
        if not gv_dict['startFlag']:
            for i in range(1, len(present_speed)):
                if present_speed[i] >= speed_recog_info['startSpeed'][recog_index] > present_speed[
                    i - 1]:
                    # 只要再次符合条件（即该测试段起始转速小于等于当前点转速且大于前一个点的转速值），则更新开始点信息
                    gv_dict['startpoint_loc'] = present_speed_loc[i]
                    gv_dict['startpoint_speed'] = present_speed[i]
                    gv_dict['startpoint_index'] = len(speed) - len(present_speed) + i
                elif present_speed[i] < speed_recog_info['startSpeed'][recog_index] <= present_speed[
                    i - 1]:
                    # 如果跳出起始转速（即该测试段起始转速大于当前点转速且小于等于上一个转速点的值），则重置开始点信息（即清空）
                    gv_dict['startpoint_loc'] = None
                    gv_dict['startpoint_speed'] = None
                    gv_dict['startpoint_index'] = None
                else:
                    # 确定是否满足波动时长的条件（及已确认的开始点指定时长内未出现任何点掉出限制范围）
                    # to confirm if no point out of range in fluctuationT
                    if gv_dict['startpoint_loc']:
                        # 如果开始点存在则进行确认
                        if (speed_recog_info['startSpeed'][recog_index] <= present_speed[i] <=
                                speed_recog_info['endSpeed'][recog_index]):
                            # 确认波动时长内的点仍在转速上下限内
                            if present_speed_loc[i] - gv_dict['startpoint_loc'] >= \
                                    speed_recog_info['fluctuationT'][recog_index]:
                                # 波动时长内无异常点，开始点识别标志置为帧并记录到日志中
                                gv_dict['startFlag'] = True
                                qDAQ_logger.info(
                                    speed_recog_info['testName'][recog_index] + ", index: " + str(
                                        recog_index) + " start point detected")
                                break

    elif speed_recog_info['speedPattern'][recog_index] == 3:
        # 降速段开始点识别
        # detect the start point of coast pattern(just like the drive pattern)
        if len(speed) > len(present_speed) > 0:
            # 确认帧内第一个点是否满足条件，是开始点，或者需要更新开始点信息
            if present_speed[0] <= speed_recog_info['startSpeed'][recog_index] < speed[
                len(speed) - len(present_speed) - 1]:
                # 只要再次符合条件（即该测试段起始转速大于等于当前点转速且小于前一个点的转速值），则更新开始点信息
                gv_dict['startpoint_loc'] = present_speed_loc[0]
                gv_dict['startpoint_speed'] = present_speed[0]
                gv_dict['startpoint_index'] = len(speed) - len(present_speed)
            elif present_speed[0] > speed_recog_info['startSpeed'][recog_index] >= \
                    speed[len(speed) - len(present_speed) - 1]:
                # 如果跳出起始转速（即该测试段起始转速小于当前点转速且大于等于上一个转速点的值），则重置开始点信息（即清空）
                gv_dict['startpoint_loc'] = None
                gv_dict['startpoint_speed'] = None
                gv_dict['startpoint_index'] = None
            else:
                if not gv_dict['startFlag']:
                    # 确认是否满足波动时长的条件
                    # to confirm if no point out of range in fluctuationT
                    if gv_dict['startpoint_loc']:
                        # 如果开始点存在则进行确认
                        if (speed_recog_info['endSpeed'][recog_index] <= present_speed[0] <=
                                speed_recog_info['startSpeed'][recog_index]):
                            # 确认该点在起始转速和终止转速之间
                            if present_speed_loc[0] - gv_dict['startpoint_loc'] >= \
                                    speed_recog_info['fluctuationT'][recog_index]:
                                # 波动时长内无异常点，开始点识别标志置为帧并记录到日志中
                                qDAQ_logger.info(
                                    speed_recog_info['testName'][recog_index] + ", index: " + str(
                                        recog_index) + " start point detected")
                                gv_dict['startFlag'] = True
        # 只要未确认开始点，将持续确认帧内其他点（从第二点开始）
        if not gv_dict['startFlag']:
            for i in range(1, len(present_speed)):
                if present_speed[i] <= speed_recog_info['startSpeed'][recog_index] < present_speed[
                    i - 1]:
                    # 只要再次符合条件（即该测试段起始转速大于等于当前点转速且小于前一个点的转速值），则更新开始点信息
                    gv_dict['startpoint_loc'] = present_speed_loc[i]
                    gv_dict['startpoint_speed'] = present_speed[i]
                    gv_dict['startpoint_index'] = len(speed) - len(present_speed) + i
                elif present_speed[i] > speed_recog_info['startSpeed'][recog_index] >= present_speed[
                    i - 1]:
                    # 如果跳出起始转速（即该测试段起始转速小于当前点转速且大于等于上一个转速点的值），则重置开始点信息（即清空）
                    gv_dict['startpoint_loc'] = None
                    gv_dict['startpoint_speed'] = None
                    gv_dict['startpoint_index'] = None
                else:
                    # 确认是否满足波动时长的条件
                    # to confirm if no point out of range in fluctuationT
                    if gv_dict['startpoint_loc']:
                        # 如果开始点存在则进行确认
                        if speed_recog_info['endSpeed'][recog_index] <= present_speed[i] <= \
                                speed_recog_info['startSpeed'][recog_index]:
                            # 确认该点在起始转速和终止转速之间
                            if present_speed_loc[i] - gv_dict['startpoint_loc'] >= \
                                    speed_recog_info['fluctuationT'][recog_index]:
                                # 波动时长内无异常点，开始点识别标志置为帧并记录到日志中，并跳出循环
                                gv_dict['startFlag'] = True
                                qDAQ_logger.info(
                                    speed_recog_info['testName'][recog_index] + ", index: " + str(
                                        recog_index) + " start point detected")
                                break
    else:
        qDAQ_logger.warning("can not confirm the speed pattern, please check the settings!")
        # 如果出现错误，如未定义的转速状态，则强制认为识别已完成，不再进行识别
        gv_dict['speedRecogFinish'] = True  # stop the speed recognition and continue speed calculation
    return gv_dict


def speed_detect_end(present_speed, speed_loc, speed, recog_index, gv_dict, speed_recog_info):
    """
    功能：在已识别并确认开始点的基础上，确认目标测试段的结束点
    输入：
    1. 当前帧转速值
    2. 累积转速曲线的X轴
    3. 累积转速曲线的Y值
    4. 目标测试段的索引
    5. 识别状态
    6. 转速识别参数信息
    返回：识别状态
    function: to detect the end point of target speed period, based on the start point, once the end point detected,
    show up the calculated results(RMS,Crest...)
    :param
    present_speed(list): the speed value info of present frame data
    speedLocation(list): the speed location for overall speed
    speed(list): overall speed info(include present frame)
    recog_index(int): index of speedRecogInfo, target speed pattern
    gv_dict(dict): global value a dict to store the information need to recode the recognition result
    speed_recog_info(dict):
        start_speed(float): start speed of target test condition
        end_speed(float): the upper speed limit of speed detect start and end point
        minT(float): the minimum time duration of the target speed pattern(only for drive and coast pattern)
        expectedT(float): the expected time duration of target speed pattern(for drive and coast is to cal the
                            limit, for constant is to decide the length of data
        maxT(float): just like the minT, this is the maximum time duration, will influence the limit when do
                    speed detection
        Permitted_exceptions(int): this is the permitted exceptions, default as zero, if not zero, it means when
                                        do detection, will skip some points out of range
    :returns(update to global variable):
    gv_dict:
        firstinFlag(bool): just retrun this flag as the input of next frame
        counter(int): just return this number to next frame to indicate how many points out if range
        fit_flag(bool): just return this fit flag to next frame
        fitted_end(float): just return this number to next frame
        endpoint_loc(int): ini value is None, once detected, update
        endpoint_speed(float): speed value match the location of endpoint
        endpoint_index(int): the endpoint index in the overall speed array
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
    if speed_recog_info['speedPattern'][recog_index] == 1:
        # 恒速段结束点判断
        # constant pattern(default range is 100)
        # 恒速段上下限值
        lower = min(speed_recog_info['startSpeed'][recog_index],
                    speed_recog_info['endSpeed'][recog_index])
        upper = max(speed_recog_info['startSpeed'][recog_index],
                    speed_recog_info['endSpeed'][recog_index])
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
                                     speed_recog_info['testName'][recog_index] + ", index: " + str(
                        recog_index))
                    # 记录当前点转速值和当前的转速识别下限
                    qDAQ_logger.info("present speed:" + str(speed[i]) + ", lower limit:" + str(lower))
                    # 停止转速识别并跳出（并不会停止转速计算）
                    gv_dict[
                        'speedRecogFinish'] = True  # stop the speed recognition and continue speed calculation
                    break
            else:
                # 如果没超出下限则是否上限条件
                if speed[i] > upper:
                    # 如果超出上限，继续判断是否符合允许的异常点限制
                    if gv_dict['exceptionsCounter'] < speed_recog_info['permittedExceptions'][
                        recog_index]:
                        # 如果仍在允许异常点限制范围内则异常点个数+1
                        gv_dict['exceptionsCounter'] = gv_dict['exceptionsCounter'] + 1
                    else:
                        # 如果不在允许异常点限制范围内了则记录该点的转速信息并跳出循环停止转速识别
                        qDAQ_logger.info("Speed Recognition Error2: over the upper limit, test name: " +
                                         speed_recog_info['testName'][recog_index] + ", index: " + str(
                            recog_index))
                        # 记录当前点转速值和当前的转速识别上限
                        qDAQ_logger.info(
                            "present speed:" + str(speed[i]) + ", upper limit:" + str(upper))
                        # 停止转速识别并跳出（并不会停止转速计算）
                        gv_dict[
                            'speedRecogFinish'] = True  # stop the speed recognition and continue speed calculation
                        break
                else:
                    # 上下限均符合条件则确认是否满足期望时长要求
                    if speed_loc[i] - startx >= speed_recog_info['expectT'][recog_index]:
                        # 一旦确认已满足期望的时长则确认恒速段已识别完成
                        # 记录结束点的信息
                        gv_dict['endpoint_loc'] = speed_loc[i]
                        gv_dict['endpoint_speed'] = speed[i]
                        gv_dict['endpoint_index'] = i
                        # 重置标志位用于下一个测试段识别，并记录该信息以判断转速识别已完成
                        gv_dict['firstinFlag'] = True
                        qDAQ_logger.info(
                            speed_recog_info['testName'][recog_index] + ", index: " + str(recog_index) +
                            " end point detected")
                        break
    elif speed_recog_info['speedPattern'][recog_index] == 2:
        # 升速段转速识别
        # drive pattern
        for i in range(starti, endi):
            # 遍历转速点
            # 计算下限值
            lower = (speed_loc[i] - startx) * speed_recog_info['slope'][recog_index] + starty + \
                    speed_recog_info['maxRange'][recog_index]
            if speed[i] < lower:
                # 如果超出下限，继续判断是否符合允许的异常点限制
                if gv_dict['exceptionsCounter'] < speed_recog_info['permittedExceptions'][recog_index]:
                    # 如果仍在允许异常点限制范围内则异常点个数+1
                    gv_dict['exceptionsCounter'] = gv_dict['exceptionsCounter'] + 1
                else:
                    # 如果不在允许异常点限制范围内了则记录该点的转速信息并跳出循环停止转速识别
                    qDAQ_logger.info("Speed Recognition Error1: under the lower limit, test name: " +
                                     speed_recog_info['testName'][recog_index] + ", index: " + str(
                        recog_index))
                    # 记录当前点转速值和当前的转速识别下限
                    qDAQ_logger.info('present speed:' + str(speed[i]) + ', lower limit:' + str(lower))
                    # 停止转速识别并跳出（并不会停止转速计算）
                    gv_dict[
                        'speedRecogFinish'] = True  # stop the speed recognition and continue speed calculation
                    break
            else:
                # 计算上限值
                upper = (speed_loc[i] - startx) * speed_recog_info['slope'][recog_index] + starty + \
                        speed_recog_info['minRange'][recog_index]
                if speed[i] > upper:
                    # 如果超出上限，继续判断是否符合允许的异常点限制
                    if gv_dict['exceptionsCounter'] < speed_recog_info['permittedExceptions'][
                        recog_index]:
                        # 如果仍在允许异常点限制范围内则异常点个数+1
                        gv_dict['exceptionsCounter'] = gv_dict['exceptionsCounter'] + 1
                    else:
                        # 如果不在允许异常点限制范围内了则记录该点的转速信息并跳出循环停止转速识别
                        qDAQ_logger.info("Speed Recognition Error2: over the upper limit, test name: " +
                                         speed_recog_info['testName'][recog_index] + ", index: " + str(
                            recog_index))
                        qDAQ_logger.info(
                            'present speed:' + str(speed[i]) + ', upper limit:' + str(upper))
                        # 停止转速识别并跳出（并不会停止转速计算）
                        gv_dict[
                            'speedRecogFinish'] = True  # stop the speed recognition and continue speed calculation
                        break
                else:
                    # 上下限均符合条件则开始拟合并判断结束点
                    if gv_dict['fitFlag']:
                        # 如果已经进行拟合（拟合点取自开始点到最短时长内的数据点），则开始判断是否满足结束点条件
                        if speed_loc[i] >= gv_dict['fitted_end']:
                            # 如果满足拟合出来的时长条件，则记录结束点信息
                            gv_dict['endpoint_loc'] = speed_loc[i]
                            gv_dict['endpoint_speed'] = speed[i]
                            gv_dict['endpoint_index'] = i
                            # 重置结束点识别标志信息
                            gv_dict['fitFlag'] = False
                            gv_dict['firstinFlag'] = True
                            # 记录结束点已识别完成
                            qDAQ_logger.info(
                                speed_recog_info['testName'][recog_index] + ", index: " + str(
                                    recog_index) + " end point detected")
                            break
                    else:
                        # 如果未进行拟合则判断是否符合拟合条件（需要满足最短时长）
                        if speed_loc[i] - startx >= speed_recog_info['minT'][recog_index]:
                            # 满足最短时长则进行拟合（利用scipy的stats的线性回归函数进行拟合），得到斜率、截距等信息
                            slope, intercept, r_value, p_value, std_err = stats.linregress(
                                speed_loc[gv_dict['startpoint_index']:i],
                                speed[gv_dict['startpoint_index']:i])
                            # 通过斜率截距等信息计算结束点位置
                            gv_dict['fitted_end'] = (speed_recog_info['endSpeed'][
                                                         recog_index] - intercept) / slope
                            gv_dict['fitFlag'] = True
                            # 判断是否已经满足结束点条件
                            if speed_loc[i] >= gv_dict['fitted_end']:
                                # 如果满足拟合出来的时长条件，则记录结束点信息
                                gv_dict['endpoint_loc'] = speed_loc[i]
                                gv_dict['endpoint_speed'] = speed[i]
                                gv_dict['endpoint_index'] = i
                                # 重置结束点识别标志信息
                                gv_dict['fitFlag'] = False
                                gv_dict['firstinFlag'] = True
                                # 记录结束点已识别完成
                                qDAQ_logger.info(
                                    speed_recog_info['testName'][recog_index] + ", index: " + str(
                                        recog_index) + " end point detected")
                                break

    elif speed_recog_info['speedPattern'][recog_index] == 3:
        # 降速段识别
        # coast pattern
        for i in range(starti, endi):
            # 计算下限值
            lower = (speed_loc[i] - startx) * speed_recog_info['slope'][recog_index] + starty + \
                    speed_recog_info['minRange'][recog_index]
            if speed[i] < lower:
                # 如果超出下限，继续判断是否符合允许的异常点限制
                if gv_dict['exceptionsCounter'] < speed_recog_info['permittedExceptions'][recog_index]:
                    # 如果仍在允许异常点限制范围内则异常点个数+1
                    gv_dict['exceptionsCounter'] = gv_dict['exceptionsCounter'] + 1
                else:
                    # 如果不在允许异常点限制范围内了则记录该点的转速信息并跳出循环停止转速识别
                    qDAQ_logger.info("Speed Recognition Error1: under the lower limit, test name: " +
                                     speed_recog_info['testName'][recog_index] + ", index: " + str(
                        recog_index))
                    # 记录当前点转速值和当前的转速识别下限
                    qDAQ_logger.info('present speed: ' + str(speed[i]) + ', lower limit: ' + str(lower))
                    # 停止转速识别并跳出（并不会停止转速计算）
                    gv_dict[
                        'speedRecogFinish'] = True  # stop the speed recognition and continue speed calculation
                    break
            else:
                # 计算上限值
                upper = (speed_loc[i] - startx) * speed_recog_info['slope'][recog_index] + starty + \
                        speed_recog_info['maxRange'][recog_index]
                if speed[i] > upper:
                    # 如果超出上限，继续判断是否符合允许的异常点限制
                    if gv_dict['exceptionsCounter'] < speed_recog_info['permittedExceptions'][
                        recog_index]:
                        # 如果仍在允许异常点限制范围内则异常点个数+1
                        gv_dict['exceptionsCounter'] = gv_dict['exceptionsCounter'] + 1
                    else:
                        # 如果不在允许异常点限制范围内了则记录该点的转速信息并跳出循环停止转速识别
                        qDAQ_logger.info("Speed Recognition Error2: over the upper limit, test name: " +
                                         speed_recog_info['testName'][recog_index] + ", index: " + str(
                            recog_index))

                        qDAQ_logger.info(
                            'present speed: ' + str(speed[i]) + ', upper limit: ' + str(upper))
                        gv_dict[
                            'speedRecogFinish'] = True  # stop the speed recognition and continue speed calculation
                        break
                else:
                    # 上下限均符合条件则开始拟合并判断结束点
                    if gv_dict['fitFlag']:
                        # 如果已经进行拟合（拟合点取自开始点到最短时长内的数据点），则开始判断是否满足结束点条件
                        if speed_loc[i] >= gv_dict['fitted_end']:
                            # 如果满足拟合出来的时长条件，则记录结束点信息
                            gv_dict['endpoint_loc'] = speed_loc[i]
                            gv_dict['endpoint_speed'] = speed[i]
                            gv_dict['endpoint_index'] = i
                            # 重置结束点识别标志信息
                            gv_dict['fitFlag'] = False
                            gv_dict['firstinFlag'] = True
                            # 记录结束点已识别完成
                            qDAQ_logger.info(
                                speed_recog_info['testName'][recog_index] + ", index: " + str(
                                    recog_index) + " end point detected")
                            break
                    else:
                        # 如果未进行拟合则判断是否符合拟合条件（需要满足最短时长）
                        if speed_loc[i] - startx >= speed_recog_info['minT'][recog_index]:
                            # 满足最短时长则进行拟合（利用scipy的stats的线性回归函数进行拟合），得到斜率、截距等信息
                            slope, intercept, r_value, p_value, std_err = stats.linregress(
                                speed_loc[gv_dict['startpoint_index']:i],
                                speed[gv_dict['startpoint_index']:i])
                            # 通过斜率截距等信息计算结束点位置
                            gv_dict['fitted_end'] = (speed_recog_info['endSpeed'][
                                                         recog_index] - intercept) / slope
                            gv_dict['fitFlag'] = True
                            # 判断是否已经满足结束点条件
                            if speed_loc[i] >= gv_dict['fitted_end']:
                                # 如果满足拟合出来的时长条件，则记录结束点信息
                                gv_dict['endpoint_loc'] = speed_loc[i]
                                gv_dict['endpoint_speed'] = speed[i]
                                gv_dict['endpoint_index'] = i
                                # 重置结束点识别标志信息
                                gv_dict['fitFlag'] = False
                                gv_dict['firstinFlag'] = True
                                # 记录结束点已识别完成
                                qDAQ_logger.info(
                                    speed_recog_info['testName'][recog_index] + ", index: " +
                                    str(recog_index) +
                                    " end point detected")
                                break
    else:
        #
        qDAQ_logger.info("can not confirm the speed pattern, please check the settings!")
        # 如果出现错误，如未定义的转速状态，则强制认为识别已完成，不再进行识别
        gv_dict['speedRecogFinish'] = True  # stop the speed recognition and continue speed calculation
    return gv_dict


def resolver(speed_sin, speed_cos, cut_level, coil, ppr, last_angle_l1f, loc_l1f, last_angle_l2f,
             loc_l2f, counter,
             frame):
    """
    双路旋变转速信号处理函数
    功能： 基于双路旋变信号（正弦余弦信号），找到每一个上升沿之间对应的角度差，根据每一个上升沿所在的角度去模拟上升沿位置，最终输出上升沿位置序列
    输入：
    1. 单帧正弦信号
    2. 单帧余弦信号
    3. 检测电平位（主要为了剔除0值附近的信号，防止干扰）
    4. 极对数
    5. 要输出的ppr
    6. 上一帧的最后一个角度值
    7. 上一帧的最后一个角度值对应的X轴（即时间点）
    8. 上两帧对应的最后一个角度值
    9. 上两帧的最后一个角度值对应的X轴（即时间点）
    10. 计数器，记录转过了多少个180度
    11. 帧长
    返回：
    1. 脉冲位置信息
    2. 上一帧的最后一个角度值
    3. 上一帧的最后一个角度值对应的X轴（即时间点）
    4. 上两帧对应的最后一个角度值
    5. 上两帧的最后一个角度值对应的X轴（即时间点）
    6. 计数器，记录转过了多少个180度
    function: to detect trigger point based on resolver signal(sin and cos)
    :param
    speed_sin(list): raw sin signal(1 frame)
    speed_cos(list): raw cos signal(1 frame)
    cut_level(float): to filter noise and detect out angle
    coil(int): coil num of resolver device(refer to the structure)
    ppr(int): target ppr num
    last_angle_l1f(float): last angle value of last frame
    loc_l1f(float): corresponding location of last_angle_l1f
    last_angle_l2f(float): last 2nd angle value of last frame
    loc_l2f(float): corresponding location of last_angle_l2f
    counter(int): index of data frame
    frame(int): length of 1 frame data
    :return:
    tl(list): trigger location of present frame
    following values just to real-time update:
        last_angle_l1f, loc_l1f, last_angle_l2f, loc_l2f, counter
    """
    # initial the trigger location
    tl = list()

    def sigma_calc(data):
        # 计算标准差
        # return np.sqrt(sum(np.power((data - np.mean(data)), 2)) / (len(data))) # 标准公式
        return np.std(data)

    def kick_out_exceptions(x_data, y_data, param_sigma, param_fit):
        # 剔除异常点
        # to remove the exception points
        counter_filter = 0
        while 1:
            # 进行线性拟合，param_fit为多项式系数，通常为1，表示线性拟合
            target_curve = np.poly1d(np.polyfit(x_data, y_data, param_fit))
            # 计算实际数据与拟合线的差值
            delta_y_value = np.array(y_data) - target_curve(x_data)
            # 计算差值的方差
            sigma_value = sigma_calc(delta_y_value)
            # 根据设定sigma去除倍数（倍数内的保留）来设定过滤条件
            condition1 = np.abs(delta_y_value - np.mean(delta_y_value)) <= (param_sigma * sigma_value)
            # 数据过滤
            filter_y_data = np.array(y_data)[condition1]
            filter_x_data = np.array(x_data)[condition1]
            # 更新计数器
            counter_filter += 1
            # 更新数据重新进行过滤，直到无法剔除
            if len(y_data) > len(filter_y_data):
                # 顾虑前后的数据长度一致表示无法再过滤了
                y_data, x_data = filter_y_data, filter_x_data
            else:
                break

        return filter_x_data, filter_y_data, counter_filter

    # 去除零附近的数据点（这些数据点干扰严重），后续进行插值补上去掉的数据点
    # cut out the better signal
    cutted_sin = speed_sin[(abs(speed_sin) > cut_level) & (abs(speed_cos) > cut_level)]
    cutted_cos = speed_cos[(abs(speed_sin) > cut_level) & (abs(speed_cos) > cut_level)]
    # 记录剩余数据点的索引信息
    print(f'{cutted_sin=},{cutted_cos=}')
    cutted_loc = np.where((abs(speed_sin) > cut_level) & (abs(speed_cos) > cut_level))[0]

    # convert into angle（0，180）
    # 计算角度值（基本正余弦信号）
    angle = 90 + np.arctan(cutted_sin / cutted_cos) * 180 / np.pi
    # 根据极对数转换点击旋转的实际角度参考值，每个ppr转过多少度
    angle_ppr = 360 * coil / ppr  # 该角度是旋变信号的角度，电机的角度为360/ppr
    # 设定判断角度跳跃的标准
    shift_angle = -76

    # 计算累积角度
    if len(angle) > 0:
        # 有数据才进行计算
        # The angle angle (0180) is changed into monotonically increasing angle_ Cum
        angle_cum = list()

        # If there is a previous frame, add the last point of the last frame first
        if loc_l1f:
            # 如果存在上一帧的数据点则进行判断是否需要更新角度值（主要处理第一个数据点）
            start_angle = last_angle_l1f - counter * 180
            if angle[0] - start_angle < shift_angle:
                # 如果出现角度跳跃，则认为需要加180度
                counter += 1
        # 记录累积角度
        angle_cum.append(angle[0] + counter * 180)

        # 处理剩余数据点（除第一个数据点外）
        for i in range(1, len(angle)):
            # 两个条件需要同时满足：1. 角度发生跳跃；2. 不是连续的数据点（防止噪点）
            if angle[i] - angle[i - 1] < shift_angle and cutted_loc[i] - cutted_loc[i - 1] > 1:
                counter += 1
            # 更新累积角度序列
            angle_cum.append(angle[i] + counter * 180)

        if len(angle) > 1:
            # 只有超过两个数据点才能进行拟合，所以必须确保数据量才进行剔除，这里设置的线性回归，5倍的标准差（超过5倍才去除）
            # At least two points can be merged to remove noise
            filter_scl, filter_angle_cum, _ = kick_out_exceptions(cutted_loc, angle_cum, 5, 1)
        else:
            # 数据量很少就不用进行剔除操作了
            # There is only one point in the frame
            filter_scl, filter_angle_cum = cutted_loc, angle_cum
        if loc_l1f:
            # 如果存在上一帧的数据点需要结合前一帧的数据点进行插值计算
            # There are data points in the last frame
            if loc_l2f:
                # 如果前两帧存在数据点则需要结合前两帧的数据进行插值计算
                # There are also data points in the previous frame, and the quadratic difference is performed
                # 生产插值曲线函数，基于角度及其对应的位置信息，这里需要注意用了二次方差值（主要考虑转速拐点），前两帧的数据点只是用于构造曲线，
                # 因为要更新上一帧的数据点
                Curve = interp1d(np.array([loc_l2f - 2 * frame, loc_l1f - frame, filter_scl[-1]]),
                                 np.array([last_angle_l2f, last_angle_l1f, filter_angle_cum[-1]]),
                                 kind='quadratic',
                                 assume_sorted=True)
            else:
                # 前两帧不存在数据的情况下，则直接根据前一帧的数据进行插值计算，这里用的是线性差值
                # If there is no data point in the previous frame, linear difference is made
                Curve = interp1d(np.array([loc_l1f - frame, filter_scl[-1]]),
                                 np.array([last_angle_l1f, filter_angle_cum[-1]]), kind='linear',
                                 assume_sorted=True)
            # 生成序列的X值（补全X轴，包括上一帧最后一个数据点与当前帧第一个数据点之间的数据）
            index_serial = np.arange(loc_l1f - frame, filter_scl[-1] + 1, 1)
            # 生成角度序列（累积角度），该序列已经补上了最开始剔除的数据点处的角度
            angle_serial = Curve(index_serial)
            # 基于每个ppr对应的角度可以得到锯齿形状的数据序列（每个跳变的位置对应的一个脉冲信号）
            angle_serial_remainder = angle_serial % angle_ppr
            # 识别脉冲触发点，angle_ppr * 0.5是为了保证能识别到（防止干扰）
            triggermarker = 1 * (np.diff(angle_serial_remainder) < -(angle_ppr * 0.5))
            # 记录每个脉冲位置信息（之后便可用于转速计算和角度域重采样），前一帧的脉冲位置应该为负
            tl.extend(list(np.where(triggermarker == 1)[0] - (frame - loc_l1f)))
            # 更新要返回的信息（前一帧，前两帧的最后一个数据点）
            last_angle_l2f = last_angle_l1f
            last_angle_l1f = filter_angle_cum[-1]
            loc_l2f = loc_l1f
            loc_l1f = filter_scl[-1]
            last_angle_l1f = filter_angle_cum[-1]
            loc_l1f = filter_scl[-1]
        else:
            # 如果上一帧不存在数据点，前两帧也认为不存在数据点（事实上也应该不存在这种上一帧没有但是上上帧有数的情况），只处理当前帧
            Curve = interp1d(np.array([0, filter_scl[-1]]),
                             np.array([filter_angle_cum[0], filter_angle_cum[-1]]),
                             kind='linear', assume_sorted=True)
            # 生成序列的X值（补全X轴）
            index_serial = np.arange(filter_scl[0], filter_scl[-1] + 1, 1)
            # 生成角度序列（累积角度），该序列已经补上了最开始剔除的数据点处的角度
            angle_serial = Curve(index_serial)
            # 基于每个ppr对应的角度可以得到锯齿形状的数据序列（每个跳变的位置对应的一个脉冲信号）
            angle_serial_remainder = angle_serial % angle_ppr
            # 识别脉冲触发点，angle_ppr * 0.5是为了保证能识别到（防止干扰）
            triggermarker = 1 * (np.diff(angle_serial_remainder) < -(angle_ppr * 0.5))
            # 记录每个脉冲位置信息（之后便可用于转速计算和角度域重采样）
            tl.extend(list(np.where(triggermarker == 1)[0]))
            # 更新要返回的信息（前一帧，前两帧的最后一个数据点）
            last_angle_l2f = last_angle_l1f
            last_angle_l1f = filter_angle_cum[-1]
            loc_l2f = loc_l1f
            loc_l1f = filter_scl[-1]
    else:
        # 如果当前帧不存在任何数据，则直接更新要返回的信息
        # no angle points
        loc_l2f = loc_l1f
        loc_l1f = None
        last_angle_l2f = last_angle_l1f
        last_angle_l1f = None

    return tl, last_angle_l1f, loc_l1f, last_angle_l2f, loc_l2f, counter


def resolver_single_signal_for_share(speed, left_index, right_index,
                                     last_frame_status, trigger_per_fish, trigger_array, index_trigger,
                                     speedCalcInfo):
    """
    需要传递给下一帧的数据为当前帧倒数第二或者倒数第一个 左上跳点之前 大概四分之一条鱼的信号数据
    需要传递给下一帧的数据为当前帧倒数第二或者倒数第一个 左上跳点之前 的哪一个右下 大概四分之一条鱼的信号数据
    如果该帧数据可以计算出倒数第一个左上跳点之后下降段的上升沿信息，就返回倒数第一个左上跳点
    如果该帧数据无法计算出倒数第一个左上跳点之后下降段的上升沿信息，就返回倒数第二个左上跳点
    该帧上升沿会包含返回跳点之后紧接的下降段，再紧接着的上升段一定不会包含
    该帧数据的上升沿计算结果 不 包括最后一个零点所在位置
    """

    # 汇川下面这四个值可能要根据电机以及采集到的信号进行调整，
    lowLevel = 6  # 下极值点在 0~lowLevel 度之间。
    highLevel = 35  # 上极值点。Hilbert变换出来的上极值点以上的角度不可用，扔掉后靠下面的点拟合后插值补上。
    # 寻找跳点的时候，要求当前跳点以及下一个跳点的值
    min_value = 20
    max_value = 20

    # # # # 比亚迪
    lowLevel = 10  # 下极值点在 0~lowLevel 度之间。
    highLevel = 60  # 上极值点。Hilbert变换出来的上极值点以上的角度不可用，扔掉后靠下面的点拟合后插值补上。
    # 寻找跳点的时候，要求当前跳点以及下一个跳点的值
    min_value = 35
    max_value = 35

    lowLevel = speedCalcInfo["lowCutOffAngle"]
    highLevel = speedCalcInfo["highCutOffAngle"]
    min_value = speedCalcInfo["lowFilterAngle"]
    max_value = speedCalcInfo["highFilterAngle"]

    speed_sin = speed[left_index:right_index]
    hSin = fftpack.hilbert(speed_sin)

    # 舍弃前10个点，后10个点
    # speed_sin=speed_sin[10:len(speed_sin)-10]
    # hSin=hSin[10:len(hSin)-10]
    # left_index+=10

    envSin = np.sqrt(speed_sin ** 2 + hSin ** 2)

    amp = np.max(envSin)
    angle0 = np.arcsin(envSin / amp) * 180 / np.pi

    # plt.figure("envSin")
    # plt.plot(angle0)
    # plt.show()

    cutted_angle = angle0[(angle0 > lowLevel) & (angle0 < highLevel)]
    # cutted_loc为所有点的索引，并非是第一个点的索引
    # 注意数组中存的数据的意义，
    cutted_loc = np.where((angle0 > lowLevel) & (angle0 < highLevel))[0]  # cutted_angle在angle0中的索引

    skipP = np.where(np.diff(cutted_loc) > 1)[0]  # 跳点在cutted_loc数组中的索引

    # 状态机
    status = last_frame_status

    # 该list中的数据为该帧sin数据中的索引，为整数
    zero_points = []

    # hilbert变换后开头的一部分数据不能用,转速为10000rpm时，每条鱼内有102400/(10000/60*4*2)=76.8个点
    # 所以在(status == 4 or status == -1) 判断中加上cutted_loc[skipP[i]]>10
    # 如果转速或者采样率发生变化，这个点应该会发生改变
    for i in range(np.size(skipP)):
        # 上左最后一个跳点 在跳点位置可能有噪声，要求该跳点的下一个跳点

        if (status == 4 or status == -1) and cutted_angle[skipP[i]] > max_value and cutted_loc[
            skipP[i]] > 10 and (i == np.size(skipP) - 1 or cutted_angle[skipP[i + 1]] < min_value):
            # if (status == 4 or status == -1) and cutted_angle[skipP[i]] > max_value  \
            #     and (i == np.size(skipP) - 1 or cutted_angle[skipP[i + 1]] < min_value):
            # 找到紧挨跳点右边的点在cutted_loc中的索引
            upLeft_index = skipP[i] + 1
            status = 1
            continue

        # 下左第一个跳点
        if status == 1 and cutted_angle[skipP[i - 1]] > max_value and (
                i == np.size(skipP) - 1 or cutted_angle[skipP[i]] < min_value):
            # 找到跳点在cutted_loc中的索引
            downLeft_index = skipP[i]
            status = 2
            kb = np.polyfit(cutted_loc[upLeft_index:downLeft_index],
                            cutted_angle[upLeft_index:downLeft_index], 1)
            left_zero_point = -kb[1] / kb[0]

        # 找到右下最后一个跳点
        if status == 2 and cutted_angle[skipP[i]] < min_value and (
                i == np.size(skipP) - 1 or cutted_angle[skipP[i + 1]] > max_value):
            # 找到紧挨跳点右边的点在cutted_loc中的索引
            downRight_index = skipP[i] + 1
            status = 3
            continue

        # 找到右上第一个跳点
        if status == 3 and cutted_angle[skipP[i]] > max_value and cutted_angle[
            skipP[i - 1]] < min_value:
            # 找到跳点在cutted_loc中的索引
            upRight_index = skipP[i]
            status = 4
            # downLeftP 与 upLeftP 两个点之间的采样点，形成线段。延长线段到0轴，得到0点

            # # 线性拟合，返回系数k,b
            kb = np.polyfit(cutted_loc[downRight_index:upRight_index],
                            cutted_angle[downRight_index:upRight_index], 1)
            right_zero_point = -kb[1] / kb[0]

            ave = (left_zero_point + right_zero_point) / 2
            zero_points.append(ave)
            if (i == np.size(skipP) - 1 or cutted_angle[skipP[i + 1]] < 20):  # 如果同时也是上左最后一个跳点
                upLeft_index = skipP[i] + 1
                status = 1

    # speed_sin_index_for_return的索引零点为left_index
    if len(zero_points) == 0 or len(zero_points) == 1:
        speed_sin_index_for_return = 0
    else:
        # 下面这句话会使第二帧无法识别第一个左上跳点
        # speed_sin_index_for_return = int((zero_points[-1] - zero_points[-2]) * 1 / 4 + zero_points[-2])
        speed_sin_index_for_return = int((zero_points[-1] - zero_points[-2]) * 0 + zero_points[-2])
    #     将两个零点之间大概3/4的数据放入下一帧

    # 根据零点的位置找脉冲信号的位置
    # 电机每转一圈，有ppr个上升沿信号，会有coil个完整的包络信号，即有2*coil个鱼，每条鱼上有ppr/2/coil个上升沿脉冲
    # 认为每一条鱼上是匀速的

    for i in range(len(zero_points) - 1):
        # 下面用的列表生成器的长度等于np.int(ppr/2/coil),即不包括一条鱼的最后一个点，这个点是下一条鱼的第一个点
        trigger_array[index_trigger:index_trigger + trigger_per_fish] = \
            np.linspace(zero_points[i], zero_points[i + 1], trigger_per_fish, endpoint=False) \
            + left_index
        index_trigger += trigger_per_fish

    # 帧内匀速
    # trigger_array[index_trigger:index_trigger + (trigger_per_fish) * (len(zero_points) - 1)] = \
    #     np.linspace(zero_points[0], zero_points[len(zero_points) - 1],
    #                         trigger_per_fish * (len(zero_points) - 1), endpoint=False) + left_index
    # index_trigger += (trigger_per_fish) * (len(zero_points) - 1)

    # ppr设置为2*coils，这样zeropoints即为上升沿
    # trigger_array[index_trigger:index_trigger + (len(zero_points) - 1)] = np.rint(
    #     zero_points[:(len(zero_points) - 1)])+left_index
    # index_trigger += (len(zero_points) - 1)
    if (status == -1):
        pass
    else:
        status = 4

    # speed_sin_index_for_return是传递给下一帧的起始位置在 第 counter*帧长到第(counter+1)*帧长 中的索引
    return index_trigger, speed_sin_index_for_return + left_index, status
    # return tl, tl_whole_frame,tl_half_frame,tl_quarter_frame, sin_to_return, status, counter


def single_resolver_butter_filter(speed_array, left_index, right_index, sampleRate,
                                  coils,
                                  trigger_array, index_trigger, ppr):

    # left_index为实际需要的段，滤波时要向左取帧长1/10
    # right_index为原始数据存储到的位置，左帧长1/10处在该帧滤波后的数据不使用
    right_more = int((right_index - left_index) / 10)
    if left_index == 0:
        left_more = 0
    else:
        left_more = right_more
    frame = speed_array[left_index - left_more:right_index]
    # 除第一帧，frame以上一帧最后一个零点的右中值跳跃点开始
    frame_filter = butter_filter(frame, [int((10000 + 10000 / 60) * 1.4)], sampleRate)
    hSin = fftpack.hilbert(frame_filter)
    envSin = np.sqrt(frame_filter ** 2 + hSin ** 2)
    max_envSin = np.max(envSin)
    low_level = max_envSin / 5
    half_loc = np.where(envSin > max_envSin / 2)[0]
    half_loc_skip = np.where(np.diff(half_loc) != 1)[0]
    min_list = list()
    trigger_per_fish = int(ppr / 2 / coils)

    if not len(half_loc_skip):
        return index_trigger, left_index

    left_index_for_next_frame = None
    for i in range(len(half_loc_skip)):
        if (half_loc[half_loc_skip[i] + 1] - half_loc[half_loc_skip[i]]) > (
                len(frame) / len(half_loc_skip) / 3 * 0.6):
            min_loc = np.argmin(envSin[half_loc[half_loc_skip[i]]:half_loc[half_loc_skip[i] + 1]])
            if envSin[min_loc + half_loc[half_loc_skip[i]]] < low_level:
                min_list.append(min_loc + half_loc[half_loc_skip[i]])
                # left_index_for_next_frame = half_loc[half_loc_skip[i]]-int(point_per_fish/4)

    # min_list 和 min_array都是以帧起点为0点
    min_array = np.array(min_list)
    for loc in min_array:
        envSin[loc:] *= -1

    try:
        envSin_filter = butter_filter(envSin,
                                      [int(len(half_loc_skip) * (sampleRate / len(frame) * 1.4))],
                                      sampleRate)[left_more:-right_more]
    except Exception:
        traceback.print_exc()

    zeroLoc = np.where(np.diff(1 * (envSin_filter >= 0)) != 0)[0]

    for i in range(len(zeroLoc) - 1):
        # 下面用的列表生成器的长度等于np.int(ppr/2/coil),即不包括一条鱼的最后一个点，这个点是下一条鱼的第一个点
        trigger_array[index_trigger:index_trigger + trigger_per_fish] = \
            np.linspace(zeroLoc[i], zeroLoc[i + 1], trigger_per_fish, endpoint=False) \
            + left_index
        index_trigger += trigger_per_fish

    if len(zeroLoc) == 0:
        return index_trigger, left_index
    else:
        return index_trigger, zeroLoc[-1] - int(len(frame) / len(half_loc_skip) / 4) + left_index


def ramp_quality(speed_curve_x, speed_curve_y, speed_recog_info, recog_index):
    """
    功能：计算转速的爬坡质量
    输入：
    1. 转速曲线的x轴，必须是array
    2. 转速曲线的y轴，必须是array
    3. 转速识别信息，里面包含了目标转速区间和期望时长
    4. 目标工况的索引，即第几个测试段
    返回：爬坡质量
    """
    if len(speed_curve_x) > 0:
        # 区别恒速段变速段
        if speed_recog_info['speedPattern'][recog_index] == 2:
            # 升速段
            target_end_speed = speed_recog_info['endSpeed'][recog_index]
            y = speed_recog_info['slope'][recog_index] * (speed_curve_x - speed_curve_x[0]) + \
                speed_curve_y[0]
            # 只对小于终止转速的部分进行评估
            target_condition = (y <= target_end_speed) & (speed_curve_y <= target_end_speed)
            target_y = y[target_condition]
            target_speed_curve_y = speed_curve_y[target_condition]
            quality = 1 - np.sqrt(
                np.sum(np.power(target_speed_curve_y / target_y - 1, 2)) / len(target_y))
        elif speed_recog_info['speedPattern'][recog_index] == 3:
            # 降速段
            target_end_speed = speed_recog_info['endSpeed'][recog_index]
            y = speed_recog_info['slope'][recog_index] * (speed_curve_x - speed_curve_x[0]) + \
                speed_curve_y[0]
            # 只对大于终止转速的数据进行评估
            target_condition = (y >= target_end_speed) & (speed_curve_y >= target_end_speed)
            target_y = y[target_condition]
            target_speed_curve_y = speed_curve_y[target_condition]
            quality = 1 - np.sqrt(
                np.sum(np.power(target_speed_curve_y / target_y - 1, 2)) / len(target_y))
        else:
            # 恒速段
            target_speed = np.mean(
                [speed_recog_info['startSpeed'][recog_index],
                 speed_recog_info['endSpeed'][recog_index]])
            quality = 1 - np.sqrt(
                np.sum(np.power(speed_curve_y / target_speed - 1, 2)) / len(speed_curve_y))
    else:
        quality = 0
    return {"name": "RampQuality", "unit": "", "value": quality, "indicatorDiagnostic": 1}


def ramp_quality_for_const(speed_curve_x, speed_curve_y, speed_recog_info, recog_index):
    """
    功能：计算转速的爬坡质量
    输入：
    1. 转速曲线的x轴，必须是array
    2. 转速曲线的y轴，必须是array
    3. 转速识别信息，里面包含了目标转速区间和期望时长
    4. 目标工况的索引，即第几个测试段
    返回：爬坡质量
    """
    if len(speed_curve_x) > 0:
        # 恒速段
        target_speed = np.mean(
            [speed_recog_info['startSpeed'][recog_index], speed_recog_info['endSpeed'][recog_index]])
        quality = 1 - np.sqrt(np.sum(np.power(speed_curve_y / target_speed - 1, 2)) / len(speed_curve_y))
    else:
        quality = 0
    return {"name": "RampQuality", "unit": "", "value": quality, "indicatorDiagnostic": 1}


def speed_calibration(vib, min_speed, max_speed, order, fs):
    '''
    speed calibration for constant speed
    ************************************************
    parameters
           vib: vibration data
     speed_ref: speed reference
         order: main order for machine
            fs: sampling rate
    return
    speed_cali: speed after calibration
    '''

    n = len(vib)
    vib_fft = np.fft.rfft(vib)
    freq = np.fft.rfftfreq(n, d=1 / fs)

    # frequency range for speed
    left = min_speed / 60 * order
    right = max_speed / 60 * order
    idx = (freq >= left) & (freq <= right)

    # find target frequency

    target = np.argmax(np.abs(vib_fft[idx]))
    # target_min = np.argmin(vib_fft[idx])
    speed_cali = freq[idx][target] / order * 60
    # speed_cali = (freq[idx][target] / order * 60+freq[idx][target_min] / order * 60)/2
    return speed_cali

def speed_calibration_with_orderlist(vib, min_speed, max_speed, order, fs):
    '''
    speed calibration for constant speed
    ************************************************
    parameters
           vib: vibration data
     speed_ref: speed reference
         order: main order for machine
            fs: sampling rate
    return
    speed_cali: speed after calibration
    '''

    frame=vib
    if len(vib)>8192:
        frame=vib[:8192]
    samplePerChan=len(vib)
    ref_order_array=np.sort(order)
    frame_fft_abs = np.abs(np.fft.rfft(frame)) / len(frame) * 2
    frame_freq = np.fft.rfftfreq(len(frame), d=1 / fs)

    # 多个阶次的话为最大阶次的频率寻找范围，单个阶次时为该阶次的寻找范围
    max_freq = max_speed / 60 * ref_order_array[-1]
    min_freq = min_speed / 60 * ref_order_array[-1]

    if len(ref_order_array) == 1:
        idx = (frame_freq >= min_freq) & (frame_freq <= max_freq)
        # find target frequency
        target = np.argmax(np.abs(frame_fft_abs[idx]))
        # target_min = np.argmin(vib_fft[idx])
        speed_cali = frame_freq[idx][target] / ref_order_array[-1] * 60
        return speed_cali

    fft_abs_cut = frame_fft_abs[frame_freq <= max_freq]
    freq_cut = frame_freq[frame_freq <= max_freq]
    peaks_index, _ = signal.find_peaks(fft_abs_cut)

    max_n = get_nlargest(fft_abs_cut[peaks_index], 100)
    max_n = np.sort(max_n)
    max_n_freq = freq_cut[peaks_index][max_n]
    max_n_vib = fft_abs_cut[peaks_index][max_n]

    peak_to_compare_list = list()
    last_peak_to_compare_index = 0
    min_diff_frequency = 0.5 * min_speed / 60

    for i in range(0, len(max_n_freq)):
        if max_n_freq[i] < min_diff_frequency:
            continue
        if max_n_freq[i] - max_n_freq[last_peak_to_compare_index] < min_diff_frequency:
            if max_n_vib[i] > max_n_vib[last_peak_to_compare_index]:
                last_peak_to_compare_index = i
        else:
            if max_n_freq[last_peak_to_compare_index] > min_diff_frequency:
                peak_to_compare_list.append(last_peak_to_compare_index)

            last_peak_to_compare_index = i

    peak_to_compare_list.append(last_peak_to_compare_index)
    peak_to_compare_array = np.array(peak_to_compare_list)
    freq_to_compare = max_n_freq[peak_to_compare_array]
    al_to_compare = max_n_vib[peak_to_compare_array]

    freq_revolution = fs / len(frame)
    # 每一个点作为第一个参考阶次的评分
    score = list()
    vib_to_compare = max_n_vib[peak_to_compare_array]
    arg_vib_max = np.argsort(vib_to_compare)
    rank = np.zeros(len(vib_to_compare))
    rank[arg_vib_max] = range(1, len(vib_to_compare) + 1)
    score_without_al = list()
    score_al = list()
    for i in range(len(freq_to_compare) - 1, -1, -1):
        if freq_to_compare[i] < min_freq:
            break
        speed_temp = freq_to_compare[i] / ref_order_array[-1]
        freq_to_find = speed_temp * ref_order_array
        argmin_index = list(map(find_argmin_in(freq_to_compare), freq_to_find))
        # if i==7:
        #     time.sleep(0.1)

        # if len(np.where(np.abs(freq_to_find - freq_to_compare[argmin_index])>3*freq_revolution)[0])!=0:
        #     score.append(0)
        #     score_without_al.append(0)
        #     score_al.append(0)
        #     continue
        # score.append(np.sum(1 - np.abs(freq_to_find - freq_to_compare[argmin_index]) / freq_to_find) + np.sum(freq_revolution  * rank[argmin_index]))
        # score_without_al.append(np.sum(1 - np.abs(freq_to_find - freq_to_compare[argmin_index]) / freq_to_find))
        # score_al.append(np.sum(freq_revolution  * rank[argmin_index]))

        freq_right_index = \
        np.where(np.abs(freq_to_find - freq_to_compare[argmin_index]) <= 3 * freq_revolution)[0]
        score.append(np.sum(al_to_compare[argmin_index][freq_right_index]))

        # score.append(np.sum(1-np.abs(speed_temp_array-freq_to_compare[l])/speed_temp_array))
        # a=freq_to_compare-speed_temp_array.reshape((len(ref_order_array)-1,1))
        # min_index=np.argmin(np.abs(a), axis=1)
        # if np.max(a[min])>min_diff_frequency:
        #     continue
        # else:
        #     rpm_list.append(speed_temp*60)
        #     break
    index_first_order = np.argmax(score)
    speed_cali=freq_to_compare[len(freq_to_compare)-1-index_first_order]/ref_order_array[-1]*60
    return speed_cali

def find_argmin_in(a):
    def find_argmin(y):
        return np.argmin(np.abs(a-y))
    return find_argmin


def get_nlargest(array, n=3):
    '''
    get n largest number in array
    ********************************************
    parameters
         array: array to be processed
    return
           arg: index of n largest numbers
    '''
    arr = np.array(array)
    arg = arr.argsort()[-n:]
    return arg


if __name__ == '__main__':
    import logging

    # 下面的代码为测试代码，针对该模块
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(pathname)s %(funcName)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='[%Y-%m-%d %H:%M:%S]',
        filename='D:/qdaq/Log/temp.log',
        filemode='a'
    )


    def simulation_real_time_angle(speed_sin, speed_cos, speedCalcInfo):
        # 该函数为旋变信号按帧处理过程模拟函数，最终得到脉冲触发点位置
        cut_level = speedCalcInfo["triggerLevel"]
        coils = speedCalcInfo["coils"]
        ppr = speedCalcInfo["ppr"]
        frame_length = 8000
        counter = 0
        last_angle_l2f, loc_l2f = 0, None
        last_angle_l1f, loc_l1f = 0, None
        trigger = list()
        ib = 0

        for i in range(0, len(speed_sin) // frame_length):
            tl, last_angle_l1f, loc_l1f, last_angle_l2f, loc_l2f, counter = resolver(
                speed_sin[i * frame_length:(i + 1) * frame_length],
                speed_cos[i * frame_length:(i + 1) * frame_length],
                cut_level, coils, ppr, last_angle_l1f, loc_l1f, last_angle_l2f, loc_l2f, counter,
                frame_length)
            trigger.extend(np.array(tl) + ib * frame_length)
            ib += 1

        return trigger


    def set_default_speedRecog():
        # 该函数为转速识别标志及相关临时结果的记录表
        _global_dict = dict()
        # first part for speed detect start
        _global_dict['startFlag'] = False
        _global_dict['startpoint_loc'] = None
        _global_dict['startpoint_speed'] = None
        _global_dict['startpoint_index'] = None
        _global_dict['tempStartFlag'] = False

        # second part for speed detect end
        _global_dict['firstinFlag'] = True
        _global_dict['exceptionsCounter'] = 0
        _global_dict['fitFlag'] = False
        _global_dict['fitted_end'] = None
        _global_dict['endpoint_loc'] = None
        _global_dict['endpoint_speed'] = None
        _global_dict['endpoint_index'] = None
        _global_dict['speedRecogFinish'] = False

        return _global_dict


    try:
        # 导入相关的包
        from parameters import Parameters, config_folder, config_file
        from auxiliary_tools import read_tdms
        from utils import write_json
        import time
        import matplotlib.pyplot as plt

        type_info = "AP4000"
        config_filename = os.path.join(config_folder, "_".join([type_info, config_file]))
        print(config_filename)
        if os.path.exists(config_filename):
            # 如果配置参数存在则继续执行
            param = Parameters(config_filename)
            print(param.speedRecogInfo)
            # 读取原始数据
            tdms_filename = r'D:\qdaq\Simu\aiwayTest\20210514\TEST0515-9.tdms'
            speed_data, pp = read_tdms(tdms_filename, 'AIData', 'Speed')

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
            for counter in range(frame_num):
                print(counter, gv_dict_speedRecog)
                # 每个工况识别完需要往前倒退一帧
                if counter == icounter:
                    # 计算转速
                    if counter == 0:
                        lfp = speed_data[0]
                    # speed calculation
                    temptl, lfp = trigger_detect(speed_data[counter * frame: (counter + 1) * frame],
                                                 lfp,
                                                 param.speedCalcInfo)
                    temptl = list(np.array(temptl) + counter * frame)
                    triggerLocation.extend(temptl)
                    temp_rpml, temp_rpm = rpm_calc(temptl, triggerLocation, sampleRate,
                                                   param.speedCalcInfo)
                    rpml.extend(temp_rpml)
                    rpm.extend(temp_rpm)
                else:
                    icounter = counter
                if not gv_dict_speedRecog['speedRecogFinish']:
                    # speed recognition
                    if not gv_dict_speedRecog['startFlag']:
                        # detect the start point of target operating mode
                        gv_dict_speedRecog = speed_detect_start(temp_rpml, temp_rpm, rpm, recog_index,
                                                                gv_dict_speedRecog,
                                                                param.speedRecogInfo)
                        # special case: start point and end point in same frame
                        if gv_dict_speedRecog['startFlag']:
                            # 第一帧识别到了开始点
                            gv_dict_speedRecog = speed_detect_end(temp_rpm, rpml, rpm,
                                                                  recog_index, gv_dict_speedRecog,
                                                                  param.speedRecogInfo)
                            if gv_dict_speedRecog['endpoint_loc']:
                                # 识别到结束点
                                temp_speed_x = np.array(
                                    rpml[gv_dict_speedRecog['startpoint_index']: gv_dict_speedRecog[
                                        'endpoint_index']])
                                temp_speed_y = np.array(
                                    rpm[gv_dict_speedRecog['startpoint_index']: gv_dict_speedRecog[
                                        'endpoint_index']])
                                # 转速曲线添加噪声
                                # temp_speed_y += np.random.randint(90, 100, len(temp_speed_y))
                                # recog index is for test name list of speed recognition
                                recog_result.append(
                                    {'testName': param.speedRecogInfo['testName'][recog_index],
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
                        gv_dict_speedRecog = speed_detect_end(temp_rpm, rpml, rpm,
                                                              recog_index, gv_dict_speedRecog,
                                                              param.speedRecogInfo)
                        if gv_dict_speedRecog['endpoint_loc']:
                            temp_speed_x = np.array(
                                rpml[gv_dict_speedRecog['startpoint_index']: gv_dict_speedRecog[
                                    'endpoint_index']])
                            temp_speed_y = np.array(
                                rpm[gv_dict_speedRecog['startpoint_index']: gv_dict_speedRecog[
                                    'endpoint_index']])
                            # 转速曲线添加噪声
                            # temp_speed_y += np.random.randint(90, 100, len(temp_speed_y))
                            # recog index is for test name list of speed recognition
                            recog_result.append(
                                {'testName': param.speedRecogInfo['testName'][recog_index],
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
            # 将识别结果标注在转速曲线上
            print(recog_result)
            recog_result_filename = r'D:\qdaq\temp\TEST0515-9.json'
            write_json(recog_result_filename, recog_result)
            speed_loc, speed_value = np.array(rpml), np.array(rpm)
            legend = list()
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 7))
            ax1.plot(speed_loc, speed_value, '-k*')
            ax1.set_title('Speed Curve')
            ax1.set_xlabel('Time/s')
            ax1.set_ylabel('Speed/rpm')
            legend.append('Raw Speed Curve')
            for result in recog_result:
                condition = (speed_loc >= result['startTime']) & (speed_loc <= result['endTime'])
                ax1.plot(speed_loc[condition], speed_value[condition])
                legend.append(result['testName'])
            plt.legend(legend)
            plt.show()
    except Exception:
        traceback.print_exc()
        logging.warning("exec failed, failed msg:" + traceback.format_exc())
        sys.exit()

