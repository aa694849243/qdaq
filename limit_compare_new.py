#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020/6/30 9:59
@Author  : Sergei@Synovate
@Email   : qxu@sonustc.com
description: limit related functions, v1.9.5
update: update over limit ratio at 20210514
"""

import json
import logging
import traceback
import sys
import time
import numpy as np

# 设置评判结果
in_limit = 1  # 合格
under_rms = 2  # 异常
under_low_limit = 0  # 不合格
over_high_limit = 0  # 不合格
without_limit = -1  # 界限值缺失
without_comp = -2  # 不进行界限值评判

# TODO: 优先级可配置（可能需要更新参数配置）


def comparator(data, limit, test_name, limit_compare_info, ond_os_calc_info):
    """
    功能：界限值评判
    输入：
    1. 结果数据，一般为未经评判的结果数据（所有结果的评判结果为-1，即界限值缺失）
    2. 界限值数据，一般是在SigMA系统中设置并下发的界限值
    3. 界限值评判参数，包括：
        1. 一维指标评判标志，0或者1,1表示开启（会与界限值进行比较），0表示不开启（一维的评判结果会置为1，即合格）
        2. 二维时间域指标评判标志，0或者1,1表示开启（会与界限值进行比较），0表示不开启（一维的评判结果会置为1，即合格），需结合线性或区域评判标志使用
        3. 二维阶次切片指标评判标志，0或者1,1表示开启（会与界限值进行比较），0表示不开启（一维的评判结果会置为1，即合格），需结合线性或区域评判标志使用
        4. 二维阶次谱指标评判标志，0或者1,1表示开启（会与界限值进行比较），0表示不开启（一维的评判结果会置为1，即合格），需结合线性或区域评判标志使用
        5. 二维倒阶次谱指标评判标志，0或者1,1表示开启（会与界限值进行比较），0表示不开启（一维的评判结果会置为1，即合格），需结合线性或区域评判标志使用
        6. 二维指标评判的线性界限值评判标志，配合二维指标评判开启标志使用
        7. 二维指标评判的区域界限值评判标志，配合二维指标评判开启标志使用
        8. 超限比计算开启标志，1表示开启（进行计算），0表示不开启（不进行计算）
    4. 一维关注阶次切片参数信息，主要用于超限比计算，剔除已经关注过的阶次切片
    返回：
    1. 更新后结果数据（完成界限值评判的结果数据）
    """

    def get_final_result_old(result_list):
        """
        功能：整理最终结果（基于优先级，界限值缺失高于合格）
        输入：
        1. 评判结果列表
        返回：
        1. 最终评判结果
        """
        if under_rms in result_list:
            # 异常为最高优先级
            return under_rms
        elif under_low_limit in result_list:
            # 第二优先级为低于下限
            return under_low_limit
        elif over_high_limit in result_list:
            # 第三优先级为高于上限
            return over_high_limit
        elif without_limit in result_list:
            # 第四优先级为界限值缺失
            return without_limit
        else:
            # 上述均不存在则认为是合格
            return in_limit

    def get_final_result(result_list):
        """
        功能：整理最终结果（基于优先级，合格高于界限值缺失，不比较为单独的评判结果）
        输入：
        1. 评判结果列表
        返回：
        1. 最终评判结果
        """
        if under_rms in result_list:
            # 异常为最高优先级
            return under_rms
        elif under_low_limit in result_list:
            # 第二优先级为低于下限
            return under_low_limit
        elif over_high_limit in result_list:
            # 第三优先级为高于上限
            return over_high_limit
        elif in_limit in result_list:
            # 第四优先级为合格
            return in_limit
        elif without_limit in result_list:
            # 第五优先级为界限值缺失
            return without_limit
        else:
            # 上述均不存在则认为是均不参与评判（最低优先级）
            return without_comp

    def generate_point(x, x1, x2, y1, y2):
        # 插值得到指定x的y值（x1<=x<=x2)
        return (y2 - y1) / (x2 - x1) * (x - x1) + y1

    def line_comparator(data_value_raw, data_speed_raw, limit_speed_raw, limit_low_raw, limit_high_raw):
        """
        功能：二维数据的线性界限值评判函数
        1. 二维结果数据的Y轴
        2. 二维结果数据的X轴（可能是正序也可能是倒序），通常为转速
        3. 二维界限值的X轴（可能是正序也可能是倒序），通常为转速
        4. 二维界限值的上限值
        5. 二维界限值的下限值
        返回：
        1. 界限值评判结果（0表示不通过，1表示通过）
        """
        # 判断下限是否存在
        if len(limit_low_raw) > 0:
            limit_low_existed = 1
        else:
            limit_low_existed = 0
        # 判断上限是否存在
        if len(limit_high_raw) > 0:
            limit_high_existed = 1
        else:
            limit_high_existed = 0

        if limit_low_existed == 0 and limit_high_existed == 0:
            # 上下限均不存在，直接认定界限值缺失
            return None

        # 确认X轴是否为正序（包括结果数据和界限值）
        if data_speed_raw[0] > data_speed_raw[-1]:
            # 降速段需要倒转（检测首尾的数值）
            data_speed = data_speed_raw[::-1]
            data_value = data_value_raw[::-1]
        else:
            data_value = data_value_raw
            data_speed = data_speed_raw
        # 确保limit是升序排列的（如果降序则进行倒转）
        if limit_speed_raw[0] > limit_speed_raw[-1]:
            limit_speed = limit_speed_raw[::-1]
            limit_low = limit_low_raw[::-1]
            limit_high = limit_high_raw[::-1]
        else:
            limit_speed = limit_speed_raw
            limit_low = limit_low_raw
            limit_high = limit_high_raw

        # 是否进行了界限值评判，初始值为0
        counter_of_enterance = 0

        for i in range(0, len(limit_speed) - 1):
            for k in range(0, len(data_speed) - 1):
                if limit_speed[i] == data_speed[k]:
                    # 如果有对应的转速点则直接比较
                    if limit_low_existed:
                        if data_value[k] < limit_low[i]:
                            # 低于下限，不合格，并直接return，不再进行其他点的评判
                            return under_low_limit
                    if limit_high_existed:
                        if data_value[k] > limit_high[i]:
                            # 高于上限，不合格，并直接return，不再进行其他点的评判
                            return over_high_limit
                    counter_of_enterance += 1

                elif limit_speed[i] < data_speed[k] < limit_speed[i + 1]:
                    # 如果结果数据的X在界限值中不存在，则生成相应的数据点的上下限再进行比较
                    if limit_low_existed:
                        generated_low_limit_point = generate_point(data_speed[k], limit_speed[i],
                                                                   limit_speed[i + 1], limit_low[i],
                                                                   limit_low[i + 1])
                        if data_value[k] < generated_low_limit_point:
                            # 低于下限则认为不合格
                            return under_low_limit
                    if limit_high_existed:
                        generated_high_limit_point = generate_point(data_speed[k], limit_speed[i],
                                                                    limit_speed[i + 1], limit_high[i],
                                                                    limit_high[i + 1])
                        if data_value[k] > generated_high_limit_point:
                            # 高于上限则认为不合格
                            return over_high_limit
                    counter_of_enterance += 1
                elif data_speed[k] < limit_speed[i] < data_speed[k + 1]:
                    # 如果界限值的X在结果数据中不存在，则生成相应的结果数据点再进行比较
                    generated_data_point = generate_point(limit_speed[i], data_speed[k],
                                                          data_speed[k + 1], data_value[k],
                                                          data_value[k + 1])
                    if limit_low_existed:
                        if generated_data_point < limit_low[i]:
                            # 如果生成的数据点超出界限值则认为不通过
                            return under_low_limit
                    if limit_high_existed:
                        if generated_data_point > limit_high[i]:
                            # 如果生成的数据点超出界限值则认为不通过
                            return over_high_limit
                    counter_of_enterance += 1

        # to compare the last point in data（对最后一个数据点进行评判）
        if limit_speed[-1] == data_speed[-1]:
            # 刚好界限值与结果数据的最后一个数据的X一致
            if limit_low_existed:
                if data_value[-1] < limit_low[-1]:
                    # 低于下限则判为不合格
                    return under_low_limit
            if limit_high_existed:
                if data_value[-1] > limit_high[-1]:
                    # 高于上限则判为不合格
                    return over_high_limit
            counter_of_enterance += 1
        else:
            # 应该倒序遍历更合理（因为是找末尾的点）
            for i in range(-1, len(limit_speed) - 1, -1):
                if limit_speed[i] > data_speed[-1]:
                    # 进行插值的前提是界限值的最后一个点在数据点的后面，基于数据点的x生成上下限
                    if limit_low_existed:
                        generated_low_limit_point = generate_point(data_speed[-1], limit_speed[i - 1],
                                                                   limit_low[i - 1], limit_speed[i],
                                                                   limit_low[i])
                        if data_value[-1] < generated_low_limit_point:
                            # 最后一个数据点低于下限也是不合格
                            return under_low_limit
                    if limit_high_existed:
                        generated_high_limit_point = generate_point(data_speed[-1], limit_speed[i - 1],
                                                                    limit_high[i - 1], limit_speed[i],
                                                                    limit_high[i])
                        if data_value[-1] < generated_high_limit_point:
                            # 最后一个数据点超出上下限也是不合格
                            return over_high_limit
                    counter_of_enterance += 1

        if counter_of_enterance == 0:
            # 如果未进行界限值评判（即结果数据和界限值无重叠部分），则认为是通过
            return in_limit
        # 可以执行到这一步说明是合格的
        return in_limit

    def oned_data_comparator(data_value, limit_low, limit_high, rms_flag=0):
        """
        功能：进行一维结果界限值评判
        输入：
        1. 一维结果数值
        2. 上限
        3. 下限
        返回：
        1. 一维评判结果
        """
        if limit_low is None:
            # 不存在下限
            if limit_high is None:
                # 不存在上限（也不存在下限
                return None
            else:
                # 存在上限（不存在下限）
                if data_value > limit_high:
                    # 高于上限
                    return over_high_limit
                else:
                    # 低于上限
                    return in_limit
        else:
            if limit_high is None:
                # 存在下限（但是不存在上限）
                if data_value < limit_low:
                    # 低于下限
                    if rms_flag:
                        # 如果是rms
                        return under_rms
                    else:
                        return under_low_limit
                else:
                    # 高于下限
                    return in_limit
            else:
                # 上下限均存在
                if limit_low <= data_value <= limit_high:
                    # 在界限值内
                    return in_limit
                elif data_value > limit_high:
                    # 高于上限
                    return over_high_limit
                else:
                    # 低于下限
                    if rms_flag:
                        return under_rms
                    else:
                        return under_low_limit

    def area_comparator(data_value, data_speed, limit_low_area, limit_high_area):
        """
        功能： 进行区域面积界限值比较（针对二维数据）
        输入：
        1. 二维结果数据的Y
        2. 二维结果数据的X
        3. 区域面积的下限
        4. 区域面积的上限
        返回：
        1. 区域面积的评判结果（0表示不通过，1表示通过）
        """
        # 区域面积值（用于累加）
        area_sum = 0

        # 计算区域面积结果
        for i in range(0, len(data_speed) - 1):
            area_sum += (abs(data_speed[i + 1] - data_speed[i])) * (
                    (data_value[i + 1] + data_value[i]) / 2)
        result_of_compare = oned_data_comparator(area_sum, limit_low_area, limit_high_area)
        if result_of_compare is not None:
            return result_of_compare
        else:
            return without_limit

    def oned_comparator(data, limit, index_sensor_data, index_sensor_limit, index_test_data,
                        index_test_limit):
        """
        功能：一维指标评判
        输入：
        1. 结果数据（所有信息）
        2. 界限值（所有信息）
        3. 结果数据中的传感器索引
        4. 结果数据中的测试段索引
        5. 界限值中的传感器索引
        6. 界限值中的测试段索引
        """
        # 初始化评判结果(合格）
        result_of_compare = list()

        # 获取结果数据和界限值的一维指标
        if "onedData" in data["resultData"][index_sensor_data]["dataSection"][index_test_data].keys():
            if data["resultData"][index_sensor_data]["dataSection"][index_test_data]["onedData"] is not None:
                result_length = len(data["resultData"][index_sensor_data]["dataSection"][index_test_data][
                                                       "onedData"])
            else:
                result_length = 0
        else:
            result_length = 0

        if "onedData" in limit["resultData"][index_sensor_limit]["dataSection"][index_test_limit].keys():
            if limit["resultData"][index_sensor_limit]["dataSection"][index_test_limit]["onedData"] is not None:
                limit_length = len(limit["resultData"][index_sensor_limit]["dataSection"][index_test_limit][
                                    "onedData"])
            else:
                limit_length = 0
        else:
            limit_length = 0
        # 存在一维结果数据和界限值才进行评判
        if limit_length > 0:
            # 获取界限值的指标列表
            limit_indicator_name_list = [dic['name'] for dic in
                                         limit["resultData"][index_sensor_limit]["dataSection"][
                                             index_test_limit][
                                             "onedData"]]
        else:
            limit_indicator_name_list = list()
        if result_length > 0:
            for index_dataInd in range(0, result_length):
                # 遍历所有指标（结果数据）
                data_name = data["resultData"][index_sensor_data]["dataSection"][index_test_data][
                    "onedData"][index_dataInd]["name"]
                if data_name in limit_indicator_name_list:
                    # 存在该指标则进行评判
                    # 直接根据指标名索引到对应的界限值
                    index_limitInd = limit_indicator_name_list.index(
                        data["resultData"][index_sensor_data]["dataSection"][index_test_data][
                            "onedData"][index_dataInd]["name"])
                    # 获取结果数据里的一维
                    data_value = data["resultData"][index_sensor_data]["dataSection"][index_test_data][
                                    "onedData"][index_dataInd]["value"]

                    limit_keys = limit["resultData"][index_sensor_limit]["dataSection"][index_test_limit][
                                "onedData"][index_limitInd].keys()
                    if 'high' in limit_keys:
                        # 如果存在上限，则进行上限评判
                        high_limit = limit["resultData"][index_sensor_limit]["dataSection"][index_test_limit][
                                "onedData"][index_limitInd]["high"]
                    else:
                        # 不存在界限值则为空
                        high_limit = None
                    if 'low' in limit_keys:
                        # 如果存在上限，则进行上限评判
                        low_limit = limit["resultData"][index_sensor_limit]["dataSection"][index_test_limit][
                                "onedData"][index_limitInd]["low"]
                    else:
                        low_limit = None
                    # 进行界限值评判（先进行rms)确认
                    if "rms" == data_name.lower():
                        rms_flag = 1
                    else:
                        rms_flag = 0
                    # 进行一维指标评判
                    compare_res = oned_data_comparator(data_value, low_limit, high_limit, rms_flag)
                    if compare_res is not None:
                        # 存在评判结果（即界限值存在），则更新评判结果
                        data["resultData"][index_sensor_data]["dataSection"][index_test_data][
                            "onedData"][index_dataInd]["indicatorDiagnostic"] = compare_res
                        if compare_res in list([under_rms, under_low_limit, over_high_limit]):
                            data["qdaqDefectDescription"].append(data_name)
                    else:
                        # 不存在界限值则保持初始值
                        pass
                else:
                    # 不存在相同的指标就直接跳过
                    pass
                # 记录每个指标的评判结果（用于最终的结果）
                result_of_compare.append(
                    data["resultData"][index_sensor_data]["dataSection"][index_test_data]["onedData"][
                        index_dataInd]["indicatorDiagnostic"])
        # 更新该测试段一维指标的评判的结果
        oned_result = get_final_result(result_of_compare)

        return oned_result

    def twod_comparator(data, limit, index_sensor_data, index_sensor_limit, index_test_data,
                        index_test_limit, name_of_data, line_trigger, area_trigger):
        """
        功能：进行二维指标评判（包括线性和区域比较）
        输入：
        1. 结果数据
        2. 界限值
        3. 结果数据的传感器索引
        4. 结果数据的测试段索引
        5. 界限值的传感器索引
        6. 界限值的测试段索引
        7. 指标类名称，如twodTD，twodOC等
        8. 是否开启线性评判
        9. 是否开启区域面积评判
        返回：
        1. 比较结果
        """
        # 初始化结果（合格）
        result_of_compare = list()
        result_length = 0

        if line_trigger or area_trigger:
            # 获取结果数据的二维指标
            if name_of_data in data["resultData"][index_sensor_data]["dataSection"][index_test_data].keys():
                if data["resultData"][index_sensor_data]["dataSection"][index_test_data][name_of_data] is not None:
                    result_length = len(data["resultData"][index_sensor_data]["dataSection"][index_test_data][
                                            name_of_data])
                else:
                    result_length = 0
            else:
                result_length = 0
        # 线性比较
        if line_trigger:
            # 需要进行线性界限值比较
            if name_of_data in limit["resultData"][index_sensor_limit]["dataSection"][index_test_limit]["lineComp"].keys():
                if limit["resultData"][index_sensor_limit]["dataSection"][index_test_limit]["lineComp"][name_of_data] is not None:
                    limit_length = len(
                        limit["resultData"][index_sensor_limit]["dataSection"][index_test_limit][
                            "lineComp"][name_of_data])
                else:
                    limit_length = 0
            else:
                limit_length = 0
            if limit_length > 0:
                limit_indicator_name_list = [dic['yName'] for dic in
                                             limit["resultData"][index_sensor_limit]["dataSection"][
                                                 index_test_limit]["lineComp"][name_of_data]]
            else:
                limit_indicator_name_list = list()
            if result_length > 0:
                # 存在结果数据和界限值才进行评判
                for index_data_indicator in range(0, result_length):
                    # 遍历并获取指标名（二维）
                    data_name = data["resultData"][index_sensor_data]["dataSection"][index_test_data][
                                name_of_data][index_data_indicator]['yName']
                    if data_name in limit_indicator_name_list:
                        # 存在该指标则进行界限值评判
                        # 根据指标名称找到对应的
                        index_limit_indicator = limit_indicator_name_list.index(data_name)
                        # 读取结果数据里的二维结果（x和y）
                        data_value = \
                            data["resultData"][index_sensor_data]["dataSection"][index_test_data][
                                name_of_data][index_data_indicator]["yValue"]
                        data_speed = \
                            data["resultData"][index_sensor_data]["dataSection"][index_test_data][
                                name_of_data][index_data_indicator]["xValue"]
                        # 读取上下限（不存在的时候是空列表）
                        limit_speed = limit["resultData"][index_sensor_limit]["dataSection"][index_test_limit]["lineComp"][name_of_data][index_limit_indicator]["xValue"]
                        limit_low = limit["resultData"][index_sensor_limit]["dataSection"][index_test_limit]["lineComp"][name_of_data][index_limit_indicator]["yLow"]
                        limit_high = limit["resultData"][index_sensor_limit]["dataSection"][
                                    index_test_limit]["lineComp"][name_of_data][index_limit_indicator]["yHigh"]

                        # 进行界限值比较
                        line_res = line_comparator(data_value, data_speed, limit_speed,
                                                   limit_low, limit_high)
                        if line_res is not None:
                            # 表示评判结果已更新
                            data["resultData"][index_sensor_data]["dataSection"][index_test_data][
                                name_of_data][
                                index_data_indicator][
                                "indicatorDiagnostic"] = line_res
                        else:
                            # 表示传入的界限值为空列表（即不存在上下限），保留初始值
                            pass
                        if line_res in list([under_low_limit, over_high_limit]):
                            # 记录超限的指标
                            data["qdaqDefectDescription"].append(data_name)
                    else:
                        # 没有对应指标的界限值则直接跳过（意味着保留初始值）
                        pass
                    # 记录比较结果
                    result_of_compare.append(data["resultData"][index_sensor_data]["dataSection"][index_test_data][
                                name_of_data][
                                index_data_indicator][
                                "indicatorDiagnostic"])
            else:
                pass

        # 区域面积比较
        # TODO：目前该功能并未在SigMA端开放，后期可能需要继续测试更新
        if area_trigger:
            # 需要进行区域面积界限值比较
            if name_of_data in limit["resultData"][index_sensor_limit]["dataSection"][index_test_limit][
                "lineComp"].keys():
                if limit["resultData"][index_sensor_limit]["dataSection"][index_test_limit]["areaComp"][
                    name_of_data] is not None:
                    limit_length = len(
                        limit["resultData"][index_sensor_limit]["dataSection"][index_test_limit][
                            "areaComp"][name_of_data])
                else:
                    limit_length = 0
            else:
                limit_length = 0
            if limit_length > 0:
                limit_indicator_name_list = [dic['yName'] for dic in
                                             limit["resultData"][index_sensor_limit]["dataSection"][
                                                 index_test_limit]["areaComp"][name_of_data]]
            else:
                limit_indicator_name_list = list()
            if result_length > 0:
                # 存在结果数据和界限值才进行评判
                for index_data_indicator in range(0, result_length):
                    # 遍历并获取指标名（二维）
                    data_name = data["resultData"][index_sensor_data]["dataSection"][index_test_data][
                        name_of_data][index_data_indicator]['yName']
                    if data_name in limit_indicator_name_list:
                        # 存在该指标则进行界限值评判
                        # 根据指标名称找到对应的界限值的索引
                        index_limit_indicator = limit_indicator_name_list.index(data_name)
                        # 读取结果数据里的二维结果（x和y）
                        data_value = \
                            data["resultData"][index_sensor_data]["dataSection"][index_test_data][
                                name_of_data][index_data_indicator]["yValue"]
                        data_speed = \
                            data["resultData"][index_sensor_data]["dataSection"][index_test_data][
                                name_of_data][index_data_indicator]["xValue"]
                        limit_keys = \
                        limit["resultData"][index_sensor_limit]["dataSection"][index_test_limit]["areaComp"][name_of_data][index_limit_indicator].keys()
                        # 读取上下限（不存在时为空）
                        if "low" in limit_keys:
                            limit_low = limit["resultData"][index_sensor_limit]["dataSection"][index_test_limit]["areaComp"][name_of_data][index_limit_indicator]['low']
                        else:
                            limit_low = None
                        if "high" in limit_keys:
                            limit_high = \
                            limit["resultData"][index_sensor_limit]["dataSection"][index_test_limit][
                                "areaComp"][name_of_data][index_limit_indicator]['high']
                        else:
                            limit_high = None
                        # 进行界限值比较
                        area_res = area_comparator(data_value, data_speed, limit_low, limit_high)

                    else:
                        # 没有对应指标则意味着该指标为界限值缺失状态
                        area_res = without_limit
                    result_of_compare.append(area_res)
            else:
                # 不存在界限值和结果数据
                pass
        # 结合不同指标不同方式的评判结果
        return_result = get_final_result(result_of_compare)

        return return_result

    def turned_off_compare_oned(data, test_data_index):
        """
        功能：如果设置为不比较界限值，则需要更新掉每个指标的评判结果
        输入：
        1. 结果数据
        2. 结果数据的制定出测试段的索引
        返回：
        1. 更新后的结果数据
        """
        for index_sensor_data in range(0, len(data["resultData"])):
            if data["resultData"][index_sensor_data]["dataSection"][test_data_index]["onedData"] is not None:
                data["resultData"][index_sensor_data]["dataSection"][test_data_index]["results"][
                    "onedData"] = without_comp
                for indexOnedData in range(0, len(
                        data["resultData"][index_sensor_data]["dataSection"][test_data_index][
                            "onedData"])):
                    data["resultData"][index_sensor_data]["dataSection"][test_data_index]["onedData"][
                        indexOnedData][
                        "indicatorDiagnostic"] = without_comp
            else:
                # 不存在一维指标直接跳过
                pass
        return data

    def turned_off_compare_twod(data, name_of_data, test_data_index):
        """
        功能：如果设置为不比较界限值，则需要更新掉每个指标的评判结果
        输入：
        1. 结果数据
        2. 结果数据的制定出测试段的索引
        返回：
        1. 更新后的结果数据
        """
        for index_sensor_data in range(0, len(data["resultData"])):
            data["resultData"][index_sensor_data]["dataSection"][test_data_index]["results"][
                name_of_data] = without_comp
            if data["resultData"][index_sensor_data]["dataSection"][test_data_index][
                name_of_data] is not None:
                for index_data_indicator in range(0, len(
                        data["resultData"][index_sensor_data]["dataSection"][test_data_index][
                            name_of_data])):
                    data["resultData"][index_sensor_data]["dataSection"][test_data_index][name_of_data][
                        index_data_indicator][
                        "indicatorDiagnostic"] = without_comp
        return data

    def update_final_result(data):
        """
        功能：更新测试段和传感器以及最终结果
        输入：
        1. 结果数据
        返回：
        1. 更新后的结果数据
        """
        overall_result = list()
        for index_sensor_data in range(0, len(data["resultData"])):
            # 遍历传感器
            result_by_sensor = list()
            for index_test in range(0, len(data["resultData"][index_sensor_data]["dataSection"])):
                # 遍历测试段
                result_by_test = list()
                result_by_test.append(
                    data["resultData"][index_sensor_data]["dataSection"][index_test]["results"][
                        "onedData"])
                result_by_test.append(
                    data["resultData"][index_sensor_data]["dataSection"][index_test]["results"][
                        "twodOC"])
                result_by_test.append(
                    data["resultData"][index_sensor_data]["dataSection"][index_test]["results"][
                        "twodTD"])
                result_by_test.append(
                    data["resultData"][index_sensor_data]["dataSection"][index_test]["results"][
                        "twodOS"])
                result_by_test.append(
                    data["resultData"][index_sensor_data]["dataSection"][index_test]["results"][
                        "twodCeps"])
                # 更新测试段评判结果
                _result_by_test = get_final_result(result_by_test)
                data["resultData"][index_sensor_data]["dataSection"][index_test]["testResult"] = _result_by_test
                result_by_sensor.append(_result_by_test)
            # 更新每个传感器的评判结果(基于测试段结果）
            _result_by_sensor = get_final_result(result_by_sensor)
            data["resultData"][index_sensor_data]["resultBySensor"] = _result_by_sensor
            overall_result.append(_result_by_sensor)
        # 返回最终评判结果
        return get_final_result(overall_result)

    def oned_controller(data, limit, index_sensor_data, index_sensor_limit, index_test_data,
                        index_test_limit, trigger_on_off):
        """
        功能：一维指标评判
        输入：
        1. 结果数据
        2. 界限值
        3. 传感器索引（结果数据）
        4. 传感器索引（界限值）
        5. 测试段索引（结果数据）
        6. 测试段索引（界限值）
        7. 是否进行一维界限值评判（1表示开启，0表示不开启）
        返回：
        1. 评判结果
        """
        if trigger_on_off:
            # 进行一维界限值评判
            result_of_compare = oned_comparator(data, limit, index_sensor_data, index_sensor_limit,
                                                index_test_data, index_test_limit)
        else:
            # to make all the diagnostic to be 1 if not to compare
            turned_off_compare_oned(data, index_test_data)
            result_of_compare = without_comp

        return result_of_compare

    def twod_controller(data, limit, index_sensor_data, index_sensor_limit, index_test_data,
                        index_limit_data, trigger_on_off, line_trigger, area_trigger, name_of_data):
        """
        功能：二维界限值评判
        输入:
        1. 结果数据
        2. 界限值
        3. 传感器索引（结果数据）
        4. 传感器索引（界限值）
        5. 测试段索引（结果数据）
        6. 测试段索引（界限值）
        7. 是否进行该类型二维界限值评判（1表示开启，0表示不开启）
        8. 是否进行线性评判（1表示开启，0表示不开启）
        9. 是否进行区域面积评判（1表示开启，0表示不开启）
        10. 二维数据的名称，包括twodTD, twodOC, twodOS, twodCeps
        返回：
        1. 评判结果
        """
        if trigger_on_off:
            # turn on this 2d compare
            result_of_compare = twod_comparator(data, limit, index_sensor_data, index_sensor_limit,
                                                index_test_data, index_limit_data,
                                                name_of_data, line_trigger, area_trigger)
        else:
            # turn off this 2d compare
            turned_off_compare_twod(data, name_of_data, index_test_data)
            result_of_compare = without_comp
        data["resultData"][index_sensor_data]["dataSection"][index_test_data]["results"][
            name_of_data] = result_of_compare
        return result_of_compare

    def same_test_finder(data, limit, index_sensor_data, index_sensor_limit, test_name):
        """

        """
        if data["resultData"][index_sensor_data]["dataSection"] is not None:
            test_length_data = len(data["resultData"][index_sensor_data]["dataSection"])
        else:
            test_length_data = 0
        if limit["resultData"][index_sensor_limit]["dataSection"] is not None:
            test_length_limit = len(limit["resultData"][index_sensor_limit]["dataSection"])
        else:
            test_length_limit = 0
        if test_length_data > 0 and test_length_limit > 0:
            # 存在测试段（数据和界限值）
            data_test_name_list = [dic['testName'] for dic in
                                   data["resultData"][index_sensor_data]["dataSection"]]
            limit_test_name_list = [dic['testName'] for dic in
                                    limit["resultData"][index_sensor_limit]["dataSection"]]
            if test_name in data_test_name_list:
                index_test_data = data_test_name_list.index(test_name)
                if test_name in limit_test_name_list:
                    index_test_limit = limit_test_name_list.index(test_name)
                    # 确认是否进行超限比计算
                    if limit_compare_info["overLimit"]:
                        # over limit calculation based on twodOS and update the value in onedData
                        over_limit_value = over_limit_calc(data, limit, index_sensor_data,
                                                           index_sensor_limit,
                                                           index_test_data,
                                                           index_test_limit, ond_os_calc_info)
                        if data["resultData"][index_sensor_data]["dataSection"][index_test_data][
                            "onedData"][0]['name'] == 'OLR':
                            # 如果存在直接赋值
                            data["resultData"][index_sensor_data]["dataSection"][index_test_data][
                                "onedData"][0][
                                'value'] = over_limit_value
                        else:
                            # 不存在则在第一个位置插入该指标
                            olr_value = {'name': 'OLR', 'unit': "", 'value': over_limit_value,
                                         "indicatorDiagnostic": without_limit}
                            data["resultData"][index_sensor_data]["dataSection"][index_test_data][
                                "onedData"].insert(0,
                                                   olr_value)
                    # 超限比计算完成后将纳入一维指标并进行界限值评判
                    onedData = oned_controller(data, limit, index_sensor_data, index_sensor_limit,
                                               index_test_data,
                                               index_test_limit,
                                               limit_compare_info['oned'])
                    # 更新结果
                    data["resultData"][index_sensor_data]["dataSection"][index_test_data]["results"][
                        "onedData"] = onedData

                    for name_of_data in ["twodTD", "twodOC", "twodOS", "twodCeps"]:

                        twod_result = twod_controller(data, limit, index_sensor_data, index_sensor_limit,
                                             index_test_data, index_test_limit,
                                             limit_compare_info[name_of_data], limit_compare_info["line"],
                                             limit_compare_info["area"], name_of_data)
                        data["resultData"][index_sensor_data]["dataSection"][index_test_data]["results"][name_of_data] = twod_result
        else:
            pass

    def same_sensor_finder(data, limit, test_name):
        """
        功能：寻找相同的传感器（结果数据和界限值）
        输入：
        1. 结果数据
        2. 界限值
        返回：无
        """
        if data["resultData"] is not None:
            sensor_length_data = len(data["resultData"])
        else:
            sensor_length_data = 0
        if limit["resultData"] is not None:
            sensor_length_limit = len(limit["resultData"])
        else:
            sensor_length_limit = 0
        for index_sensor_data in range(0, sensor_length_data):
            for index_sensor_limit in range(0, sensor_length_limit):
                if str(data["resultData"][index_sensor_data]["sensorId"]) == str(
                        limit["resultData"][index_sensor_limit]["sensorId"]):
                    data["resultData"][index_sensor_data]["limitVersion"] = \
                        limit["resultData"][index_sensor_limit]["limitVersion"]
                    # 找到相同传感器的结果数据和界限值，则开始确认测试段
                    same_test_finder(data, limit, index_sensor_data, index_sensor_limit, test_name)
        # 返回最终评判结果
        return update_final_result(data)

    def compare_controller(data, limit, test_name):
        """
        function: to record the basic info of limit into data
        :param
            data(dict): overall test result
            limit(dict): overall limit
        :return
            data(dict): updated data(with final result)
        """
        if limit is not None:
            # 进行界限值评判
            data["overallResult"] = same_sensor_finder(data, limit, test_name)
            return data
        else:
            return data

    def dict_for_twod_area_res(data, index_sensor_data, index_test_data, name_of_data, index_indicator):
        """
        功能：计算区域面积结果并放到一维指标中
        输入：
        1. 结果数据（包含全部信息）
        2. 传感器索引
        3. 测试段索引
        4. 结果数据类型，包括twodTD, twodOC, twodOS and Ceps
        5. 指标索引
        返回：
        1. 一维指标（标准格式）
        """
        # TODO: 确认该指标的逻辑和意义，目前并未应用
        # 获取数据的X和Y
        data_value = \
            data["resultData"][index_sensor_data]["dataSection"][index_test_data][name_of_data][
                index_indicator]["yValue"]
        data_speed = \
            data["resultData"][index_sensor_data]["dataSection"][index_test_data][name_of_data][
                index_indicator][
                "xValue"]
        # 初始值为0
        area_sum = 0
        # 求整个二维曲线的面积结果
        for i in range(0, len(data_speed) - 1):
            area_sum += (abs(data_speed[i + 1] - data_speed[i])) * (
                    (data_value[i + 1] + data_value[i]) / 2)

        one_indicator = \
            {"name": data["resultData"][index_sensor_data]["dataSection"][index_test_data][name_of_data][
                        index_indicator]["yName"] + "_Area",
             "value": area_sum,
             "unit": "area",
             "indicatorDiagnostic": without_limit}

        return one_indicator

    def add_twod_data_area_to_oned(data, test_name, name_of_data):
        """
        功能：将二维指标的区域面积结果加入一维指标中
        输入：
        1. 结果数据
        2. 测试段名称
        3. 二维数据名称
        4. 是否进行区域面积计算
        返回: 结果数据
        """
        # TODO：区域面积暂时不对外开放，故而暂不考虑该功能
        try:
            for index_sensor_data in range(0, len(data["resultData"])):
                data_test_name_list = [dic['testName'] for dic in
                                       data["resultData"][index_sensor_data]["dataSection"]]
                if test_name in data_test_name_list:
                    index_test_data = data_test_name_list.index(test_name)
                    if data["resultData"][index_sensor_data]["dataSection"][index_test_data][name_of_data] is not None:
                        for index_data_indicator in range(0, len(
                                data["resultData"][index_sensor_data]["dataSection"][index_test_data][
                                    name_of_data])):
                            data["resultData"][index_sensor_data]["dataSection"][index_test_data][
                                "onedData"].append(
                                dict_for_twod_area_res(data, index_sensor_data, index_test_data, name_of_data, index_data_indicator))
        except Exception:
            pass

        return data

    def over_limit_calc(data, limit, index_sensor_data, index_sensor_limit, index_test, index_limit,
                        oned_os_calc_info):
        """
        功能：计算超限比
        输入：
        1. 结果数据
        2. 界限值
        3. 结果数据的传感器索引，获取指定传感器的数据
        4. 界限值的传感器索引，获取指定传感器的界限值
        5. 结果数据的测试段索引，获取指定测试段的数据
        6. 界限值的测试段索引，获取指定测试段的界限值
        7. 一维关注阶次参数信息，用于超限比计算时剔除关注阶次
        返回：超限比结果值（一维）
        :func: calculate the over limit indicator of target test
        twod_os: order spectrum of target test
        twod_os: limit of target order spectrum
        oned_os_calc_info: targat oned order, can read from config file
        """
        # 初始化超限比值
        over_limit_value = 0
        target_order_spectrum = data["resultData"][index_sensor_data]["dataSection"][index_test][
            'twodOS']

        if target_order_spectrum:
            # 防止数据为空
            if "twodOS" in limit["resultData"][index_sensor_limit]["dataSection"][index_limit]["lineComp"].keys():
                target_order_spectrum_limit = \
                    limit["resultData"][index_sensor_limit]["dataSection"][index_limit][
                        "lineComp"]["twodOS"]
                if target_order_spectrum_limit:
                    # 防止界限值为空
                    try:
                        # 剔除关注阶次及邻近的点（包括阶次谱和界限值）
                        remove_index = list()
                        side_num = oned_os_calc_info['pointNum'] // 2
                        for order in oned_os_calc_info['orderList']:
                            for suborder in order:
                                # 确定需要剔除的阶次的索引（结合阶次谱分辨率确定）
                                target_order_index = round(suborder / (
                                        target_order_spectrum[0]['xValue'][1] -
                                        target_order_spectrum[0]['xValue'][0]))
                                # 记录所要剔除阶次的索引信息
                                remove_index.extend(
                                    list(range(target_order_index - side_num,
                                               target_order_index + side_num + 1)))
                        # 根据待剔除阶次的索引踢掉阶次谱中对应数据点
                        new_os_data_y = np.delete(target_order_spectrum[0]['yValue'], remove_index)
                        # 根据待剔除阶次的索引踢掉阶次谱线性上限中对应数据点
                        new_os_limit_y = np.delete(target_order_spectrum_limit[0]['yHigh'],
                                                   remove_index)
                        # 计算平方差
                        condition = new_os_data_y > new_os_limit_y
                        power_array = np.power(new_os_data_y[condition], 2) - np.power(
                            new_os_limit_y[condition], 2)
                        # 计算超限比
                        over_limit_value = np.sum(power_array) / np.sum(np.power(new_os_limit_y, 2))
                    except Exception:
                        traceback.print_exc()
                        logging.warning(
                            "over limit calculation failed, failed msg:" + traceback.format_exc())
                else:
                    logging.warning("no twodOS line limit")
            else:
                logging.warning("no twodOS line limit")
        else:
            logging.warning("no twodOS data")

        return over_limit_value

    # add twodOC_area to oned SA
    if limit_compare_info['area']:
        add_twod_data_area_to_oned(data, test_name, "twodOC")

    # the main function of limit compare SA
    compare_controller(data, limit, test_name)

    return data


if __name__ == '__main__':
    # 以下为代码测试部分
    # 记录出错信息
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(pathname)s %(funcName)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='[%Y-%m-%d %H:%M:%S]',
        filename='D:/qdaq/Log/temp.log',
        filemode='a'
    )
    try:
        from parameters import Parameters
        import os
        import matplotlib.pyplot as plt

        # 读取配置文件
        filepath = r'D:\qdaq\test\210825-1\no_limit'

        config_filename = os.path.join(filepath, 'test_paramReceived.json')
        param = Parameters(config_filename)

        # 读取结果数据
        file_path_data = os.path.join(filepath,
                                      'test_raw_result.json')
        with open(file_path_data, 'r') as f:
            data = json.load(f)
        # 读取界限值
        file_path_limit = os.path.join(filepath, 'test_limitReceived.json')
        with open(file_path_limit, 'r') as f:
            limit = json.load(f)

        # 记录时间，检验界限值评判所需要的的时间
        start = time.time()
        testName = param.speedRecogInfo['testName']
        for test_name in testName[:]:
            print(test_name)
            data = comparator(data, limit, test_name, param.limitCompareFlag, param.onedOSCalcInfo)
        print('compare time: ', time.time() - start)

        target_limit = limit['resultData'][0]['dataSection'][0]['lineComp']['twodTD'][0]
        target_data = data['resultData'][0]['dataSection'][0]['twodTD'][0]
        plt.plot(target_limit['xValue'], target_limit['yHigh'], 'r')
        plt.plot(target_limit['xValue'], target_limit['yLow'], 'b')
        plt.plot(target_data['xValue'], target_data['yValue'], 'k')
        plt.title(target_data['yName'])
        plt.xlabel('Speed/rpm')
        plt.ylabel('Amp')
        plt.legend(['high limit', 'low limit', 'result data'])
        plt.show()
        file_path_output = os.path.join(filepath, 'test_limit_new.json')
        with open(file_path_output, "w") as write_file:
            json.dump(data, write_file, indent=4)
    except Exception:
        traceback.print_exc()
        logging.warning("exec failed, failed msg:" + traceback.format_exc())
        sys.exit()
