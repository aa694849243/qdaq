# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:27:59 2020

@author: Wall@Synovate

function: define the parameters that used in the whole software
update: 2021-5-31 17:12
"""

import logging
import traceback
import sys
import json
import os
import numpy as np
import pprint
from datetime import datetime, timedelta


def time_get():
    """
    功能：用于获取时间戳（当前时间），测试开始时的时间
    返回：时间戳，datetime类型
    """
    # get start time
    timestamp = datetime.now()
    return timestamp


def time_convert(timestamp):
    """
    功能：转换时间戳为特定格式，用于confirm_target_folder函数，包括：
    1. 年月（%y%m），用于创建Data和Report下的目录结构（类型的下级目录）
    2. 年月日时（%y%m%d%H），用于创建Data和Report下的目录结构（年月的下级目录）
    3. 年月日时分秒（%y%m%d%H%M%S），用于文件名的时间戳信息
    输入：时间戳
    返回：3类时间字符串
    """
    # get target format time string
    ym_time = timestamp.strftime("%y%m")
    ymdh_time = timestamp.strftime("%y%m%d%H")
    ymdhms_time = timestamp.strftime("%y%m%d%H%M%S")
    return ym_time, ymdh_time, ymdhms_time


def single_folder_confirm(folder_path):
    """
    功能：确认文件路径是否存在，不存在则新建
    输入：文件目录
    返回：无
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def confirm_target_folder(timestamp, folder_info, basic_info):
    """
    function: to check if target folder have Rawdata, Report, JSON_NetError, JSON_InfoMissing
    功能：确认目标路径下是否存在对应的目录结构，若不存在则新建包括：
    1. 原始数据的目录结构
    2. 报告的目录结构（这里指的阶次谱彩图）
    3. 发送失败的结果数据（网络原因）
    4. 发送失败的结果数据（内容错误）
    输入：
    1. 时间戳，这里就是测试开始的时间
    2. 文件目录信息
    3. 该产品基本信息，主要指的是产品类型
    返回：
    1. 原始数据存放目录信息
    2. 阶次彩图数据存档目录信息
    3. 文件名时间戳信息
    """
    # 时间统一制式为utc时间，故而需要减去8小时切换为utc时间
    ym_t, ymdh_t, ymdhms_t = time_convert(timestamp - timedelta(hours=8))
    # confirm all the target folder already existed
    # 确认第一层目录结构
    for _, target_folder in folder_info.items():
        single_folder_confirm(target_folder)
    # 确认原始数据存放的目录结构
    raw_data_type_folder = os.path.join(folder_info['rawData'], basic_info['type'])
    target_field1 = os.path.join(raw_data_type_folder, ym_t)
    target_field1_1 = os.path.join(target_field1, ymdh_t)
    single_folder_confirm(target_field1_1)
    # 确认宝阶次谱彩图数据的存放目录结构
    report_data_type_folder = os.path.join(folder_info['reportData'], basic_info['type'])
    target_field2 = os.path.join(report_data_type_folder, ym_t)
    target_field2_1 = os.path.join(target_field2, ymdh_t)
    single_folder_confirm(target_field2_1)
    return target_field1_1, target_field2_1, ymdhms_t


def create_empty_final_result(timestamp, basic_info, speed_recog_info, over_limit_flag):
    """
    功能：初始化一份结果数据，用于数据分析完之后进行更新
    输入：
    1. 时间戳信息，即测试开始的时间
    2. 基本信息，包括检测系统名称，产品类型，序列号，传感器等信息
    3. 测试段信息
    返回：一份数值为空的结果数据
    function: create the empty final_result and use it in following indicators calculation(include oned,
    twodTD,twodOC,twodOS), and for the initial.
    :param
    basic_info(dict): basic information used for the final final_result
    test_name(list): one test should have one final_result, such as [drive1, coast1, constant1]
    :return
    final_result(dict): the final_result as the same format align with Server end
    """
    final_result = dict()
    final_result['systemNo'] = basic_info['systemNo']
    final_result['type'] = basic_info['type']
    final_result['serialNo'] = basic_info['serialNo']
    # put some judgement info into final result
    final_result['artificialJudgment'] = None
    final_result['artificialDefectDescription'] = None
    final_result['judgedBy'] = None
    final_result['intelligenceStatus'] = None
    final_result['intelligenceDefectDescription'] = None
    # 这里的时间为当地时间，并不是UTC时间
    final_result['time'] = datetime.strftime(timestamp, '%Y-%m-%d %H:%M:%S')
    final_result['overallResult'] = -1
    final_result['qdaqDefectDescription'] = list()
    final_result['resultData'] = list()
    for i, sensorId in enumerate(basic_info['sensorId']):
        final_result['resultData'].append(dict())
        final_result['resultData'][i]['sensorId'] = sensorId
        final_result['resultData'][i]['resultBySensor'] = -1
        # limit version的信息是与传感器一起的
        final_result['resultData'][i]['limitVersion'] = "-1"
        data_section = list()
        j = 0
        for index, testName in enumerate(speed_recog_info['testName']):
            # 跳过dummy段，即最后的结果数据不应包含dummy段结果（实际上dummy段也不参与数据分析）
            if speed_recog_info['notDummyFlag'][index]:
                data_section.append(dict())
                data_section[j]['testName'] = testName
                data_section[j]['testResult'] = -1
                data_section[j]['testSensorState'] = 'ok'
                data_section[j]['startTime'] = None
                data_section[j]['endTime'] = None
                data_section[j]['results'] = dict()
                data_section[j]['results']['onedData'] = -1
                data_section[j]['results']['twodTD'] = speed_recog_info['initial_indicator_diagnostic'][index]
                data_section[j]['results']['twodOC'] = speed_recog_info['initial_indicator_diagnostic'][index]
                data_section[j]['results']['twodOS'] = -1
                data_section[j]['results']['twodCeps'] = -1
                if over_limit_flag:
                    # 新建一维指标如果需要计算超限比
                    data_section[j]['onedData'] = [{'name': 'OLR',
                                                    'unit': "",
                                                    'value': -1,
                                                    "indicatorDiagnostic": -1
                                                    }]
                else:
                    data_section[j]['onedData'] = None
                data_section[j]['twodTD'] = None
                data_section[j]['twodOC'] = None
                data_section[j]['twodOS'] = None
                data_section[j]['twodCeps'] = None
                j += 1
        final_result['resultData'][i]['dataSection'] = data_section
    return final_result


def create_empty_twodtd(time_domain_calc_info, sensor_index, indicator_diagnostic=-1):
    """
    功能：初始化二维时间域指标，用于实时更新
    输入：
    1. 时间域指标计算配置参数信息
    2. 传感器索引，用于不同传感器信号对应的单位
    3. 初始评判结果，可以指定的值进行初始化
    返回：不包含具体数值的二维时间域结果
    function: create the empty twodTD final_result and use it in following indicators calculation(real
    time)
    :param
    time_domain_calc_info(dict): time domain indicator calculation parameters, include the indicators
    list, e.g. ['RMS','Crest','Kurtosis']
    :return
    twodtd_result(list): a list include all 2D time domain indicator information
    """
    twodtd_result = list()
    for i, indicator in enumerate(time_domain_calc_info['indicatorList']):
        twodtd_result.append({
            'xName': time_domain_calc_info['xName'],
            'xUnit': time_domain_calc_info['xUnit'],
            'xValue': list(),
            'yName': indicator,
            'yUnit': time_domain_calc_info['indicatorUnit'][sensor_index][i],
            'yValue': list(),
            "indicatorDiagnostic": indicator_diagnostic
        })
    return twodtd_result


def create_empty_twodtd_for_share(time_domain_calc_info, sensor_index, sensor_name, max_len, indicator_diagnostic=-1):
    """
    :param
    功能：初始化二维时间域指标，用于实时更新（共享内存模式）
    输入：
    1. 时间域指标计算配置参数信息
    2. 传感器索引，用于不同传感器信号对应的单位
    3. 二维时间域数据的最大长度（X和Y）
    4. 初始评判结果，可以指定的值进行初始化
    返回：不包含具体数值的二维时间域结果
    """
    twodtd_result = {}
    for info in time_domain_calc_info[sensor_name]:
        twodtd_result[tuple(info['value'])] = {
            'xName': info['xName'],
            'xUnit': info['xUnit'],
            'xValue': np.zeros(max_len),
            'yName': info['index'],
            'yUnit': info['unit'][0] if len(info['unit']) > 1 else info['unit'][sensor_index],
            'yValue': [None] * max_len,
            "indicatorDiagnostic": indicator_diagnostic
        }
    return twodtd_result


def create_empty_temptd():
    """
    功能：创建数据结构用于保存时间域计算中的临时数据（最终用于一维时间域指标计算），每计算一段信号就会有一个对应的值
    返回：字典型数据包括：
    1. 平方和值xi2
    2. 最大值xmax
    3. 均值xmean
    4. 三次方和值xi3
    5. 四次方和值xi4
    6. A计权的平方值xi2_A
    function: create empty list to store the temp value for 1d time domain indicators
    :return
    temptd_result(dict): dict to store temp td value
    """
    temptd_result = dict()
    temptd_result['xi2'] = list()
    temptd_result['xmax'] = list()
    temptd_result['xmean'] = list()
    temptd_result['xi3'] = list()
    temptd_result['xi4'] = list()
    temptd_result['xi2_A'] = list()
    return temptd_result


def create_empty_temptd_for_share(max_len):
    """
    功能：创建数据结构用于保存时间域计算中的临时数据（最终用于一维时间域指标计算），每计算一段信号就会有一个对应的值（共享内存）
    返回：字典型数据包括：
    1. 平方和值xi2
    2. 最大值xmax
    3. 均值xmean
    4. 三次方和值xi3
    5. 四次方和值xi4
    6. A计权的平方值xi2_A
    function: create empty list to store the temp value for 1d time domain indicators
    :return
    temptd_result(dict): dict to store temp td value
    """
    temptd_result = dict()
    temptd_result['xi2'] = np.zeros(max_len)
    temptd_result['xmax'] = np.zeros(max_len)
    temptd_result['xmean'] = np.zeros(max_len)
    temptd_result['xi3'] = np.zeros(max_len)
    temptd_result['xi4'] = np.zeros(max_len)
    temptd_result['xi2_A'] = np.zeros(max_len)
    return temptd_result


def create_empty_twodoc(order_cut_calc_info, task_info, sensor_index, db_flag=0,
                        indicator_diagnostic=-1):
    """
    功能：初始化二维阶次切片结果，用户实时计算中进行更新
    输入：
    1. 阶次切片计算参数，主要包含关注阶次和阶次宽度
    2. 数据采集信息，主要提供单位
    3. 传感器索引，用于获取指定的单位
    4. 单位是否为dB， 默认不是（0表示不是，1表示是）
    返回：不包含具体数值的二维阶次切片结果
    function: create the empty twodoc_result and use it in following indicators calculation
    :param
    order_cut_calc_info(dict): the parameters for order cut calculation, include the cutted order list,
    e.g. ['1st','2nd']
    :return:
    twodoc_result(list): a list include all 2D order cutting indicator twodoc_result
    """
    twodoc_result = list()
    if db_flag:
        target_unit = 'dB'
    else:
        target_unit = task_info['units'][task_info["indicatorsUnitChanIndex"][sensor_index]]
    for indicator in order_cut_calc_info['orderName']:
        twodoc_result.append({
            'xName': order_cut_calc_info['xName'],
            'xUnit': order_cut_calc_info['xUnit'],
            'xValue': list(),
            'yName': indicator,
            # follow the unit of NI Unit, g or Pa, or dB
            'yUnit': target_unit,
            'yValue': list(),
            "indicatorDiagnostic": indicator_diagnostic
        })
    return twodoc_result


def create_empty_twodoc_for_share(order_cut_calc_info, task_info, sensor_index, os_max_len, db_flag=0,
                                  indicator_diagnostic=-1):
    """
    功能：初始化二维阶次切片结果，用户实时计算中进行更新（共享内存）
    输入：
    1. 阶次切片计算参数，主要包含关注阶次和阶次宽度
    2. 数据采集信息，主要提供单位
    3. 传感器索引，用于获取指定的单位
    4. 最多计算多少次阶次切片
    5. 单位是否为dB， 默认不是（0表示不是，1表示是）
    返回：不包含具体数值的二维阶次切片结果
    function: create the empty twodoc_result and use it in following indicators calculation
    :param
    order_cut_calc_info(dict): the parameters for order cut calculation, include the cutted order list,
    e.g. ['1st','2nd']
    :return:
    twodoc_result(list): a list include all 2D order cutting indicator twodoc_result
    """
    twodoc_result = list()
    if db_flag:
        target_unit = 'dB'
    else:
        target_unit = task_info['units'][task_info["indicatorsUnitChanIndex"][sensor_index]]
    for indicator in order_cut_calc_info['orderName']:
        twodoc_result.append({
            'xName': order_cut_calc_info['xName'],
            'xUnit': order_cut_calc_info['xUnit'],
            'xValue': np.zeros(os_max_len),
            'yName': indicator,
            # follow the unit of NI Unit, g or Pa, or dB
            'yUnit': target_unit,
            'yValue': [None] * os_max_len,
            "indicatorDiagnostic": indicator_diagnostic
        })
    return twodoc_result


def create_empty_twodos(task_info, sensor_index, db_flag=0, indicator_diagnostic=-1):
    """
    暂时未用到，该结果需要在三维阶次谱计算完成后才能生成，不用实时更新
    功能：初始化二维阶次谱结果
    输入：
    1. 数据采集信息，主要提供单位
    2. 传感器索引，用于获取指定单位
    3. 单位是否为dB，默认不是（0表示不是，1表示是）
    4. 初始化的评判结果，默认是为-1（表示缺失）
    返回：不包含具体数值的二维阶次谱结果
    function: create the empty twodOS result and use it in following indicators calculation(this result is based on
    3D order spectrum), not necessary
    :param
    taskInfo(dict): to provide the unit infomation
    :return:
    result(list): a list include all 2D order spectrum indicator result
    """
    if db_flag:
        target_unit = 'dB'
    else:
        target_unit = task_info['units'][task_info["indicatorsUnitChanIndex"][sensor_index]]
    twodos_result = [{
        'xName': 'Order',
        'xUnit': '',
        'xValue': list(),
        'yName': 'Order Spectrum',
        'yUnit': target_unit,
        'yValue': list(),
        "indicatorDiagnostic": indicator_diagnostic
    }]
    return twodos_result


def create_empty_threedos(order_spectrum_calc_info, task_info, sensor_index, db_flag=0,
                          indicator_diagnostic=-1):
    """
    功能：初始化三维阶次谱结果，用于实时阶次谱结果记录更新
    输入：
    1. 阶次谱计算信息
    2. 数据采集信息，用于提供单位
    3. 传感器索引，用于获取指定单位
    4. 单位是否为dB，默认不是（0表示不是，1表示是）
    5. 初始化的评判结果，默认是为-1（表示缺失）
    返回：不包含具体数值的三维阶次谱结果
    function: create the empty 3D order spectrum use it in following indicators calculation
    :param
    order_spectrum_calc_info(dict): the parameters for order spectrum calculation, include the xAxis and xUnit
    task_info(dict): update the unit of indicators
    :return:
    threedos_result(dict): a list include all 3D order spectrum, used for real time update
    """
    if db_flag:
        target_unit = 'dB'
    else:
        target_unit = task_info['units'][task_info["indicatorsUnitChanIndex"][sensor_index]]
    threedos_result = {
        'xName': order_spectrum_calc_info['xName'],
        'xUnit': order_spectrum_calc_info['xUnit'],
        'xValue': list(),
        'yName': 'Order',
        'yUnit': '',
        'yValue': list(order_spectrum_calc_info['order']),
        'zName': 'Order Spectrum',
        'zUnit': target_unit,
        'zValue': list(),
        "indicatorDiagnostic": indicator_diagnostic,
        'tempXi2': np.zeros(len(order_spectrum_calc_info['order']))
    }
    # 其中tempXi2是用于二维阶次谱计算
    return threedos_result


def create_empty_threedos_for_share(order_spectrum_calc_info, task_info, sensor_index, sensor_name, os_max_len,
                                    db_flag=0, indicator_diagnostic=-1):
    """
    功能：初始化三维阶次谱结果，用于实时阶次谱结果记录更新
    输入：
    1. 阶次谱计算信息
    2. 数据采集信息，用于提供单位
    3. 传感器索引，用于获取指定单位
    4. 单位是否为dB，默认不是（0表示不是，1表示是）
    5. 初始化的评判结果，默认是为-1（表示缺失）
    返回：不包含具体数值的三维阶次谱结果
    function: create the empty 3D order spectrum use it in following indicators calculation
    :param
    order_spectrum_calc_info(dict): the parameters for order spectrum calculation, include the xAxis and xUnit
    task_info(dict): update the unit of indicators
    :return:
    threedos_result(dict): a list include all 3D order spectrum, used for real time update
    """
    if db_flag:
        target_unit = 'dB'
    else:
        target_unit = task_info['units'][task_info["indicatorsUnitChanIndex"][sensor_index]]
    threedos_result = {
        'xName': order_spectrum_calc_info['xName'],
        'xUnit': order_spectrum_calc_info['xUnit'],
        'xValue': np.zeros(os_max_len),
        'yName': 'Order',
        'yUnit': '',
        'yValue': list(order_spectrum_calc_info['order']),
        'zName': 'Order Spectrum',
        'zUnit': target_unit,
        'zValue': np.zeros(shape=(os_max_len, len(order_spectrum_calc_info['order']))),
        "indicatorDiagnostic": indicator_diagnostic,
        'tempXi2': np.zeros(len(order_spectrum_calc_info['order']))
    }
    # 其中tempXi2是用于二维阶次谱计算
    return threedos_result


def update_nvh_data_for_thread(twod_td, temp_td, index_twod_td, twod_sf, temp_sf, twod_sf_counter,
                               threed_os, twod_oc, counter_or):
    """
    Function:
    由于开辟是开了很大的数组，在测试段结束后要进行整个测试段的计算，将xValue和yValue切割
    Args:
    twod_td:
    temp_td:
    index_twod_td:
    twod_sf:
    temp_sf:
    twod_sf_counter:
    Returns:
    更新后的二维指标结果
    """
    for result in twod_td:
        result['xValue'] = result['xValue'][:index_twod_td]
        result['yValue'] = result['yValue'][:index_twod_td]
    for key in temp_td.keys():
        temp_td[key] = temp_td[key][:index_twod_td]

    for key in temp_sf.keys():
        counter_twod_sf = twod_sf_counter[key]
        temp_sf[key]['xi2'] = temp_sf[key]['xi2'][:counter_twod_sf]
        temp_sf[key]['xmax'] = temp_sf[key]['xmax'][:counter_twod_sf]
        temp_sf[key]['xmean'] = temp_sf[key]['xmean'][:counter_twod_sf]
        temp_sf[key]['xi3'] = temp_sf[key]['xi3'][:counter_twod_sf]
        temp_sf[key]['xi4'] = temp_sf[key]['xi4'][:counter_twod_sf]
        temp_sf[key]['xi2_A'] = temp_sf[key]['xi2_A'][:counter_twod_sf]

    for result in twod_sf:
        result['xValue'] = result['xValue'][:twod_sf_counter[result['yName'].rsplit('-', 1)[0]]]
        result['yValue'] = result['yValue'][:twod_sf_counter[result['yName'].rsplit('-', 1)[0]]]

    threed_os['xValue'] = threed_os['xValue'][:counter_or]
    threed_os['zValue'] = threed_os['zValue'][:counter_or]

    for result in twod_oc:
        result['xValue'] = result['xValue'][:counter_or]
        result['yValue'] = result['yValue'][:counter_or]
    return twod_td, temp_td, twod_sf, temp_sf, threed_os, twod_oc


def create_empty_threedtfm(time_freq_map_calc_info, freq, task_info, sensor_index, db_flag=0,
                           indicator_diagnostic=-1):
    """
    功能：初始化时频彩图结果
    输入：
    1. 时频分析配置参数信息
    2. 数据采集参数，提供单位信息
    3. 传感器索引，用于获取指定单位
    4. 单位是否为dB，默认不是（0表示不是，1表示是）
    5. 初始化的评判结果，默认是为-1（表示缺失）
    返回：
    function: create an empty 3D time frequency map and use it in following indicators calculation
    :param
    time_freq_map_calc_info(dict):include the parameters used for time frequency map calculation
    task_info(dict): update the unit of indicators
    :return
    threedtfm_result(dict): a dict include all 3D time/speed frequency colormap final_result
    """
    if db_flag:
        target_unit = 'dB'
    else:
        target_unit = task_info['units'][task_info["indicatorsUnitChanIndex"][sensor_index]]
    threedtfm_result = {
        'yName': time_freq_map_calc_info['xName'],
        'yUnit': time_freq_map_calc_info['xUnit'],
        'yValue': list(),
        'xName': 'Frequency',
        'xUnit': 'Hz',
        'xValue': list(freq),
        'zName': 'Colormap',
        'zUnit': target_unit,
        'zValue': list(),
        "indicatorDiagnostic": indicator_diagnostic
    }
    return threedtfm_result


def create_empty_ssa(ssa_calc_info, task_info, sensor_index):
    """
    功能：初始化ssa结果，用于实时更新
    输入:
    1. SSA计算参数信息
    2. 数据采集参数信息，主要提供单位信息
    3. 传感器索引，用于获取指定单位
    返回：不带具体数值的SSA结果
    function: create empty TSA final_result for real-time update 与ssa相同
    :param
    ssa_calc_info(dict): the parameters used for ssa
    task_info(dict): update the unit of indicators
    :return
    ssa_result(list): empty final_result need to update
    """
    temp_result = dict()
    ssa_result = list()
    counters = list()
    for i, gearName in enumerate(ssa_calc_info['gearName']):
        counters.append(0)
        ssa_result.append({
            'xName': ssa_calc_info["xName"],
            'xUnit': ssa_calc_info["xUnit"],
            'xValue': list(),
            'yName': 'vibration',
            'yUnit': task_info['units'][task_info["indicatorsUnitChanIndex"][sensor_index]],
            'yValue': 0,
            'Name': gearName,
            'isInputShaft': 0
        })
        if i == 0:
            ssa_result[i]['isInputShaft'] = 1
    temp_result['counters'] = counters
    temp_result['resultData'] = ssa_result
    return temp_result


def create_empty_twodsf(stat_factor_calc_info, sensor_index, indicator_diagnostic=-1):
    """
    功能：初始化按圈计算的统计学指标结果，用于实时更新
    输入：
    1. 按圈计算参数信息
    2. 传感器索引（用于指定单位）
    3. 初始化的评判结果，默认是为-1（表示缺失）
    返回：包含结构但不包含具体数值的按圈计算结果
    """
    twodsf_result = list()
    if not stat_factor_calc_info['indicatorList']:
        return twodsf_result
    for gearName in stat_factor_calc_info['gearName']:
        for i, indicator in enumerate(stat_factor_calc_info['indicatorList']):
            twodsf_result.append({
                'xName': stat_factor_calc_info['xName'],
                'xUnit': stat_factor_calc_info['xUnit'],
                'xValue': list(),
                'yName': '-'.join([gearName, indicator]),
                'yUnit': stat_factor_calc_info['indicatorUnit'][sensor_index][i],
                'yValue': list(),
                "indicatorDiagnostic": indicator_diagnostic
            })
    return twodsf_result


def create_empty_tempsf(stat_factor_calc_info):
    """
    功能：初始化按圈计算的统计学指标临时结果，用于最后计算一维指标
    输入：计算参数信息
    返回：包含结构但不包含具体数值的临时结果，包括：
    1. 平方和值xi2
    2. 最大值xmax
    3. 均值xmean
    4. 三次方和值xi3
    5. 四次方和值xi4
    每个轴都有相应的结果，所以是按gear name保存的
    """
    tempsf_result = dict()
    if not stat_factor_calc_info['indicatorList']:
        return tempsf_result
    for gearName in stat_factor_calc_info['gearName']:
        tempsf_result[gearName] = dict()
        tempsf_result[gearName]['xi2'] = list()
        tempsf_result[gearName]['xmax'] = list()
        tempsf_result[gearName]['xmean'] = list()
        tempsf_result[gearName]['xi3'] = list()
        tempsf_result[gearName]['xi4'] = list()
        tempsf_result[gearName]['xi2_A'] = list()
        tempsf_result[gearName]['counter'] = 0
    return tempsf_result


def create_empty_twodsf_for_share(stat_factor_calc_info, sensor_index, sensor_name, rsp_max_len,
                                  indicator_diagnostic=-1):
    """
    功能：初始化按圈计算的统计学指标结果，用于实时更新
    输入：
    1. 按圈计算参数信息
    2. 传感器索引（用于指定单位）
    3. 重采样后数据的最大长度
    4. 初始化的评判结果，默认是为-1（表示缺失）
    返回：包含结构但不包含具体数值的按圈计算结果
    """
    twodsf_result = {}
    if not stat_factor_calc_info:
        return twodsf_result
    for statInfo in stat_factor_calc_info[sensor_name]:
        # 计算得到可以进行多少次计算（即二维按圈计算结果的长度）
        max_len = rsp_max_len // statInfo['stepPoints'] + 1
        twodsf_result[tuple(statInfo['value'])] = {'xName': statInfo['xName'],
                                                   'xUnit': statInfo['xUnit'],
                                                   'xValue': np.zeros(max_len),
                                                   'yName': statInfo['index'],
                                                   'yUnit': statInfo['unit'][0] if len(statInfo['unit']) == 1 else
                                                   statInfo['unit'][sensor_index],
                                                   'yValue': [None] * max_len,
                                                   "indicatorDiagnostic": indicator_diagnostic
                                                   }
        return twodsf_result


def create_empty_twodsf_for_const(stat_factor_calc_info, sensor_index, max_circle, indicator_diagnostic=-1):
    """
    功能：初始化按圈计算的统计学指标结果，用于实时更新，恒速电机版
    输入：计算参数信息
    返回：包含结构但不包含具体数值的按圈计算结果
    """
    twodsf_result = list()
    if not stat_factor_calc_info['indicatorNestedList'][sensor_index]:
        return twodsf_result
    for k, gearName in enumerate(stat_factor_calc_info['gearName']):
        max_len = int(max_circle // stat_factor_calc_info['stepNums'][k]) + 1
        for i, indicator in enumerate(stat_factor_calc_info['indicatorNestedList'][sensor_index]):
            twodsf_result.append({
                'xName': stat_factor_calc_info['xName'],
                'xUnit': stat_factor_calc_info['xUnit'],
                'xValue': np.zeros(max_len),
                'yName': '-'.join([gearName, indicator]),
                'yUnit': stat_factor_calc_info['indicatorUnit'][sensor_index][i],
                'yValue': [None] * max_len,
                "indicatorDiagnostic": indicator_diagnostic
            })
    return twodsf_result


def create_empty_tempsf_for_share(stat_factor_calc_info, sensor_index, sensor_name, rspMaxLen):
    """
    功能：初始化按圈计算的统计学指标临时结果，用于最后计算一维指标
    输入：
    1. 计算参数信息
    返回：包含结构但不包含具体数值的临时结果，包括：
    1. 平方和值xi2
    2. 最大值xmax
    3. 均值xmean
    4. 三次方和值xi3
    5. 四次方和值xi4
    每个轴都有相应的结果，所以是按gear name保存的
    """
    tempsf_result = dict()
    if not stat_factor_calc_info:
        return tempsf_result
    for statInfo in stat_factor_calc_info[sensor_name]:
        maxLen = rspMaxLen // stat_factor_calc_info['stepPoints'] + 1
        tempsf_result[tuple(statInfo['value'])] = {key: np.zeros(maxLen) for key in
                                                   ('xi2', 'xmax', 'xmean', 'xi3', 'xi4', 'xi2_A')}
        # 事先开辟了足够的内存，index为下一个需要赋值的索引，
        # 该值与同一根轴上下一个twodsf需要赋值的索引相同，也与下面的
        tempsf_result[tuple(statInfo['value'])]['tempsf_index'] = 0
        tempsf_result[tuple(statInfo['value'])]['counter'] = 0
    return tempsf_result


def create_empty_tempsf_for_const(stat_factor_calc_info, sensor_index, max_circle):
    """
    功能：初始化按圈计算的统计学指标临时结果，用于最后计算一维指标,恒速电机版
    输入：计算参数信息
    返回：包含结构但不包含具体数值的临时结果，包括：
    1. 平方和值xi2
    2. 最大值xmax
    3. 均值xmean
    4. 三次方和值xi3
    5. 四次方和值xi4
    每个轴都有相应的结果，所以是按gear name保存的
    """
    tempsf_result = dict()
    if not stat_factor_calc_info['indicatorNestedList'][sensor_index]:
        return tempsf_result
    for i, gearName in enumerate(stat_factor_calc_info['gearName']):
        maxLen = int(max_circle // stat_factor_calc_info['stepNums'][i]) + 1
        tempsf_result[gearName] = dict()
        tempsf_result[gearName]['xi2'] = np.zeros(maxLen)
        tempsf_result[gearName]['xmax'] = np.zeros(maxLen)
        tempsf_result[gearName]['xmean'] = np.zeros(maxLen)
        tempsf_result[gearName]['xi3'] = np.zeros(maxLen)
        tempsf_result[gearName]['xi4'] = np.zeros(maxLen)
        tempsf_result[gearName]['xi2_A'] = np.zeros(maxLen)
        # 事先开辟了足够的内存，index为下一个需要赋值的索引，
        # 该值与同一根轴上下一个twodsf需要赋值的索引相同，也与下面的
        tempsf_result[gearName]['tempsf_index'] = 0
        tempsf_result[gearName]['counter'] = 0
        tempsf_result[gearName]["lastRevIndex"] = 0
    return tempsf_result


if __name__ == '__main__':
    # 测试上述函数
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(pathname)s %(funcName)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='[%Y-%m-%d %H:%M:%S]',
        filename='D:/qdaq/Log/temp.log',
        filemode='a'
    )
    try:
        from parameters import Parameters
        from common_info import config_folder, config_file

        type_info = "Seres-TTL-Type1"
        config_filename = os.path.join(config_folder, "_".join([type_info, config_file]))
        print(config_filename)
        if os.path.exists(config_filename):
            param = Parameters(config_filename)
            # test the function in this module
            # check and create folder list
            param.basicInfo['serialNo'] = 'demo1'
            sensor_i = 0
            test_i = 0
            test_time = time_get()
            r = confirm_target_folder(test_time, param.folderInfo, basic_info=param.basicInfo)
            print(r[0], r[1], r[2])
            print(os.listdir(r[0]))
            test_result = create_empty_final_result(test_time, param.basicInfo,
                                                    param.speedRecogInfo, param.limitCompareFlag['overLimit'])
            test_result['resultData'][sensor_i]['dataSection'][test_i]['twodTD'] = create_empty_twodtd(
                param.timeDomainCalcInfo,
                sensor_i, indicator_diagnostic=param.speedRecogInfo['initial_indicator_diagnostic'][test_i])
            test_result['resultData'][sensor_i]['dataSection'][test_i]['twodOC'] = create_empty_twodoc(
                param.orderCutCalcInfo,
                param.taskInfo, sensor_i,
                param.basicInfo['dBFlag'],
                indicator_diagnostic=param.speedRecogInfo['initial_indicator_diagnostic'][test_i])
            test_result['resultData'][sensor_i]['dataSection'][test_i]['twodOS'] = create_empty_twodos(
                param.taskInfo, sensor_i,
                param.basicInfo['dBFlag'],
                indicator_diagnostic=param.speedRecogInfo['initial_indicator_diagnostic'][test_i])
            pprint.pprint(test_result)
            pprint.pprint(create_empty_threedos(param.orderSpectrumCalcInfo, param.taskInfo, sensor_i,
                                                param.basicInfo['dBFlag'], indicator_diagnostic=
                                                param.speedRecogInfo['initial_indicator_diagnostic'][test_i]))
            freq = list([1, 2, 3, 4])
            pprint.pprint(
                create_empty_threedtfm(param.timeFreqMapCalcInfo, freq, param.taskInfo, sensor_i,
                                       param.basicInfo['dBFlag']))
            pprint.pprint(create_empty_ssa(param.ssaCalcInfo, param.taskInfo, sensor_i))
            # 写入数据到文件进行确认
            with open('D:/qdaq/Report/emptyResult.json', 'w') as f:
                json.dump(test_result, f, indent=4)
    except Exception:
        traceback.print_exc()
        logging.warning("exec failed, failed msg:" + traceback.format_exc())
        sys.exit()
