import numpy as np
from scipy import signal, stats
import os

ref_g = 1e-6 / 9.8  # accel unit: m/s^2
# 声压信号参考值
ref_sound_pressure = 20e-6  # unit: Pa
version = 4  # 目前只适用基础版本的qDAQ
# qdaq版这样写
if version in [1, 2]:
    xName = "Speed"
    xUnit = "rpm"
# 恒速电机版这样写
elif version in [3, 4, 5]:
    xName = "Time"
    xUnit = "s"
constant_initial_indicator_diagnostic = -1


# 窗函数
def filter_win(nfft, wtype):
    """
    # create window applied on signal before fft
    功能：生成窗函数（根据类型和点数），类型包括：
    1. hanning：汉宁窗
    2. hamming：海明窗
    3. kaiser：9阶凯塞窗
    4. blackman：布雷克曼窗
    5. bartlett：巴雷特窗
    6. flattop：平顶窗
    7. 其他：矩形窗（相当于不加窗）
    返回：
    1. 窗函数序列
    2. 修正因子（主要用于能量修正，以加窗后的能量可保持与未加窗时相同）
    """
    if wtype == 'hanning':
        win = np.hanning(nfft)
        win_factor = 1.633
    elif wtype == 'hamming':
        win = np.hamming(nfft)
        win_factor = 1.586
    elif wtype == 'kaiser':
        win = np.kaiser(nfft, 9)
        win_factor = 1.81
    elif wtype == 'blackman':
        win = np.blackman(nfft)
        win_factor = 1.812
    elif wtype == 'bartlett':
        # triangle window
        win = np.bartlett(nfft)
        win_factor = 1.732
    elif wtype == 'flattop':
        win = signal.flattop(nfft, sym=False)
        win_factor = 1.069
    elif wtype == 'hanning1':
        # create hanning window manually
        win = np.array(
            [0.5 - 0.5 * np.cos(2 * np.pi * n / (nfft - 1)) for n in range(nfft)])
        win_factor = 1.633
    else:
        win = np.ones(nfft)
        win_factor = 1.0
    return win, win_factor


# 读取参数配置

def basic_info_update(basic_info):
    """
    功能：确认并更新基础信息
    输入：配置文件中的基础信息
    输出：确认后的基础信息
    """
    if "unit" not in basic_info.keys():
        # 不存在这个信息则直接认为不需要转换
        basic_info['dBFlag'] = 0
    else:
        if basic_info['unit'].lower() == 'db':
            # 如果单位为dB， 则认为需要转换
            basic_info['dBFlag'] = 1
        else:
            basic_info['dBFlag'] = 0
    return basic_info


def task_info_update(task_info):
    """
    功能：更新单位索引信息
    返回：更新后的NI DAQmx数据采集任务配置信息
    """
    # 振动或声音信号的参考值（dB转换）
    task_info['refValue'] = list()
    # 获取用于NVH分析的通道并记录通道名
    task_info['targetChan'] = list()
    # 获取振动或声音通道的原始单位
    task_info['sensorUnit'] = list()
    task_info['indicatorsUnitChanIndex'] = list()
    for channel_index, channel_name in enumerate(task_info['channelNames']):
        if channel_name.startswith('Vib'):
            task_info['indicatorsUnitChanIndex'].append(channel_index)
            task_info['refValue'].append(ref_g)
            task_info['targetChan'].append(channel_name)
            task_info['sensorUnit'].append(task_info['units'][channel_index])
        elif channel_name.startswith('Mic'):
            task_info['indicatorsUnitChanIndex'].append(channel_index)
            task_info['refValue'].append(ref_sound_pressure)
            task_info['targetChan'].append(channel_name)
            task_info['sensorUnit'].append(task_info['units'][channel_index])
        elif channel_name.lower().startswith('umic'):
            task_info['indicatorsUnitChanIndex'].append(channel_index)
            task_info['sensorChan'].append(channel_name)
            task_info['refValue'].append(ref_sound_pressure)
            task_info['sensorUnit'].append(task_info['units'][channel_index])
    return task_info


def speed_calc_info_update(speed_calc_info):
    """
    功能：确认转速比信息
    返回：更新后的转速计算信息
    """
    # 确认转速比
    if 'speedRatio' not in speed_calc_info.keys():
        # 若没有设置转速比则默认是1.0
        speed_calc_info['speedRatio'] = 1.0
    if speed_calc_info['speedRatio'] <= 0:
        # 若转速比异常（小于等于0）则认为设置出错，强制设置为1
        speed_calc_info['speedRatio'] = 1.0
    return speed_calc_info


def speed_recog_info_update(speed_recog_info):
    """
    # just update some speed recognition info to avoid re calculation inside loop
    功能：提前计算转速识别所需要的的参数，包括：
    1. minRange 和 maxRange：变速段对应的上上下限
    2. slope：目标转速的斜率（预期）
    3. speedPattern：转速类型，1表示恒速段；2表示升速段；3表示降速段
    4. minSpeed：测试段最小转速
    返回：
    1. 更新后的工况识别参数
    """
    speed_recog_info['minRange'] = list()
    speed_recog_info['maxRange'] = list()
    speed_recog_info['slope'] = list()
    speed_recog_info['speedPattern'] = list()
    speed_recog_info['minSpeed'] = list()
    speed_recog_info['notDummyFlag'] = list()
    speed_recog_info['initial_indicator_diagnostic'] = [constant_initial_indicator_diagnostic] * len(
        speed_recog_info['startSpeed'])
    # 确认扭矩识别信息
    if "torqueRecogFlag" not in speed_recog_info.keys():
        speed_recog_info["torqueRecogFlag"] = list(
            np.zeros(len(speed_recog_info["testName"]), dtype='int'))
    speed_recog_info["minTorque"] = list()
    speed_recog_info["maxTorque"] = list()
    # 定义转速范围（用于区分恒速和变速段）
    if 'speedRange' not in speed_recog_info.keys():
        speed_recog_info['speedRange'] = 100

    speed_recog_info['overallMinSpeed'] = min(speed_recog_info["startSpeed"])
    # 对于恒速电机，要更新时间
    if "waitTime" in speed_recog_info.keys():
        speed_recog_info["startTime"] = list()
        speed_recog_info["endTime"] = list()
        speed_recog_info["startTime"].append(speed_recog_info["waitTime"][0])
        speed_recog_info["endTime"].append(
            speed_recog_info["startTime"][0] + speed_recog_info["expectT"][0])

        for i in range(1, len(speed_recog_info["testName"])):
            speed_recog_info["startTime"].append(
                speed_recog_info["waitTime"][i] + speed_recog_info["endTime"][i - 1])
            speed_recog_info["endTime"].append(
                speed_recog_info["startTime"][i] + speed_recog_info["expectT"][i])
    return speed_recog_info


def time_domain_calc_info_update(time_domian_calc_info, task_info, basic_info):
    """
    # update the RMS, Crest, Kurtosis, and Skewness unit
    功能：更新RMS，Crest，Kurtosis和Skewness，以及SPL或SPLA的单位，具体信息如下：
    1. RMS：单位来源于指定的采集通道
    2. Crest，Kurtosis，Skewness：无单位
    3. SPL：声压级，单位为dB
    4. SPL(A)：A计权声压级，单位为dB(A)
    返回：更新后的时间域指标计算参数
    """

    time_domian_calc_info['indicatorUnit'] = list()
    time_domian_calc_info['refValue'] = task_info['refValue']
    time_domian_calc_info["indicatorNestedList"] = list()

    # if "Speed" in time_domian_calc_info['indicatorList']:
    #     time_domian_calc_info['indicatorList'].remove("Speed")
    # for channel_index, channel_name in enumerate(task_info['channelNames']):
    #     time_domian_calc_info["indicatorNestedList"].append(time_domian_calc_info['indicatorList'])
    # for unit_index in task_info["indicatorsUnitChanIndex"]:
    #     temp_unit_list = list()
    #     for indicator in time_domian_calc_info['indicatorList']:
    #         if indicator == 'RMS':
    #             if basic_info['dBFlag']:
    #                 temp_unit_list.append('dB')
    #             else:
    #                 temp_unit_list.append(task_info['units'][unit_index])
    #         elif indicator == 'SPL(A)':
    #             temp_unit_list.append('dB(A)')
    #         elif indicator == 'SPL':
    #             temp_unit_list.append('dB')
    #         else:
    #             temp_unit_list.append('')
    #     time_domian_calc_info['indicatorUnit'].append(temp_unit_list)
    # time_domian_calc_info['xName'] = xName
    # time_domian_calc_info['xUnit'] = xUnit
    # time_domian_calc_info['calSize'] = int(
    #     task_info["sampleRate"] / time_domian_calc_info["calRate"])

    # return time_domian_calc_info

    if "Speed" in time_domian_calc_info['vibrationIndicatorList']:
        time_domian_calc_info['vibrationIndicatorList'].remove("Speed")
    if "Speed" in time_domian_calc_info['soundIndicatorList']:
        time_domian_calc_info['soundIndicatorList'].remove("Speed")

    for channel_index, channel_name in enumerate(task_info['channelNames']):
        if channel_name.lower().startswith('vib'):
            time_domian_calc_info["indicatorNestedList"].append(
                time_domian_calc_info['vibrationIndicatorList'])
        elif channel_name.lower().startswith('mic') or channel_name.lower().startswith("umic"):
            time_domian_calc_info["indicatorNestedList"].append(
                time_domian_calc_info['soundIndicatorList'])

    for i, unit_index in enumerate(task_info["indicatorsUnitChanIndex"]):
        temp_unit_list = list()
        for indicator in time_domian_calc_info['indicatorNestedList'][i]:
            if indicator == 'RMS':
                if basic_info['dBFlag']:
                    temp_unit_list.append('dB')
                else:
                    temp_unit_list.append(task_info['units'][unit_index])
            elif indicator == 'SPL(A)':
                temp_unit_list.append('dB(A)')
            elif indicator == 'SPL':
                temp_unit_list.append('dB')
            else:
                temp_unit_list.append('')
        time_domian_calc_info['indicatorUnit'].append(temp_unit_list)
    time_domian_calc_info['xName'] = xName
    time_domian_calc_info['xUnit'] = xUnit
    time_domian_calc_info['calSize'] = int(task_info["sampleRate"] / time_domian_calc_info["calRate"])
    return time_domian_calc_info


def order_spectrum_calc_info_update(order_spectrum_calc_info, speed_calc_info, min_speed,
                                    task_info):
    """
    # just update some order spectrum calculation info to avoid re calculation inside loop
    功能：提前计算阶次谱计算的参数，包括：
    1. 确认重叠比例是否合理（overlapRatio）
    2. 角度域重采样间隔（dr_af)
    3. fft变换点数（nfft）
    4. fft变换步进点数（nstep）
    5. fft变换的窗函数（win），不同通道的振动信号可以加不同的窗函数
    6. fft变化的归一化因子（normFactor），可以决定需不需要补偿（根据winCorrectFlag来决定）
    7. 阶次谱x轴（order），根据变换点数和分辨率决定，然后根据设定的最大关注阶次进行截取
    返回：更新后的阶次谱计算参数
    """
    # 确认最大关注阶次是否合理
    if version == 1 or version == 2:
        max_order_available = (60 * task_info['sampleRate']) / (
                min_speed * 2 * 1.6384)
    else:
        max_order_available = (60 * task_info['sampleRate']) / (
                min_speed * 2)
    if order_spectrum_calc_info['maxOrder'] > max_order_available:
        raise ValueError(
            "max order: {} set error, should less than {}".format(
                order_spectrum_calc_info['maxOrder'],
                max_order_available))
    if order_spectrum_calc_info['overlapRatio'] >= 1:
        order_spectrum_calc_info['overlapRatio'] = 0
    # 定义角度域降采样之前的采样点角度间隔
    order_spectrum_calc_info['dr_bf'] = min_speed / 60 / task_info['sampleRate']
    # 定义角度域降采样之后的采样点角度间隔
    # order_spectrum_calc_info['dr_af'] = 1 / order_spectrum_calc_info[
    #     'maxOrder'] / 2 / 1.6384  # 1.6384=1.28*1.28
    order_spectrum_calc_info['dr_af'] = 1 / order_spectrum_calc_info[
        'maxOrder'] / 2 / 1.28  # 1.6384=1.28*1.28
    order_spectrum_calc_info['nfft'] = int(
        order_spectrum_calc_info['revNum'] / order_spectrum_calc_info['dr_af'])
    order_spectrum_calc_info['nstep'] = int(order_spectrum_calc_info['nfft'] *
                                            (1 - order_spectrum_calc_info[
                                                'overlapRatio']))
    # 构建窗函数
    if order_spectrum_calc_info['window']:
        order_spectrum_calc_info['win'] = list()
        order_spectrum_calc_info['normFactor'] = list()  # 1.414 for rms normalization
        for i, winType in enumerate(order_spectrum_calc_info['window']):
            win_temp = filter_win(order_spectrum_calc_info['nfft'], winType)
            order_spectrum_calc_info['win'].append(win_temp[0])
            # 是否开启修正系数 0是不开启 赋值会被压下来
            if order_spectrum_calc_info['winCorrectFlag'][i]:
                order_spectrum_calc_info['normFactor'].append(
                    1.414 / order_spectrum_calc_info['nfft'] * win_temp[1])
            else:
                order_spectrum_calc_info['normFactor'].append(
                    1.414 / order_spectrum_calc_info['nfft'])
    order_spectrum_calc_info['order'] = (
        (np.fft.rfftfreq(order_spectrum_calc_info['nfft']) *
         order_spectrum_calc_info['orderResolution'] *
         order_spectrum_calc_info['nfft']))
    if order_spectrum_calc_info['maxOrder']:
        order_spectrum_calc_info['order'] = (order_spectrum_calc_info['order'][:(
                order_spectrum_calc_info['revNum'] * order_spectrum_calc_info[
            'maxOrder'])])
    if speed_calc_info['speedRatio'] != 1:
        # 提前计算需要更换的阶次轴（若转速比不为1才需要进行转换）
        order_spectrum_calc_info['convertOrder'] = (
                order_spectrum_calc_info['order'] / speed_calc_info[
            'speedRatio']).tolist()
    order_spectrum_calc_info['ppr'] = speed_calc_info['ppr']
    order_spectrum_calc_info['refValue'] = task_info['refValue']
    order_spectrum_calc_info['xName'] = xName
    order_spectrum_calc_info['xUnit'] = xUnit
    arPoints = round(task_info["sampsPerChan"] / 200)
    order_spectrum_calc_info["arPoints"] = arPoints if arPoints > 10 else 10
    return order_spectrum_calc_info


def order_cut_calc_info_update(order_cut_calc_info, order_spectrum_calc_info):
    """
    功能：提前计算二维阶次切片所需要的参数，主要是限制目标阶次的边界，包括：
    1. 最小阶次（orderMin）：提取阶次切片时的目标阶次应大于该最小阶次，与阶次切片的宽度有关（左右的点数）
    2. 最大阶次（orderMax）：提取阶次切片时的目标阶次应小于该最大阶次，与阶次切片的宽度有关（左右的点数）
    返回：更新后的二维阶次切片计算参数
    """
    # update the order boundary for target order confirm
    min_order_available = order_spectrum_calc_info['orderResolution'] * (
            order_cut_calc_info['pointNum'] // 2)
    max_order_available = \
        order_spectrum_calc_info['maxOrder'] - order_spectrum_calc_info['orderResolution'] * \
        (order_cut_calc_info['pointNum'] // 2 + 1)
    min_order = min(map(min, order_cut_calc_info['orderList']))
    max_order = max(map(max, order_cut_calc_info['orderList']))
    # 校验关注阶次
    if min_order < min_order_available:
        raise ValueError(
            'min order of 2D order slice: {} set is out of range, should bigger than: {}'.format(
                min_order,
                min_order_available))
    if max_order > max_order_available:
        raise ValueError(
            'max order of 2D order slice: {} set is out of range, should smaller than: {}'.format(
                max_order,
                max_order_available))
    order_cut_calc_info['xName'] = xName
    order_cut_calc_info['xUnit'] = xUnit
    return order_cut_calc_info


def oned_os_calc_info_update(oned_os_calc_info, order_spectrum_calc_info):
    """
    功能：提前计算一维阶次切片指标所需要的参数，主要是限制目标阶次的边界，包括：
    1. 最小阶次（orderMin）：提取阶次切片时的目标阶次应大于该最小阶次，与阶次切片的宽度有关（左右的点数）
    2. 最大阶次（orderMax）：提取阶次切片时的目标阶次应小于该最大阶次，与阶次切片的宽度有关（左右的点数）
    返回：更新后的一维阶次切片计算参数
    """
    min_order_available = order_spectrum_calc_info['orderResolution'] * (
            oned_os_calc_info['pointNum'] // 2)
    max_order_available = order_spectrum_calc_info['maxOrder'] - (
            order_spectrum_calc_info['orderResolution'] * (oned_os_calc_info['pointNum'] // 2 + 1))
    min_order = min(map(min, oned_os_calc_info['orderList']))
    max_order = max(map(max, oned_os_calc_info['orderList']))
    # 校验关注阶次
    if min_order < min_order_available:
        raise ValueError(
            'min order of 1D order indicator: {} set is out of range, should bigger than: {}'.format(
                min_order,
                min_order_available))
    if max_order > max_order_available:
        raise ValueError(
            'max order of 1D order indicator: {} set is out of range, should smaller than: {}'.format(
                max_order,
                max_order_available))
    return oned_os_calc_info


def cepstrum_calc_info_update(order_spectrum_calc_info):
    """
    功能：生成倒阶次谱所需要的计算参数，包括：
    1. 圈数（revNum），主要用于形成倒阶次谱的x轴信息，由阶次谱的阶次分辨率得到
    返回：倒阶次谱计算的参数信息
    """
    cepstrum_calc_info = dict()
    cepstrum_calc_info['revNum'] = 1 / order_spectrum_calc_info['orderResolution']
    return cepstrum_calc_info


def ssa_calc_info_update(ssa_calc_info, speed_calc_info):
    """
    功能：确认和提前计算SSA分析所需要的的配置参数（按圈计算的指标也会用到），包括：
    1. 生成每根轴对应的名称（gearName），比如输入轴，中间轴，输出轴
    2. 生成每根轴对应的ppr（pprNum），由转速来源轴和齿数比决定
    3. 更新每根轴的齿数比（gearRatio），主要是插入输入轴的比值，例如3个齿数比对应的是4根轴
    返回：更新后的SSA参数
    """
    # 默认只有输入轴，更新每个转轴名称
    ssa_calc_info['gearName'] = list()
    ssa_calc_info['pprNum'] = list()
    ssa_calc_info['factors'] = list()
    # 确认是否输入轴和输出轴为同一个
    if len(ssa_calc_info['gearRatio']) == 1 and ssa_calc_info['gearRatio'][0] == 1:
        ssa_calc_info['gearRatio'] = list()
    ssa_calc_info['gearRatio'].insert(0, 1.0)
    if ssa_calc_info['gearRatio']:
        if ssa_calc_info['onInputShaft'] == 0:
            # shaftIndex主要是用于每根轴的ppr计算（正向还是逆向），转速来源于输出轴则为逆向
            ssa_calc_info['shaftIndex'] = len(ssa_calc_info['gearRatio']) - 1
        else:
            # 转速来源于输入轴则为正向
            ssa_calc_info['shaftIndex'] = 0
        for i in range(len(ssa_calc_info['gearRatio'])):
            # 更新ppr信息
            factor = np.prod(ssa_calc_info['gearRatio'][:i + 1]) / np.prod(
                ssa_calc_info['gearRatio'][:ssa_calc_info['shaftIndex'] + 1])
            ssa_calc_info['pprNum'].append(speed_calc_info['ppr'] * factor)
            ssa_calc_info['factors'].append(factor)
            # 更新转轴名称
            if i == 0:
                ssa_calc_info['gearName'].append('InputShaft')
            elif i + 1 == len(ssa_calc_info['gearRatio']):
                ssa_calc_info['gearName'].append('OutputShaft')
            elif i == 1 and len(ssa_calc_info['gearRatio']) == 3:
                ssa_calc_info['gearName'].append('CounterShaft')
            else:
                ssa_calc_info['gearName'].append('CounterShaft' + str(i))
    ssa_calc_info["xName"] = "angle"
    ssa_calc_info["xUnit"] = "°"
    return ssa_calc_info


def stat_factor_calc_info_update(stat_factor_calc_info, ssa_calc_info,
                                 order_spectrum_calc_info, task_info, basic_info):
    """
    功能：统计学指标按圈计算参数（指标名称参考时间域指标），包括：
    1. 计算圈数（revNum），即多少圈计算一次。默认为1
    2. 重叠比例（overlapRatio），默认为0.5
    3. 计算的点数（pointsNum），根据转一圈需要的点数依次对应不同的轴
    4. 步进长度（stepPoints），每次步进的点数，由重叠比例决定，每根轴对应不同的值
    输入：
    1. 时间域指标参数信息
    2. ssa分析参数信息
    3. 阶次谱计算参数信息
    4. 数据采集参数信息
    返回：按圈计算的参数信息
    """
    # 更新统计学指标按圈计算参数
    if 'revNum' not in stat_factor_calc_info.keys():
        # 每次计算的圈数未设置则默认为1
        stat_factor_calc_info['revNum'] = 1
    if 'overlapRev' not in stat_factor_calc_info.keys():
        # 重叠比例未设置则默认为0.5
        stat_factor_calc_info['overlapRev'] = 0
    stat_factor_calc_info['overlapRatio'] = stat_factor_calc_info['overlapRev'] / \
                                            stat_factor_calc_info['revNum']
    # 确定重叠比率是否设置合理
    if stat_factor_calc_info['overlapRatio'] >= 1:
        # 若重叠比率超过1则强制归零
        stat_factor_calc_info['overlapRatio'] = 0

    # 计算的圈数定义在stat_factor_calc_info的revNum中，转速来源轴定义在ssa_calc_info的onInputShaft中
    # 基于转速来源轴和齿轮副信息得到每个轴转过固定圈数所需要的数据点数
    stat_factor_calc_info['pointsNum'] = list()
    stat_factor_calc_info['stepPoints'] = list()
    # 基于角度域重采样后的振动信号
    stat_factor_calc_info['sampleRate'] = round(1 / order_spectrum_calc_info['dr_af'])
    temp_num = int(stat_factor_calc_info['revNum'] / order_spectrum_calc_info['dr_af'])
    # print(temp_num)
    if ssa_calc_info['gearRatio']:
        stat_factor_calc_info['revNums'] = list()
        stat_factor_calc_info['stepNums'] = list()
        for i in range(len(ssa_calc_info['gearRatio'])):
            stat_factor_calc_info['pointsNum'].append(
                int(round(temp_num * np.prod(ssa_calc_info['gearRatio'][:i + 1]) / np.prod(
                    ssa_calc_info['gearRatio'][:ssa_calc_info['shaftIndex'] + 1]))))
            stat_factor_calc_info['stepPoints'].append(
                int(stat_factor_calc_info['pointsNum'][i] * (
                        1 - stat_factor_calc_info['overlapRatio'])))

            # 其它轴每转过revNum，转速来源轴转了多少圈
            stat_factor_calc_info['revNums'].append(
                stat_factor_calc_info['revNum'] * np.prod(ssa_calc_info['gearRatio'][:i + 1]) / np.prod(
                    ssa_calc_info['gearRatio'][:ssa_calc_info['shaftIndex'] + 1]))
            stat_factor_calc_info['stepNums'].append(stat_factor_calc_info['revNums'][i] * (
                    1 - stat_factor_calc_info['overlapRatio']))
    stat_factor_calc_info['indicatorUnit'] = list()
    stat_factor_calc_info['refValue'] = task_info['refValue']
    stat_factor_calc_info['indicatorNestedList'] = list()

    # if "Speed" in stat_factor_calc_info["indicatorList"]:
    #     stat_factor_calc_info["indicatorList"].remove("Speed")
    # for channel_index, channel_name in enumerate(task_info['channelNames']):
    #     stat_factor_calc_info["indicatorNestedList"].append(stat_factor_calc_info['indicatorList'])
    # for unit_index in task_info["indicatorsUnitChanIndex"]:
    #     temp_unit_list = list()
    #     for indicator in stat_factor_calc_info['indicatorList']:
    #         if indicator == 'RMS':
    #             if basic_info['dBFlag']:
    #                 temp_unit_list.append('dB')
    #             else:
    #                 temp_unit_list.append(task_info['units'][unit_index])
    #         elif indicator == 'SPL(A)':
    #             temp_unit_list.append('dB(A)')
    #         elif indicator == 'SPL':
    #             temp_unit_list.append('dB')
    #         else:
    #             temp_unit_list.append('')
    #     stat_factor_calc_info['indicatorUnit'].append(temp_unit_list)
    # stat_factor_calc_info['gearName'] = ssa_calc_info['gearName']
    # stat_factor_calc_info['indicatorNum'] =[len(indicatorList) for indicatorList in stat_factor_calc_info["indicatorNestedList"]]
    # stat_factor_calc_info['xName'] = xName
    # stat_factor_calc_info['xUnit'] = xUnit

    if "Speed" in stat_factor_calc_info["soundIndicatorList"]:
        stat_factor_calc_info["soundIndicatorList"].remove("Speed")
    if "Speed" in stat_factor_calc_info["vibrationIndicatorList"]:
        stat_factor_calc_info["vibrationIndicatorList"].remove("Speed")
    for channel_index, channel_name in enumerate(task_info['channelNames']):
        if channel_name.lower().startswith('vib'):
            stat_factor_calc_info["indicatorNestedList"].append(
                stat_factor_calc_info['vibrationIndicatorList'])
        elif channel_name.lower().startswith('mic') or channel_name.lower().startswith("umic"):
            stat_factor_calc_info["indicatorNestedList"].append(
                stat_factor_calc_info['soundIndicatorList'])
    for i, unit_index in enumerate(task_info["indicatorsUnitChanIndex"]):
        temp_unit_list = list()
        for indicator in stat_factor_calc_info['indicatorNestedList'][i]:
            if indicator == 'RMS':
                if basic_info['dBFlag']:
                    temp_unit_list.append('dB')
                else:
                    temp_unit_list.append(task_info['units'][unit_index])
            elif indicator == 'SPL(A)':
                temp_unit_list.append('dB(A)')
            elif indicator == 'SPL':
                temp_unit_list.append('dB')
            else:
                temp_unit_list.append('')
        stat_factor_calc_info['indicatorUnit'].append(temp_unit_list)
    stat_factor_calc_info['gearName'] = ssa_calc_info['gearName']
    stat_factor_calc_info['indicatorNum'] = [len(indicatorList) for indicatorList in
                                             stat_factor_calc_info["indicatorNestedList"]]
    stat_factor_calc_info['xName'] = xName
    stat_factor_calc_info['xUnit'] = xUnit

    return stat_factor_calc_info


if __name__ == '__main__':
    import os
    from common_info import config_folder, config_file, encrypt_flag
    import time

    type_info = "ktz999x_cj"
    config_filename = os.path.join(config_folder, "_".join([type_info, config_file]))
