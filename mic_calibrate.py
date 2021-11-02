import numpy as np


def mic_setdefault(mic_parameter):
    if 'currentExcitVal' in mic_parameter:
        mic_parameter['current_excit_val'] = mic_parameter['currentExcitVal']
    mic_parameter.setdefault('time', 1)
    mic_parameter.setdefault("bufferSize", 1000000)
    mic_parameter.setdefault("channelNames", ['Mic'])
    mic_parameter.setdefault("channelType", ["Sound"])
    mic_parameter.setdefault("current_excit_val", [0.0021])
    mic_parameter.setdefault("maxVal", [100])
    mic_parameter.setdefault("minVal", [0])
    mic_parameter.setdefault("physicalChannels", ["ai0"])
    mic_parameter.setdefault("sampleRate", 102400)
    mic_parameter.setdefault("sampsPerChan", 16384)
    mic_parameter.setdefault("sensitivity", [1])  # 不开放给用户
    mic_parameter.setdefault("units", ["Pa"])  # 不开放给用户
    mic_parameter.setdefault("timeout", 120)  # 不开放给用户
    mic_parameter.setdefault("sPL", 114)  # 有效声压


def cal_sensity(p, array) -> float:
    """
    p: 声压
    array: 传入的数组
    """
    return np.sqrt(np.sum(np.power(array, 2)) / len(array)) / p


