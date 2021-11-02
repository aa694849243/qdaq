# -*- coding: utf-8 -*-
"""
Created on Mon Mar 9

@author: Wall@Synovate

function: define the global variable of TTL and 2 vibrations

"""
import logging
import traceback
import sys


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

def set_default_flag():
    _global_dict = dict()
    # set flag to start the process or thread
    _global_dict['dataAcquisition'] = True
    _global_dict['speedCalclation'] = True
    _global_dict['nvhCalclation'] = True
    _global_dict['dataPack'] = True
    _global_dict['rawDataWrite'] = True

    return _global_dict


def set_default_status():
    _global_dict = dict()
    # status for front real time showing page
    _global_dict['code'] = None
    _global_dict["result"] = None
    _global_dict['msg'] = None
    _global_dict['data'] = {"type": "", "serialNo": "", "testName": list(), "startX": list(), "startY": list(), "endX":
                            list(), "endY": list(), "x": 0.0, "y": 0.0}
    _global_dict['xml'] = list()
    _global_dict['sensitivity'] = {'sensitivity': list(), 'rawData': list()}
    return _global_dict


def set_default_common():
    _global_dict = dict()
    # store common value for data processing(used in multi processes)
    _global_dict['revCount'] = None
    _global_dict['sampleRate'] = None
    _global_dict['sampsPerChan'] = None
    _global_dict['fileName'] = None

    return _global_dict


def set_default_rawdata():
    _global_dict = dict()
    # to store raw data
    _global_dict['speed'] = list()
    _global_dict['vib1'] = list()
    _global_dict['vib2'] = list()

    return _global_dict


def set_default_rawdata_with_channelNames(channel_names):
    _global_dict = dict()
    # to store raw data

    for channel_name in channel_names:
        _global_dict[channel_name] = list()

    return _global_dict

def set_default_speeddata():
    _global_dict = dict()
    # store speed curve
    _global_dict['RPML'] = list([0.0])
    _global_dict['RPM'] = list([0.0])

    return _global_dict

def set_default_tempspeed():
    _global_dict = dict()
    # store the speed curve of present test
    _global_dict['rpml'] = list()
    _global_dict['rpm'] = list()

    return _global_dict

def set_default_nvhdata():
    _global_dict = dict()
    # store nvh calculation info
    _global_dict['calSize'] = None
    _global_dict['freq'] = None
    _global_dict['arPoints'] = None
    _global_dict['revCount'] = None
    _global_dict['threedOS'] = list()
    _global_dict['threedOS'].append(list())
    _global_dict['threedOS'].append(list())
    _global_dict['tempThreedOS'] = list()

    return _global_dict


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(pathname)s %(funcName)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='[%Y-%m-%d %H:%M:%S]',
        filename='D:/qdaq/Log/temp.log',
        filemode='a'
    )
    try:
        a = set_default_rawdata()
        for i in range(10):
            a['vib1'].append(i)
        print(a)
    except Exception:
        traceback.print_exc()
        logging.warning("exec failed, failed msg:" + traceback.format_exc())
        sys.exit()
