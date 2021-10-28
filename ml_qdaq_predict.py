#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 2021
App for ML
@author: Sergei Afanasev
@company: Synovate Technology
"""


import os
import pickle
import numpy as np
from scipy.signal import resample


def ml_qdaq(type_, system_no, sensor_id, os_y_data, sensor_inspection_on_off, quality_prediction_on_off):
    def create_path(level, param_):
        path = level

        if 'type' in param_:
            path += '_' + param_['type']

        if 'system_no' in param_:
            path += '_' + param_['system_no']

        if 'sensor_id' in param_:
            path += '_' + param_['sensor_id']

        return path

    def model_loader(file_path):
        model = pickle.load(open(file_path, 'rb'))
        return model

    def inspection_sensor(y_data, path):
        if sensor_inspection_on_off:
            if os.path.exists(path):
                os_data = []

                if len(y_data) != 4800:
                    os_data.append(np.abs(resample(y_data, 4800)))
                else:
                    os_data.append(y_data)

                model = model_loader(path)
                predict = model.predict_proba(os_data)
                return np.round(predict[0][1], 2)
            else:
                return -1
        else:
            return 0

    def inspection_quality(y_data, path):
        if quality_prediction_on_off:
            if os.path.exists(path):
                os_data = []

                if len(y_data) != 4800:
                    os_data.append(np.abs(resample(y_data, 4800)))
                else:
                    os_data.append(y_data)

                model = model_loader(path)
                predict = model.predict_proba(os_data)
                return np.round(predict[0][1], 2)
            else:
                return -1
        else:
            return 0

    param = {'type': type_, 'system_no': system_no, 'sensor_id': sensor_id}

    path_sensor = os.path.join('model', create_path('sensor', param))
    res_sensor = inspection_sensor(os_y_data, path_sensor)

    path_quality = os.path.join('model', create_path('quality', param))
    res_quality = inspection_quality(os_y_data, path_quality)

    res = {'ml_sensor': res_sensor, 'ml_quality': res_quality}

    return res


if __name__ == '__main__':
    result=ml_qdaq("fd", "da", "0", [1,2], True, True)
    print(result)
