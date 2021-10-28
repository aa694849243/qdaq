#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2020/10/21 0:00
@Author : Sergei@Synovate
@Email  : qxu@sonustc.com
readme: no encode of 2D data
"""

import json
import time
from datetime import datetime, timedelta
from xml.dom import minidom
import logging
import traceback
import sys


def dict_to_xml(data, test_number, dO, revStep, revCount):
    """
    功能： 制定测试段的结果数据转XML（二维数据不进行重编码处理），基于测试段
    输入：
    1. 结果数据
    2. 测试段索引
    3. 阶次谱分辨率
    4. 阶次谱计算时的步进圈数
    5. 该测试段累计圈数
    返回：xml对象
    input: data=dict, dO=double, revStep=int, revCount=int
    return: string
    """

    def time_stamp(data):
        time_json = time.strptime(data["time"], '%Y-%m-%d %H:%M:%S')
        timestamp = int((datetime(time_json.tm_year, time_json.tm_mon, time_json.tm_mday, time_json.tm_hour,
                                  time_json.tm_min, time_json.tm_sec) + timedelta(hours=8)).timestamp())

        return str(timestamp)

    def base64_converter(data):
        ret_str = ';'.join(map(str, data))
        return ret_str

    def test_diagnostic(data):
        status = "passed"
        if data["overallResult"] == -1 or data["overallResult"] == 0:
            status = "failed"
        return status

    def indicator_diagnostic(data):
        status = "passed"
        if data == -1 or data == 0:
            status = "failed"
        return status

    def oned_indicators(data, test_number):
        doc = minidom.Document()
        root = doc.createElement('Indicators')

        try:
            for sensor in range(0, len(data["resultData"])):
                oned_data = data["resultData"][sensor]["dataSection"][test_number]["onedData"]
                for i in range(0, len(oned_data)):
                    if len(data["resultData"]) == 1:
                        indicator = doc.createElement('ATEOLIndicator')
                        indicator.setAttribute('Name', oned_data[i]["name"])
                        indicator.setAttribute('Value', str(oned_data[i]["value"]))
                        indicator.setAttribute('Lo', "NaN")
                        indicator.setAttribute('Hi', "NaN")
                        indicator.setAttribute('IndicatorDiagnostic',
                                               indicator_diagnostic(oned_data[i]["indicatorDiagnostic"]))
                        indicator.setAttribute('Unit', oned_data[i]["unit"])
                        root.appendChild(indicator)
                    else:
                        indicator = doc.createElement('ATEOLIndicator')
                        indicator.setAttribute('Name', oned_data[i]["name"] + "-vib" + str(sensor + 1))
                        indicator.setAttribute('Value', str(oned_data[i]["value"]))
                        indicator.setAttribute('Lo', "NaN")
                        indicator.setAttribute('Hi', "NaN")
                        indicator.setAttribute('IndicatorDiagnostic',
                                               indicator_diagnostic(oned_data[i]["indicatorDiagnostic"]))
                        indicator.setAttribute('Unit', oned_data[i]["unit"])
                        root.appendChild(indicator)
        except:
            pass
        return root

    def rms2d_indicators(data, test_number):
        doc = minidom.Document()
        root = doc.createElement('RMS2D')

        try:
            for sensor in range(0, len(data["resultData"])):
                rms2d_data = data["resultData"][sensor]["dataSection"][test_number]["twodTD"]
                for i in range(0, len(rms2d_data)):
                    if len(data["resultData"]) == 1:
                        indicator = doc.createElement('ATEOL2DIndicatorXY')
                        indicator.setAttribute('Name', rms2d_data[i]["yName"])
                        indicator.setAttribute('IndicatorDiagnostic',
                                               indicator_diagnostic(rms2d_data[i]["indicatorDiagnostic"]))
                        indicator.setAttribute('Unit', rms2d_data[i]["yUnit"])
                        indicator.setAttribute('XName', rms2d_data[i]["xName"])
                        indicator.setAttribute('XUnit', rms2d_data[i]["xUnit"])

                        values = doc.createElement('Values')
                        values_enc = doc.createTextNode(base64_converter(rms2d_data[i]["yValue"]))
                        values.appendChild(values_enc)
                        indicator.appendChild(values)

                        x = doc.createElement('X')
                        x_enc = doc.createTextNode(base64_converter(rms2d_data[i]["xValue"]))
                        x.appendChild(x_enc)
                        indicator.appendChild(x)

                        root.appendChild(indicator)
                    else:
                        indicator = doc.createElement('ATEOL2DIndicatorXY')
                        indicator.setAttribute('Name', rms2d_data[i]["yName"] + "-vib" + str(sensor + 1) + "_2D")
                        indicator.setAttribute('IndicatorDiagnostic',
                                               indicator_diagnostic(rms2d_data[i]["indicatorDiagnostic"]))
                        indicator.setAttribute('Unit', rms2d_data[i]["yUnit"])
                        indicator.setAttribute('XName', rms2d_data[i]["xName"])
                        indicator.setAttribute('XUnit', rms2d_data[i]["xUnit"])

                        values = doc.createElement('Values')
                        values_enc = doc.createTextNode(base64_converter(rms2d_data[i]["yValue"]))
                        values.appendChild(values_enc)
                        indicator.appendChild(values)

                        x = doc.createElement('X')
                        x_enc = doc.createTextNode(base64_converter(rms2d_data[i]["xValue"]))
                        x.appendChild(x_enc)
                        indicator.appendChild(x)

                        root.appendChild(indicator)
        except:
            pass

        return root

    def oa2d_indicators(data, test_number):
        doc = minidom.Document()
        root = doc.createElement('OA2D')
        first_level_indicator = doc.createElement('Indicators')

        try:
            for sensor in range(0, len(data["resultData"])):
                oa2d_data = data["resultData"][sensor]["dataSection"][test_number]["twodOC"]
                for i in range(0, len(oa2d_data)):
                    if len(data["resultData"]) == 1:
                        indicator = doc.createElement('ATEOL2DIndicator')
                        indicator.setAttribute('Name', oa2d_data[i]["yName"])
                        indicator.setAttribute('IndicatorDiagnostic',
                                               indicator_diagnostic(oa2d_data[i]["indicatorDiagnostic"]))
                        indicator.setAttribute('Unit', oa2d_data[i]["yUnit"])

                        values = doc.createElement('Values')
                        values_enc = doc.createTextNode(base64_converter(oa2d_data[i]["yValue"]))
                        values.appendChild(values_enc)
                        indicator.appendChild(values)

                        first_level_indicator.appendChild(indicator)

                        root.appendChild(first_level_indicator)
                    else:
                        indicator = doc.createElement('ATEOL2DIndicator')
                        indicator.setAttribute('Name', oa2d_data[i]["yName"] + "-vib" + str(sensor + 1) + "_2D")
                        indicator.setAttribute('IndicatorDiagnostic',
                                               indicator_diagnostic(oa2d_data[i]["indicatorDiagnostic"]))
                        indicator.setAttribute('Unit', oa2d_data[i]["yUnit"])

                        values = doc.createElement('Values')
                        values_enc = doc.createTextNode(base64_converter(oa2d_data[i]["yValue"]))
                        values.appendChild(values_enc)
                        indicator.appendChild(values)

                        first_level_indicator.appendChild(indicator)

                        root.appendChild(first_level_indicator)

            x = doc.createElement('X')
            x_enc = doc.createTextNode(
                base64_converter(data["resultData"][0]["dataSection"][test_number]["twodOC"][0]["xValue"]))
            x.appendChild(x_enc)
            root.appendChild(x)
            root.setAttribute('XName', data["resultData"][0]["dataSection"][test_number]["twodOC"][0]["xName"])
            root.setAttribute('XUnit', data["resultData"][0]["dataSection"][test_number]["twodOC"][0]["xUnit"])
        except:
            pass

        return root

    def os_data(data, test_number, dO, revStep, revCount):
        doc = minidom.Document()
        root = doc.createElement('OrderSpectrums')

        try:
            for sensor in range(0, len(data["resultData"])):
                os = data["resultData"][sensor]["dataSection"][test_number]["twodOS"]
                for i in range(0, len(os)):
                    if len(data["resultData"]) == 1:
                        indicator = doc.createElement('ATEOLOSIndicator')
                        indicator.setAttribute('Name', os[i]["yName"])
                        indicator.setAttribute('dO', str(dO))
                        indicator.setAttribute('revStep', str(revStep))
                        indicator.setAttribute('revCount', str(revCount))
                        indicator.setAttribute('IndicatorDiagnostic',
                                               indicator_diagnostic(os[i]["indicatorDiagnostic"]))

                        values = doc.createElement('Spectrum')
                        values_enc = doc.createTextNode(base64_converter(os[i]["yValue"]))
                        values.appendChild(values_enc)
                        indicator.appendChild(values)

                        root.appendChild(indicator)
                    else:
                        indicator = doc.createElement('ATEOLOSIndicator')
                        indicator.setAttribute('Name', os[i]["yName"] + "-vib" + str(sensor + 1) + "_2D")
                        indicator.setAttribute('dO', str(dO))
                        indicator.setAttribute('revStep', str(revStep))
                        indicator.setAttribute('revCount', str(revCount))
                        indicator.setAttribute('IndicatorDiagnostic',
                                               indicator_diagnostic(os[i]["indicatorDiagnostic"]))

                        values = doc.createElement('Spectrum')
                        values_enc = doc.createTextNode(base64_converter(os[i]["yValue"]))
                        values.appendChild(values_enc)
                        indicator.appendChild(values)

                        root.appendChild(indicator)
        except:
            pass

        return root

    def header(data, dO, revStep, revCount, test_number):
        doc = minidom.Document()
        try:
            root = doc.createElement('ATEOLResults')
            root.setAttribute('xmlns:xsd', "http://www.w3.org/2001/XMLSchema")
            root.setAttribute('xmlns:xsi', "http://www.w3.org/2001/XMLSchema-instance")
            root.setAttribute('SerialNo', data["serialNo"])
            root.setAttribute('DateTimeUTC', time_stamp(data))
            doc.appendChild(root)
            results = doc.createElement('Results')
            root.appendChild(results)
            ATEOLResult = doc.createElement("ATEOLResult")
            ATEOLResult.setAttribute('xsi:type', "ATEOLResultRamp")
            ATEOLResult.setAttribute('InstanceName', "ATEOL")
            ATEOLResult.setAttribute('TestName', data["resultData"][0]["dataSection"][test_number]["testName"])
            ATEOLResult.setAttribute('TestDiagnostic', test_diagnostic(data))
            ATEOLResult.setAttribute('TestSensorState',
                                     data["resultData"][0]["dataSection"][test_number]["testSensorState"])
            results.appendChild(ATEOLResult)

            ATEOLResult.appendChild(oned_indicators(data, test_number))

            ATEOLResult.appendChild(rms2d_indicators(data, test_number))

            ATEOLResult.appendChild(oa2d_indicators(data, test_number))

            ATEOLResult.appendChild(os_data(data, test_number, dO, revStep, revCount))
        except:
            pass

        xml_str = doc.toprettyxml(indent="  ")

        return xml_str

    root = header(data, dO, revStep, revCount, test_number)

    return root


def xml_write(xml, path_to_xml):
    with open(path_to_xml, "w") as f:
        f.write(xml)
    pass


def read_json(path_to_json):
    with open(path_to_json) as f:
        data = json.loads(f.read())
    return data


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(pathname)s %(funcName)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='[%Y-%m-%d %H:%M:%S]',
        filename='D:/aqs2rt/Log/temp.log',
        filemode='a'
    )
    try:
        path_to_json = r'D:\qdaq\JSON_NetError\inovance1vib-new_ino-2_210107115408.json'
        path_to_xml = r'D:\qdaq\JSON_NetError\test3.xml'

        dO, revStep, revCount = 0.03125, 8, 27
        data = read_json(path_to_json)

        xml_write(dict_to_xml(data, 0, dO, revStep, revCount), path_to_xml)
    except Exception:
        traceback.print_exc()
        logging.warning("exec failed, failed msg:" + traceback.format_exc())
        sys.exit()
