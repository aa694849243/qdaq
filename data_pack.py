import threading
from multiprocessing import shared_memory
from queue import Empty
import time
import os
import traceback

import numpy as np

from common_info import limit_folder, limit_file, sensor_count, flag_index_dict, Cryptor, encrypt_flag, \
    qDAQ_logger
from ml_qdaq_predict import ml_qdaq
from utils import write_json, read_json, send_result_data, decrypt_data
from limit_compare_new import comparator
from json2xml import dict_to_xml, xml_write
import gc

allCount = 0
abnormalCount = 0
unqualifiedCount = 0
is_simu_mode = None


def datapack_process(Q_datapack_in, Q_datapack_out, Q_nvh_datapack):
    global gv_dict_flag, is_simu_mode, allCount, abnormalCount, unqualifiedCount
    try:
        # 连接flag的共享内存
        shm_flag = shared_memory.SharedMemory(name="shm_flag")
        gv_dict_flag = shm_flag.buf
    except Exception:
        qDAQ_logger.error("datapack process memory error, error msg:" + traceback.format_exc())
    while True:
        data = Q_datapack_in.get()
        if data == "reset":
            allCount = 0
            abnormalCount = 0
            unqualifiedCount = 0
            continue
        is_simu_mode = data["is_simu_mode"]
        try:

            data["gv_dict_status"]["allCount"] = allCount
            data["gv_dict_status"]["abnormalCount"] = abnormalCount
            data["gv_dict_status"]["unqualifiedCount"] = unqualifiedCount
            dataPack(Q_nvh_datapack, data['gv_dict_status'], data['param'],
                     data['test_result'], data['time_click_start'])
        except Exception:
            qDAQ_logger.error(traceback.format_exc())
            data['gv_dict_status']["code"] = 3000
            data['gv_dict_status']["msg"] = "datapack进程出现错误!"
            gv_dict_flag[flag_index_dict["datapack_error"]] = 1
        qDAQ_logger.info("data pack stop")
        Q_datapack_out.put({"datapack": 1})
        del data


def dataPack(Q4, gv_dict_status, param, test_result, time_click_start):
    global gv_dict_flag, allCount, abnormalCount, unqualifiedCount, is_simu_mode

    result_code_dict = dict()
    result_code_dict[-2] = "不参与界限值比较"
    result_code_dict[-1] = "界限值缺失"
    result_code_dict[0] = "不合格(超上限)"
    result_code_dict[1] = "合格"
    result_code_dict[2] = "异常(rms超下限，信号异常)"
    result_code_dict[3] = "次异常(其他指标超下限)"

    gc.enable()
    gv_dict_status_temp = dict()
    # m个测试段 n个传感器 m*n个智能预测结果的list
    allIntelligenceStatus = list()
    allIntelligenceDefectDescription = list()

    # pack the data
    counter_test = 0

    qDAQ_logger.debug("dataPack:pid={},ppid={},thread={}".format(os.getpid(), os.getppid(),
                                                                 threading.current_thread().name))

    limits_filename = "_".join([param.basicInfo["type"], limit_file])
    full_limit_path = os.path.join(limit_folder, limits_filename)
    # read in the limit if existed
    # 确认界限值文件是否存在，若存在则读取
    qDAQ_logger.info("*" * 30 + str(gv_dict_status["data"]["testResult"]) + "&" * 30)
    if os.path.exists(full_limit_path):
        qDAQ_logger.info("limit found, file name: " + full_limit_path)
        if encrypt_flag:
            limits = decrypt_data(Cryptor, full_limit_path)
        else:
            limits = read_json(full_limit_path)
    else:
        qDAQ_logger.info("limit not found, file name: " + full_limit_path)
        limits = None

    # 从nvh拿到的数据是第几个测试段
    testNameIndex = 0
    error_flag = 0
    while gv_dict_flag[flag_index_dict['dataPack']]:
        for i in range(sensor_count):
            while gv_dict_flag[flag_index_dict['dataPack']]:
                try:
                    result = Q4.get(timeout=1)

                    test_result["resultData"][result["sensorIndex"]]["dataSection"][
                        result["testNameIndex"]] = \
                        result["data"]["resultData"][result["sensorIndex"]]["dataSection"][
                            result["testNameIndex"]]
                    testNameIndex = result["testNameIndex"]
                    if result["error"]:
                        error_flag = 1
                    break
                except Empty:
                    continue
        # 点击stop要跳出上面的for循环，此时没有结果数据，需要跳出最外层的while循环
        if not gv_dict_flag[flag_index_dict['dataPack']]:
            break
        # 界限值比较
        qDAQ_logger.info("limit compare start")
        if limits:
            try:
                # 界限值评判（按测试段）
                test_result = comparator(test_result, limits, param.speedRecogInfo['testName'][
                    testNameIndex], param.limitCompareFlag, param.onedOSCalcInfo)
            except Exception:
                gv_dict_status["code"] = 3000
                gv_dict_status["msg"] = "界限值比较错误!"
                traceback.print_exc()
                qDAQ_logger.error("limit compare failed, failed msg:" + traceback.format_exc())
                break

        qDAQ_logger.info("limit compare finsih")

        # 智能预测
        sectionMlResult = dict()
        try:
            for sensor_id in range(sensor_count):
                if not error_flag:
                    # 没有错误才能进行评判
                    intelligenceResult = ml_qdaq(test_result["type"], test_result["systemNo"],
                                                 str(sensor_id),
                                                 test_result["resultData"][sensor_id]["dataSection"][
                                                     testNameIndex][
                                                     "twodOS"][0]["yValue"], True, True)
                    intelligenceResult["ml_sensor"] = int(intelligenceResult["ml_sensor"])
                    intelligenceResult["ml_quality"] = float(intelligenceResult["ml_quality"])
                    test_result["resultData"][sensor_id]["dataSection"][testNameIndex]["twodOS"][
                        0].update(intelligenceResult)
                    # 1-bad 0-good -1 no-mark
                    allIntelligenceStatus.append(intelligenceResult["ml_sensor"])
                    # 1-bad 0-good -1 no-mark 0-1中的double数据
                    allIntelligenceDefectDescription.append(intelligenceResult["ml_quality"])
                    qDAQ_logger.debug(test_result["resultData"][sensor_id]["sensorId"])
                    sectionMlResult[test_result["resultData"][sensor_id]["sensorId"]] = intelligenceResult["ml_sensor"]
        except Exception:
            traceback.print_exc()
            qDAQ_logger.error("ml failed, failed msg:" + traceback.format_exc())

        # 更新该测试段的指标结果，返回给前端
        gv_dict_status_temp["sectionResult"] = gv_dict_status["sectionResult"]
        gv_dict_status_temp["sectionResult"]["testName"].append(
            test_result["resultData"][0]["dataSection"][
                result["testNameIndex"]]["testName"])
        sectionLimitResult = dict()
        for i in range(sensor_count):
            sectionLimitResult[test_result["resultData"][i]["sensorId"]] = \
                test_result["resultData"][i]["dataSection"][testNameIndex]["testResult"]
        gv_dict_status_temp["sectionResult"]["limitComResult"].append(sectionLimitResult)
        gv_dict_status_temp["sectionResult"]["mlResult"].append(sectionMlResult)
        gv_dict_status["sectionResult"] = gv_dict_status_temp["sectionResult"]

        try:
            gv_dict_status_temp['xml'] = gv_dict_status['xml']
            revCount = len(test_result["resultData"][result["sensorIndex"]]["dataSection"][
                               result["testNameIndex"]]["twodOC"][0]["xValue"])
            qDAQ_logger.debug("rev num of test index: {} is {}".format(counter_test, revCount))
            gv_dict_status_temp["xml"].append(
                dict_to_xml(test_result, testNameIndex, param.orderSpectrumCalcInfo[testNameIndex]["orderResolution"],
                            int(param.orderSpectrumCalcInfo[testNameIndex]["revNum"] * (
                                        1 - param.orderSpectrumCalcInfo[testNameIndex]["overlapRatio"])), revCount))
            gv_dict_status['xml'] = gv_dict_status_temp['xml']
            if param.dataSaveFlag['xml']:
                xml_filename = os.path.join(param.folderInfo["temp"], param.speedRecogInfo['testName'][
                    testNameIndex] + '_' + gv_dict_status["data"]['type'] + '_' + param.basicInfo[
                                                "fileName"] + '.xml')
                xml_write(gv_dict_status["xml"][-1], xml_filename)
                qDAQ_logger.info(f"xml of test index: {counter_test} saved, filename: {xml_filename}")
            else:
                qDAQ_logger.info(f"xml of test index: {counter_test} xml not saved")
        except Exception:
            gv_dict_status["code"] = 3000
            gv_dict_status["msg"] = "xml生成出错!"
            qDAQ_logger.error("xml creation failed, failed msg:" + traceback.format_exc())
            break

        counter_test += 1

        del result

        if counter_test < param.speedRecogInfo["test_count_except_dummy"]:
            continue
        else:
            qDAQ_logger.info("all xml finished at time:{}".format(time.time()))
            # 以下内容需要在有完整的结果数据之后才能进行，所以要先执行下面的代码，再break，
            # 若先break，若nvh出错，点击stop，这里照样会跳出while flag[nvh]循环，此时是没有结果数据的

            # 智能预测结果与界限值比较结果的维护
            # 目前智能预测的结果不可靠，故只维护intelligenceStatus和intelligenceDefectDescription字段
            # 维护results
            # 维护 testResult
            # 维护 resultBySensor
            # 维护 intelligenceStatus 存在不合格的则认为不合格,没有不合格时存在合格为合格，全部no-mark才是no-mark
            if len(allIntelligenceStatus) != 0:
                test_result["intelligenceStatus"] = max(allIntelligenceStatus)
            # 维护 intelligenceDefectDescription 取最差的那个
            if len(allIntelligenceDefectDescription) != 0:
                test_result["intelligenceDefectDescription"] = max(allIntelligenceDefectDescription)

            # 维护 overallResult
            # 更新缺陷记录的描述（即哪些指标不合格）
            if test_result["qdaqDefectDescription"]:
                test_result["qdaqDefectDescription"] = ", ".join(set(test_result["qdaqDefectDescription"]))
            else:
                test_result["qdaqDefectDescription"] = ""

            # 统计实采模式下一共检测的数量，以及异常数量，不合格数量
            if not is_simu_mode:
                allCount += 1
                # 异常
                if test_result["overallResult"] in [2, 3]:
                    abnormalCount += 1
                # 不合格
                if test_result["overallResult"] == 0:
                    unqualifiedCount += 1

            gv_dict_status["allCount"] = allCount
            gv_dict_status["abnormalCount"] = abnormalCount
            gv_dict_status["unqualifiedCount"] = unqualifiedCount

            gv_dict_status["code"] = 4

            gv_dict_status['result'] = test_result["overallResult"]

            gv_dict_status["msg"] = "界限值比较完成！"
            qDAQ_logger.debug("开始检测到界限值比较完成:{}".format(time.time() - time_click_start))

            # # 先复制到temp，再修改temp，再复制回去
            # gv_dict_status_temp['data'] = gv_dict_status['data']
            # gv_dict_status_temp["data"]["testResult"] = test_result["overallResult"]
            # gv_dict_status['data'] = gv_dict_status_temp['data']

            qDAQ_logger.info('Limit compare finished!')
            result_json_filename = gv_dict_status["data"]["type"] + '_' + param.basicInfo[
                "fileName"] + '.json'
            # 将result_json中的数据不可json化的变为可以json化的格式

            if error_flag:
                # 数据存在错误则保存到info missing文件夹
                result_json_filepath = os.path.join(
                    param.folderInfo["reportInfoMissing"], result_json_filename)
                write_json(result_json_filepath, test_result)
                qDAQ_logger.info(
                    "result Json already saved, filename: " + result_json_filepath)
            else:
                # 数据无误则发送
                report_path = send_result_data(param.sendResultInfo, test_result, result_json_filename,
                                               param.folderInfo)
                gv_dict_status["code"] = 5
                gv_dict_status_temp['data'] = gv_dict_status['data']
                # gv_dict_status_temp["data"]["testResult"] = test_result["overallResult"]
                gv_dict_status_temp["data"]["reportPath"] = report_path
                gv_dict_status['data'] = gv_dict_status_temp['data']
                if report_path:
                    gv_dict_status["msg"] = "结果数据上传完成！"
                    qDAQ_logger.info("*" * 30 + str(gv_dict_status["data"]["testResult"]) + "&" * 30)
                    qDAQ_logger.info('data send succeed!')
                else:
                    gv_dict_status["msg"] = "结果数据上传失败！"
                    qDAQ_logger.info('data send error!')
                if param.dataSaveFlag['resultJson']:
                    result_json_filepath = os.path.join(param.folderInfo["temp"], result_json_filename)
                    write_json(result_json_filepath, test_result)
                    qDAQ_logger.info("result Json saved, filename: " + result_json_filepath)
                else:
                    qDAQ_logger.info("result Json not saved!")
                # 整个测试的所有测试段pack完成，datapack进程结束
                gv_dict_flag[flag_index_dict["datapack_finish"]] = 1
                break

    qDAQ_logger.info("datapack finish")
    del counter_test
    del limits_filename
    del limits
    del gv_dict_status_temp
    # gc.collect()
