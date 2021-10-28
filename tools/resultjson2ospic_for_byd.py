import json
import matplotlib.pyplot as plt
import numpy as np
import os
from icecream import ic

if __name__ == '__main__':

    path="D:\\qdaq\\对比\\byd_selectspeed"
    filename1=os.path.join(path,"resolver-ttl-3vibs_PM61191-jinkang-3vibs_210913081533.json")
    filename2=os.path.join(path,"ttl-resolver-3vibs_PM61191-jinkang-3vibs_210914015954.json")
    filename2=os.path.join(path,"ttl-resolver-3vibs-speedRatio_PM61191-jinkang-3vibs_210924093627.json")
    filename3=os.path.join(path,"PM61191-2vibs-resolver_PM61191-2vibs_210910093211.json")
    filename4=os.path.join(path,"jinkang-2vibs-byd_jinkang-2vibs_210913093109.json")
    # filename2=os.path.join(path,"ttl-resolver-3vibs_PM61191-jinkang-3vibs-filter2.56_210924084858.json")

    with open(filename1,encoding="utf-8") as f:
        data1 = json.load(f)
    with open(filename2,encoding="utf-8") as f:
        data2 = json.load(f)
    with open(filename3,encoding="utf-8") as f:
        data3 = json.load(f)
    with open(filename4,encoding="utf-8") as f:
        data4 = json.load(f)

    sensor_list=range(1)
    test_list=range(6)

    os_dict=dict()

    # test_list=[0,3]

    # for sensor_id in sensor_list:
    #     for test_id in test_list:
    #         rms1=data1["resultData"][sensor_id]["dataSection"][test_id]["twodTD"][0]
    #         rms3 = data3["resultData"][sensor_id]["dataSection"][test_id]["twodTD"][0]
    #         plt.figure("sensor:{} test:{}".format(sensor_id,test_id))
    #         plt.plot(rms1["xValue"],np.array(rms1["yValue"]),c="r",label="const")
    #         plt.plot(rms3["xValue"],np.array(rms3["yValue"]),c="g",label="v3")
    # 先旋变信号后ttl信号


    for sensor_id in range(2):
        for test_id in range(3):
            orderSpectrum1=data1["resultData"][sensor_id]["dataSection"][test_id]["twodOS"][0]
            orderSpectrum3=data3["resultData"][sensor_id]["dataSection"][test_id]["twodOS"][0]
            plt.figure("resover-ttl PM61191  sensor:{} test:{}".format(sensor_id,test_id))
            plt.title("orderSpectrum")
            plt.plot(orderSpectrum1["xValue"],np.array(orderSpectrum1["yValue"]),c="r",label="pinjie")
            plt.plot(orderSpectrum3["xValue"],np.array(orderSpectrum3["yValue"]),c="g",label="huichuan")
            plt.legend()

    plt.figure("resover-ttl jinkang")

    plt.title("orderSpectrum")
    orderSpectrum1 = data1["resultData"][2]["dataSection"][3]["twodOS"][0]
    orderSpectrum4 = data4["resultData"][0]["dataSection"][1]["twodOS"][0]
    plt.plot(orderSpectrum1["xValue"], np.array(orderSpectrum1["yValue"]), c="r", label="pinjie")
    plt.plot(orderSpectrum4["xValue"], np.array(orderSpectrum4["yValue"]), c="g", label="jinkang")



    plt.figure("ttl-resolver-jinkang sensor 0 test0")
    orderSpectrum2=data2["resultData"][2]["dataSection"][0]["twodOS"][0]
    orderSpectrum4=data4["resultData"][0]["dataSection"][0]["twodOS"][0]
    plt.plot(orderSpectrum4["xValue"], np.array(orderSpectrum4["yValue"]), c="g", label="jinkang")
    plt.plot(orderSpectrum2["xValue"], np.array(orderSpectrum2["yValue"]), c="r", label="pinjie")

    plt.figure("ttl-resolver-jinkang sensor 0 test 1")
    orderSpectrum2=data2["resultData"][2]["dataSection"][1]["twodOS"][0]
    orderSpectrum4=data4["resultData"][0]["dataSection"][1]["twodOS"][0]
    plt.plot(orderSpectrum4["xValue"], np.array(orderSpectrum4["yValue"]), c="g", label="jinkang")
    plt.plot(orderSpectrum2["xValue"], np.array(orderSpectrum2["yValue"]), c="r", label="pinjie")

    for sensor_id in range(2):
        for test_id in range(2,4):
            orderSpectrum2=data2["resultData"][sensor_id]["dataSection"][test_id]["twodOS"][0]
            orderSpectrum3=data3["resultData"][sensor_id]["dataSection"][test_id+2]["twodOS"][0]
            plt.figure("ttl-resolver-PM61191  sensor:{} test:{}".format(sensor_id,test_id))
            plt.title("orderSpectrum")
            plt.plot(orderSpectrum2["xValue"],np.array(orderSpectrum2["yValue"]),c="r",label="pinjie")
            plt.plot(orderSpectrum3["xValue"],np.array(orderSpectrum3["yValue"]),c="g",label="huichuan")
            plt.legend()


    plt.legend()
    plt.show()