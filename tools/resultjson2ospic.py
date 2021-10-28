import json
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic

if __name__ == '__main__':


    filename1 = "D:\\qdaq\\对比\\v2v3\\jinkang-2vibs-v2.2.5-1.json"
    # filename2 = "D:\\qdaq\\对比\\v2v3\\PM61191-2vibs-ppr32-6462-float.json"
    filename3 = "D:\\qdaq\\对比\\v2v3\\jinkang-2vibs-v3-1.json"
    # filename2 = "D:\\qdaq\\对比\\qdaqv2.0和v3.0对比\\PM61191-1vib_PM61191-1vib_210719073127.json"
    filename2 = "D:\\qdaq\\对比\\qdaqv2.0和v3.0对比\\PM61191-1vibs-resolver2-6tests-v2.2.5-1.json"

    filename3 = "D:\\qdaq\\对比\\const\\KTZ66X32S-alltests-resolver2_byd-alltests_210729085522.json"
    # filename1 = "D:\\qdaq\\对比\\v2v3\\PM61191-2vibs-ppr32-6462-float.json"
    # filename1 = "D:\\qdaq\\对比\\v2v3\\PM61191-2vibs-resolver2-v3.0.json"
    filename1 = "D:\\qdaq\\对比\\const\\canshupeizhizuixin.json"
    # filename1 = "D:\\qdaq\\对比\\xuanbiandianji\\xuanbianhengsu_demo1_210816064115.json"

    # filename3 = "D:\\qdaq\\对比\\v2v3\\PM61191-2vibs-resolver2-v2.json"

    filename1="D:/qdaq/temp/105K2_105OK3-1_210826093151_210902060422.json"
    filename1="D:/qdaq/对比/insert0/3150-8192_1OK-1_210826074336_210903021009.json"
    filename2="D:/qdaq/对比/insert0/3150-8192_1OK-1_210826074336_210903021348.json"
    # filename3="D:/qdaq/对比/insert0/3150-65536_1OK-1_210826074336_210903020937.json"
    filename3="D:/qdaq/对比/insert0/3150-8192_1OK-1_210826074336_210903021241.json"
    filename3="D:/qdaq/对比/insert0/3150-66000_1OK-1_210826074336_210903022808.json"
    filename1 = "D:\\qdaq\\对比\\fluctuation_norsp\\KTZ66X32S-alltests-resolver2_byd-alltests_210908032939.json"
    filename3 = "D:\\qdaq\\对比\\fluctuation_norsp\\KTZ66X32S-alltests-const-norsp_byd-alltests_210908032601-longframe.json"

    filename1=r"D:\qdaq\对比\danluxuanbianxin\KTZ66X32S-alltests-resolver-32-64-62_byd-alltests_210924060240.json"
    filename3=r"D:\qdaq\对比\danluxuanbianxin\KTZ66X32S-alltests-resolver2_byd-alltests_210924060554.json"
    #
    # filename1=r"D:\qdaq\对比\danluxuanbianxin\PM61191-2vibs-resolver-72-144-140_PM61191-2vibs_210924034249.json"
    # filename2=r"D:\qdaq\对比\danluxuanbianxin\PM61191-2vibs-resolver-8-16-15_PM61191-2vibs_210924040047.json"
    # filename3=r"D:\qdaq\对比\danluxuanbianxin\PM61191-2vibs_PM61191-2vibs_210924035336.json"
    #
    # filename1=r"D:\qdaq\对比\byd_selectspeed\jinkang-2vibs_jinkang-2vibs_210913034750.json"
    # filename3=r"D:\qdaq\对比\byd_selectspeed\jinkang-2vibs_jinkang-2vibs-filter2.56_210924085643.json"
    # filename1=r"D:\qdaq\temp\105K2_105NG1-1_210826093800_210929074232.json"
    # filename3=r"D:\qdaq\temp\105K2_105NG1-1_210826093800_210929074439.json"
    with open(filename1,encoding="utf-8") as f:
        data1 = json.load(f)
    with open(filename2,encoding="utf-8") as f:
        data2 = json.load(f)
    with open(filename3,encoding="utf-8") as f:
        data3 = json.load(f)
    # with open(filename1,encoding="utf-8") as f:
    #     data2 = json.load(f)
    sensor_list=range(1)
    test_list=range(3)
    # test_list=[0,3]

    # for sensor_id in sensor_list:
    #     for test_id in test_list:
    #         rms1=data1["resultData"][sensor_id]["dataSection"][test_id]["twodTD"][0]
    #         rms3 = data3["resultData"][sensor_id]["dataSection"][test_id]["twodTD"][0]
    #         plt.figure("sensor:{} test:{}".format(sensor_id,test_id))
    #         plt.plot(rms1["xValue"],np.array(rms1["yValue"]),c="r",label="const")
    #         plt.plot(rms3["xValue"],np.array(rms3["yValue"]),c="g",label="v3")

    plt.show()
    for sensor_id in sensor_list:
        for test_id in test_list:
            orderSpectrum1=data1["resultData"][sensor_id]["dataSection"][test_id]["twodOS"][0]
            # orderSpectrum2=data2["resultData"][sensor_id]["dataSection"][test_id]["twodOS"][0]
            orderSpectrum3=data3["resultData"][sensor_id]["dataSection"][test_id]["twodOS"][0]
            plt.figure("sensor:{} test:{}".format(sensor_id,test_id))
            plt.title("orderSpectrum")
            plt.plot(orderSpectrum1["xValue"],np.array(orderSpectrum1["yValue"]),c="r",label="resolver")
            # plt.plot(orderSpectrum2["xValue"],np.array(orderSpectrum2["yValue"]),c="b",label="resolver816")
            plt.plot(orderSpectrum3["xValue"],np.array(orderSpectrum3["yValue"]),c="g",label="resolver2")

            plt.legend()
            # plt.figure("sensor:{} test:{} delta-orderSpectrum".format(sensor_id,test_id))
            # plt.plot(orderSpectrum1["xValue"],np.array(orderSpectrum1["yValue"])-np.array(orderSpectrum2["yValue"]),label="1-2")

            # orderSlice_40_1=data1["resultData"][sensor_id]["dataSection"][test_id]["twodOC"][7]
            # # orderSlice_40_2=data2["resultData"][sensor_id]["dataSection"][test_id]["twodOC"][7]
            # orderSlice_40_3=data3["resultData"][sensor_id]["dataSection"][test_id]["twodOC"][7]
            # plt.figure("sensor:{} test:{} 40 order slice ".format(sensor_id, test_id))
            # plt.title("sensor:{} test:{} 40 order slice ".format(sensor_id, test_id))
            # plt.plot(orderSlice_40_1["xValue"],orderSlice_40_1["yValue"],c="r",label="filename1")
            # # plt.plot(orderSlice_40_2["xValue"],orderSlice_40_2["yValue"],c="b",label="filename2")
            # plt.plot(orderSlice_40_3["xValue"],orderSlice_40_3["yValue"],c="g",label="filename3")
            # plt.legend()
            #
            # orderSlice_48_1=data1["resultData"][sensor_id]["dataSection"][test_id]["twodOC"][8]
            # # orderSlice_48_2=data2["resultData"][sensor_id]["dataSection"][test_id]["twodOC"][8]
            # orderSlice_48_3=data3["resultData"][sensor_id]["dataSection"][test_id]["twodOC"][8]
            # plt.figure("sensor:{} test:{} 48 order slice ".format(sensor_id, test_id))
            # plt.title("sensor:{} test:{} 48 order slice ".format(sensor_id, test_id))
            # plt.plot(orderSlice_48_1["xValue"],orderSlice_48_1["yValue"],c="r",label="filename1")
            # # plt.plot(orderSlice_48_2["xValue"],orderSlice_48_2["yValue"],c="b",label="filename2")
            # plt.plot(orderSlice_48_3["xValue"],orderSlice_48_3["yValue"],c="g",label="filename3")
            # plt.legend()


    plt.legend()
    plt.show()