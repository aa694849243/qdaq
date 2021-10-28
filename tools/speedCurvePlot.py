import numpy as np
import h5py
import matplotlib.pyplot as plt
def read_hdf5(filename,groupname,channelnames,start=None,end=None):
    data=dict()

    h5file=h5py.File(filename,"r")
    for channel in channelnames:
        data[channel]=np.array(h5file[groupname][channel][start:end])
    del h5file
    return data


if __name__ == '__main__':
    # speed_Curve_resolver=read_hdf5("D:\\qdaq\\Data\hegnsurtnorsp_0\\2108\\21082308\\0823demo3_210823083510.h5","speedData",["speedLoc","speedValue"])
    # rawdata=read_hdf5("D:\\qdaq\\Data\hegnsurtnorsp_0\\2108\\21082308\\0823demo3_210823083510.h5","AIData",["Mic"])
    # speed_Curve_resolver=read_hdf5("D:\\qdaq\\对比\\v2v3\\PM61191-2vibs-resolver2-v2.h5","speedData",["speedLoc","speedValue"])
    speed_Curve_resolver=read_hdf5("D:\\qdaq\\Data\\resolver-ttl\\2109\\21091007\\PM61191-jinkang-3vibs_210910074501.h5","speedData",["speedLoc","speedValue"])
    speed_Curve_pinjie=read_hdf5("D:\\qdaq\\Data\\ttl-resolver\\2109\\21091401\\PM61191-jinkang-3vibs_210914015954.h5","speedData",["speedLoc","speedValue"])
    speed_Curve_single_resolver=read_hdf5(r"D:\qdaq\Data\PM61191\2109\21092403\PM61191-2vibs_210924034249.h5","speedData",["speedLoc","speedValue"])
    speed_Curve_jinkang=read_hdf5(r"D:\qdaq\Data\PM61191\2109\21092404\PM61191-2vibs_210924040047.h5","speedData",["speedLoc","speedValue"])

    # speed_Curve_single_resolver=read_hdf5("D:\\qdaq\\对比\\v2v3\\BYD-test-10603535-nofilter.h5","speedData",["speedLoc","speedValue"])
    # speed_Curve_single_resolver=read_hdf5("D:\\qdaq\\对比\\v2v3\\PM61191-2vibs-ppr72-144140-float.h5","speedData",["speedLoc","speedValue"])
    # speed_Curve_single_resolver=read_hdf5("D:\\qdaq\\对比\\v2v3\\PM61191-2vibs-ppr32-6462-float.h5","speedData",["speedLoc","speedValue"])
    # speed_Curve_single_resolver=read_hdf5("D:\\qdaq\\temp\\PM61191_PM61191-2vibs_210802072430.h5","speedData",["speedLoc","speedValue"])

    # plt.figure("rawdata")
    # plt.plot(rawdata["Mic"])
    plt.figure("speedCurve")
    plt.ylabel("rpm")
    plt.xlabel("s")
    plt.scatter(speed_Curve_single_resolver["speedLoc"],speed_Curve_single_resolver["speedValue"],s=0.3,marker="*",c='b',label="72")
    plt.scatter(speed_Curve_jinkang["speedLoc"],speed_Curve_jinkang["speedValue"],s=0.3,marker="^",c='r',label="8")
    # plt.plot(speed_Curve_pinjie["speedLoc"],speed_Curve_pinjie["speedValue"],c='r',label="pinjie")
    # plt.plot(speed_Curve_resolver["speedLoc"],speed_Curve_resolver["speedValue"],c='r',label="pinjie")

    plt.legend()
    plt.show()
    print("over")