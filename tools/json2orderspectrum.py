import json
import os
import matplotlib.pyplot as plt
from nptdms import ChannelObject, TdmsWriter


def write_tdms(filename, groupname, channelname, data, properties=None, mode='a'):
    """
    function: write the data into TDMS file
    :param
        filename(string): the full path of target TDMS file
        groupname(string): the group name for TDMS write
        channelname(string): the channel name for TDMS write
        data(list): data need to write into TDMS
        properties(dict): the properties of channel data, default as {}
        mode(char): 'w' or 'a', 'w' means it will remove all the existed data and write new data,
        'a' means it just append new data, hold the existed data, default as 'a'
    :return:
        existed TDMS file
    """
    channel_object = ChannelObject(groupname, channelname, data, properties)
    with TdmsWriter(filename, mode) as tdms_writer:
        tdms_writer.write_segment([channel_object])
    del tdms_writer
    
if __name__ == '__main__':
    path = "D:/ShirohaUmi/work_document/mcc/aqsrt/ordercolormap"
    files = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x)) and os.path.splitext(x)[1] == ".json"]

    length_for_save=int(300/0.03125)
    twodOS=dict()
    flag=True
    for file in files:
        with open(os.path.join(path,file)) as f:
            data=json.load(f)
            # print(file)
            # 传感器索引  测试段索引  指标索引
            twodOS[file]=data['resultData'][0]['dataSection'][0]['twodOS'][0]
            if flag:
                write_tdms(os.path.join(path, "twodOS.tdms"), "aqsrt", "order", twodOS[file]["xValue"][:length_for_save])
                flag=False
            write_tdms(os.path.join(path, "twodOS.tdms"), "aqsrt", file.split('_')[1], twodOS[file]["yValue"][:length_for_save])
            plt.plot(twodOS[file]['xValue'][:length_for_save],twodOS[file]['yValue'][:length_for_save],label=file)

    # write_tdms(os.path.join(path, "twodOS.tdms"), "aqsrt", "",twodOS)
    plt.legend()
    plt.show()
    print("over")
