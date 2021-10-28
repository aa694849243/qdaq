import os.path

from nptdms import TdmsFile, ChannelObject, TdmsWriter



def readAllRawData(filename, channelName):
    allrawdata = dict()

    with TdmsFile.open(filename) as tdms_file:
        for channelName in channelName:
            allrawdata[channelName] = list(tdms_file['AIData'][channelName][:])
            # allrawdata[channelName] = allrawdata[channelName][360000:1303000]
            # allrawdata.append(tdms_file[dtask.groupName][channelName].data)

    return allrawdata

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

    file_path="d:\\qdaq\\Simu"

    allrawdata1=readAllRawData(os.path.join(file_path,"PM61191-2vibs.tdms"),["Sin","Cos","Vib1","Vib2"])
    allrawdata2=readAllRawData(os.path.join(file_path,"jinkang-2vibs.tdms"),["Speed","Vib1"])

    
    filename="d:\\qdaq\\Simu\\PM61191-jinkang-3vibs.tdms"
    write_tdms(filename,"AIData","Sin",allrawdata1["Sin"])
    write_tdms(filename,"AIData","Cos",allrawdata1["Cos"])
    write_tdms(filename,"AIData","Speed",allrawdata2["Speed"][:8323072])
    write_tdms(filename,"AIData","Vib1",allrawdata1["Vib1"])
    write_tdms(filename,"AIData","Vib2",allrawdata1["Vib2"])
    write_tdms(filename,"AIData","Vib3",allrawdata2["Vib1"][:8323072])


    print(2)
