import os

from nptdms import TdmsFile
raw_data_filename="D:/qdaq/rawdata/xiaodianji/21082609/105kOK3-3-192000_1.tdms"

file="test01_3mm_8k_210802064140.tdms"
data = TdmsFile.read(raw_data_filename)