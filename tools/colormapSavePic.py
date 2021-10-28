# -*- coding: utf-8 -*-
"""
Created on Mon Jar 18 18:45:03 2021

@author: Wall@Synovate

function: draw out the time frequency colormap and save it as picture, X as Frequency or Order and Y as Speed or Time

usage: python colormapSave_v1.py -i D:\Work\script_for_TX\colormap -f Colormap.json -o test.png
"""

import zlib
import json
import os
import sys
import getopt
import logging
import traceback
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


logging.basicConfig(
        level=logging.DEBUG,  # 日志级别，只有日志级别大于等于设置级别的日志才会输出
        format='%(asctime)s %(filename)s %(funcName)s[line:%(lineno)d] %(levelname)s %(message)s',  # 日志输出格式
        datefmt='[%Y-%m-%d %H:%M:%S]',  # 日期表示格式
        filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'colormap_savePic.log'),  # 输出定向的日志文件路径
        filemode='a'  # 日志写模式，是否尾部添加还是覆盖
    )

    
# set initial values
config = {
     "input_filepath": "",
     "input_filename": "",
     "output_filename": "",
}
returnInfo = dict()

# Step1: get user's input (input file path, file name)
try:
    opts, args = getopt.getopt(sys.argv[1:], '-i:-f:-o:-h-v',
                               ['input_filepath=', 'input_filename=', 'output_filename=', 'help', 'version'])
    for option, value in opts:
        if option in ["-h", "--help"]:
            print("usage:%s -i input_filepath -f input_filename -o output_filename")
            sys.exit()
        if option in ["-v", "--version"]:
            print("version: v1.0.0")
            sys.exit()
        elif option in ['-i']:
            config["input_filepath"] = value
        elif option in ['-f']:
            config['input_filename'] = value
        elif option in ['-o']:
            config['output_filename'] = value
except Exception:
    returnInfo["code"] = 300
    returnInfo["msg"] = 'Input has error, please check! or you can use -h for help'
    print(returnInfo)
    logging.info('Input has error, please check! or you can use -h for help')
    sys.exit()


if __name__ == "__main__":
    try:

        #this part is for internal test
        # config = {
        #     "input_filepath": r"D:\Wall_Work\3_Project\301_SigMA\Python_script\test_data\HDF5_Version\aiway\testData-1",
        #     "input_filename": "sensor02-constant3_colormap.json",
        #     "output_filename": "sensor02-constant3_colormap.png"
        #         }

        try:
            # Step 2: merge the file path and file name, and generate the file name and file path for saving picture
            source_filepath = os.path.join(config["input_filepath"], config["input_filename"])
            save_filepath = os.path.join(config["input_filepath"], config["output_filename"])

            # Step3: read the compressed colormap data
            with open(source_filepath, 'r') as f:
                data = json.load(f)
        except Exception:
            returnInfo["code"] = 400
            returnInfo["msg"] = 'data reading error!'
            traceback.print_exc()
            logging.warning("data reading failed, failed msg:" + traceback.format_exc())
            sys.exit()
        else:
            try:
                # Step4: plot out the colormap and save it as .png file
                matplotlib.use('Agg')
                plt.figure()
                cm = plt.pcolormesh(data['xValue'], data['yValue'], data['zValue'], cmap='jet')
                plt.colorbar(cm)
                plt.title(data['xName'] + '-' + data['yName'] + ' Colormap')
                if data['xUnit']:
                    plt.xlabel(data['xName'] + '/' + data['xUnit'])
                else:
                    plt.xlabel(data['xName'])
                plt.xlabel(data['xName'])
                plt.ylabel(data['yName'] + '/' + data['yUnit'])
                plt.savefig(save_filepath)
                returnInfo["code"] = 200
                returnInfo["msg"] = 'colormap picture saved!'
            except Exception:
                returnInfo["code"] = 400
                returnInfo["msg"] = 'colormap picture save error!'
                traceback.print_exc()
                logging.warning("colormap picture save failed, failed msg:" + traceback.format_exc())
                sys.exit()
    except Exception:
        print('other error')
        traceback.print_exc()
        logging.warning("colormap save picture script errors, failed msg:" + traceback.format_exc())
        sys.exit()
    finally:
        print(returnInfo)

