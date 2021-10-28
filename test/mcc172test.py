# -*- coding: utf-8 -*-
#
#
import sys
import numpy as np
# import parameters
import matplotlib.pyplot as plt
import time
from daqhats import hat_list, HatIDs, mcc172,SourceType,OptionFlags



# get hat list of MCC daqhat boards
board_list = hat_list(filter_by_id = HatIDs.ANY)
if not board_list:
    print("No boards found")
    sys.exit()


# Read and display every channel
for entry in board_list:
    if entry.id == HatIDs.MCC_172:
        # print(parameters)
        print("Board {}: MCC 172".format(entry.address))
        board = mcc172(entry.address)
        print(board.info())
        print(board.calibration_date())
        print(board.calibration_coefficient_read(0))

        board.iepe_config_write(0,1)
        board.iepe_config_write(1,1)
        print(board.iepe_config_read(0))
        print(board.iepe_config_read(1))
        # channel 0 读转速信号 读到的信号为mv，灵敏度设置为1000可以将单位变为V
        board.a_in_sensitivity_write(0,1)


        board.a_in_sensitivity_write(1,1)

        # 多个板的时候应该是需要一个设置为SourceType.MASTER,其它设置为SourceType.SLAVE,MASTER只能有一个
        # 其它需要注意的细节看该方法的描述
        board.a_in_clock_config_write(SourceType.MASTER,51200)
        print(board.a_in_clock_config_read())

        # 1启用一个channel ，2 启用另一个channel 3 启用两个channel
        board.a_in_scan_start(3,512000,OptionFlags.CONTINUOUS)

        # buffer_size 是两个通道buffer的和，
        # board.a_in_scan_start(3,10000,OptionFlags.CONTINUOUS) buffer_size为20000（注意给的10000是大于等于默认值的）
        # board.a_in_scan_start(1,10000,OptionFlags.CONTINUOUS) buffer_size为10000 (注意给的10000是大于等于默认值的）
        # 缓冲区中实际的大小应该是比这一个大小稍微大一点。
        print("buffer_size",board.a_in_scan_buffer_size())


        # print(board.a_in_scan_actual_rate(0,102400))

        # a_in_scan_read返回的samples_available是每一个通道剩余的数据，不是两个通道的和
        # 返回的数据，两个通道是如何排列的？？？？？

        # 超时时间（可以小于/等于/大于0）(单位是秒)内 在扫描中读不出来会阻塞等待，扫描结束能读多少是多少，看该方法的具体说明
        # 两个传感器会读出来双倍数据，交错排列
        data1=board.a_in_scan_read_numpy(2048, 100)
        # reshape不会改变原数组，只是返回重塑的数组
        # reshape返回的数组与原数组共享存储空间，只是逻辑结构不同
        vib=data1.data.reshape(-1,2)
        vibT=vib.T
        plt.subplot(211)
        plt.plot(vibT[0])
        plt.subplot(212)
        plt.plot(vibT[1])
        plt.show()
        # plt.show(block=False)



        print(data1.data.size)
        print(data1.running)
        print(data1)


        print(board.a_in_scan_read_numpy(4, 1000))

        # 单位是秒
        # time.sleep(4)

        print("count", board.a_in_scan_channel_count())

        print(board.a_in_scan_buffer_size())
        print(board.a_in_scan_status())

        board.a_in_scan_stop()
        print("stop")
        # stop 之后马上查看status 可能running=True
        print(board.a_in_scan_status())
        print(board.a_in_scan_read(10, 1000))
        print(board.a_in_scan_status())
        board.a_in_scan_cleanup()


        board.blink_led(10)

        # time.sleep(10)

        # board.
        # for channel in range(board.info().NUM_AI_CHANNELS):
        #     board.a_in_sensitivity_write(0,10)
        #     board.a_in_sensitivity_write(1,10)
        #     value = board.a_in_scan_read(channel,1000)
        #     print("Ch {0}: {1:.3f}".format(channel, value))

