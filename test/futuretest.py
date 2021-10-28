from nptdms import TdmsFile
import numpy as np
import time
import logging
from speed_tools import rpm_calc, speed_detect_start, speed_detect_end, resolver, resolver_single_signal
import os
from parameters import Parameters, config_folder, config_file, limit_folder, limit_file, log_folder
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import traceback
#
# global_arr_shared = None
# def init_pool(arr_shared):
#     global global_arr_shared
#     global_arr_shared = arr_shared
#
# def task(cutted_angle):
#     print("task pid:{},ppid:{}".format(os.getpid(), os.getppid()))
#     # print(cutted_angle)
#
#
# if __name__ == '__main__':
#     print("main pid:{},ppid:{}".format(os.getpid(), os.getppid()))
#     cutted_angle=np.array([1,2])
#     cutted_loc=np.array([3,4])
#     shared_cutted_angle=multiprocessing.RawArray('d',cutted_angle)
#     shared_cutted_loc=multiprocessing.RawArray('d',cutted_loc)
#
#     # multiprocessing.Manager().Array()
#
#     executor=ProcessPoolExecutor(max_workers=4,initializer=init_pool,initargs=(shared_cutted_angle,))
#     future1=executor.submit(task,shared_cutted_angle)
#     print(future1.exception())
#     future2=executor.submit(task,shared_cutted_angle)
#     print(future2.exception())
#     future3=executor.submit(task,shared_cutted_angle)
#     print(future3.exception())
#     future4=executor.submit(task,shared_cutted_angle)
#     print(future4.exception())

import multiprocessing
import time
import numpy as np
SHAPE = (2, 3)
global_arr_shared = None
def init_pool(arr_shared):
    global global_arr_shared
    global_arr_shared = arr_shared
def worker(i):
    arr = np.frombuffer(global_arr_shared, np.double).reshape(SHAPE)
    time.sleep(1)  # some other operations
    return np.sum(arr * i)
if __name__ == '__main__':
    arr = np.array([[1,2,3],[4,5,6]])
    arr_shared = multiprocessing.RawArray('d', arr.ravel())
    # with multiprocessing.Pool(processes=2, initializer=init_pool, initargs=(arr_shared,)) as pool:  # initargs传入tuple
    #     for result in pool.map(worker, [1,2,3]):
    #         print(result)
    executor=ProcessPoolExecutor(max_workers=4,initializer=init_pool,initargs=(arr_shared,))

    arr = np.random.randn(*SHAPE)
    global_arr_shared = multiprocessing.RawArray('d', arr.ravel())

    future1=executor.submit(worker,1)
    print(future1.exception())
    print(future1.result())
    future2=executor.submit(worker,2)
    print(future2.exception())
    print(future2.result())
    future3=executor.submit(worker,3)
    print(future3.exception())
    print(future3.result())