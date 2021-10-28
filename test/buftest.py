#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2021/6/30 9:57
@Author : Wall@Synovate
@Email  : qxu@sonustc.com
@Version: 1.0.0
"""
from multiprocessing import shared_memory,Lock as Process_Lock,Queue as Process_Queue
from multiprocessing import Process

def process(lock,index,Q):
    shm=shared_memory.SharedMemory(name="shm_buf")
    shm_buf=shm.buf
    count=0
    while True:
        Q.get()
        lock.acquire()
        shm_buf[5]-=1
        lock.release()
        print("count_p{}:{}".format(index,count))
        count+=1


if __name__ == '__main__':
    shm = shared_memory.SharedMemory(name="shm_buf", create=True, size=10)
    shm_buf=shm.buf
    shm_buf[5]=0
    count=0
    lock = Process_Lock()
    Q_list=list()
    Q_list.append(Process_Queue(1))
    Q_list.append(Process_Queue(1))
    process0=Process(target=process,args=(lock,0,Q_list[0]))
    process1=Process(target=process,args=(lock,1,Q_list[1]))
    process0.start()
    process1.start()
    while True:
        while shm_buf[5]!=0:
            pass
        lock.acquire()
        shm_buf[5]=int("2")
        print("count_p_main:{}".format(count))
        count+=1
        for Q in Q_list:
            Q.put(1)
        lock.release()