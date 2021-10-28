#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2021/7/29 13:38
@Author : Wall@Synovate
@Email  : qxu@sonustc.com
@Version: 1.0.0
"""
import schedule
from datetime import datetime
from pythonping import ping


def job():
    now = datetime.now()
    time = now.strftime("%H:%M:%S")
    print(f'现在的时间是: {time}')


host = 'www.cisco.com'


def ping_test():
    ping_result = ping(host)
    now = datetime.now()
    time = now.strftime("%H:%M:%S")
    if 'Reply' in str(ping_result):
        print(host + '可达。' + f'现在的时间是: {time}')
    else:
        print(host + '不可达。' + f'现在的时间是: {time}')


if __name__ == '__main__':
    schedule.every(3).seconds.do(job)
    schedule.every(5).seconds.do(ping_test)
    while True:
        schedule.run_pending()
