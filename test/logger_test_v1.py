#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2021/7/28 11:35
@Author : Wall@Synovate
@Email  : qxu@sonustc.com
@Version: 1.0.0
"""

import os
import logging
import re
from logging.handlers import TimedRotatingFileHandler
import time


def setup_log(log_name):
    # 创建logger对象。传入logger名字
    logger = logging.getLogger(log_name)
    log_path = os.path.join("D:/qdaq/test/210728-1", log_name)
    # 设置日志记录等级
    logger.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s %(filename)s %(funcName)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='[%Y-%m-%d %H:%M:%S]')
    console_formatter = logging.Formatter(
        "[%(asctime)s] [%(process)d] [%(levelname)s] - %(module)s.%(funcName)s (%(filename)s:%(lineno)d) - %(message)s"
    )
    # interval 滚动周期，
    # when="MIDNIGHT", interval=1 表示每天0点为更新点，每天生成一个文件
    # backupCount  表示日志保存个数
    file_handler = TimedRotatingFileHandler(
        filename=log_path, when="M", interval=1, backupCount=5
    )
    # filename="mylog" suffix设置，会生成文件名为mylog.2020-02-25.log
    file_handler.suffix = "%Y-%m-%d_%H-%M.log"
    # extMatch是编译好正则表达式，用于匹配日志文件名后缀
    # 需要注意的是suffix和extMatch一定要匹配的上，如果不匹配，过期日志不会被删除。
    file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}.log$")
    # 定义日志输出格式
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    # 日志文件操作加入到logger对象中
    logger.addHandler(file_handler)

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger


if __name__ == "__main__":
    logger = setup_log("mylog")
    logger.debug("this is debug message")
    logger.info("this is info message")
    logger.warning("this is a warning message")
    t1 = time.time()
    while time.time() - t1 < 1000:
        try:
            int("xjk")
        except ValueError as e:
            logger.error(e)
        time.sleep(1)
        logger.debug("debug msg")
        logger.info('info msg')
    # 如果其他py文件想使用此配置日志，只需 logging.getLogger(日志的名字)  即可
