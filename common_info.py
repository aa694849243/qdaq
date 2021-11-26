#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2021/7/29 9:47
@Author : Wall@Synovate
@Email  : qxu@sonustc.com
@Version: 1.0.0
该脚本定义了公共信息
"""

import os
import sys
import re
import time
import datetime
import configparser
import json
import logging
import platform
import traceback
from logging.handlers import TimedRotatingFileHandler
from Cryptodome.Cipher import AES
from binascii import b2a_base64, a2b_base64


class Aescrypt:
    # AES加密解密类
    def __init__(self, key):
        self.AES_LENGTH = 32  # 只能是16或32
        self.pad_str = ' '  # 最好是空格，防止冲突
        self.mode = AES.MODE_ECB  # 加密模式，ECB为电码本模式
        self.cryptor = AES.new(self.pad_key(key).encode(), self.mode)

    # 加密函数，如果text不是16或32的倍数【加密文本text必须为16的倍数！】，那就补足为16或32的倍数
    # 加密内容需要长达16位字符，所以进行指定字符(默认是空格）拼接
    def pad(self, text):
        while len(text) % self.AES_LENGTH != 0:
            text += self.pad_str
        return text

    # 加密密钥需要长达16或者32位字符，不足用指定字符(默认是空格）拼接，更长则去除前面多出来的部分
    def pad_key(self, key):
        if len(key) < self.AES_LENGTH:
            key += self.pad_str * (self.AES_LENGTH - len(key))
        else:
            key = key[-self.AES_LENGTH:]
        return key

    def encrypt(self, text):
        # 这里密钥key 长度必须为16（AES-128）、24（AES-192）、或32（AES-256）Bytes 长度.目前AES-128足够用
        # 加密的字符需要转换为bytes
        # print(self.pad(text))
        self.ciphertext = self.cryptor.encrypt(self.pad(text).encode())
        # 因为AES加密时候得到的字符串不一定是ascii字符集的，输出到终端或者保存时候可能存在问题
        # 所以这里统一把加密后的字符串转化为16进制字符串
        return b2a_base64(self.ciphertext)

        # 解密后，去掉补足的空格用strip() 去掉

    def decrypt(self, text):
        plain_text = self.cryptor.decrypt(a2b_base64(text)).decode()
        return plain_text.rstrip(self.pad_str)


try:
    import codecs
except ImportError:
    codecs = None


class MultiprocessHandler(logging.FileHandler):
    """支持多进程的TimedRotatingFileHandler"""

    def __init__(self, filepath, filename, when='D', backupCount=30, encoding=None, delay=False):
        """filename 日志文件名,when 时间间隔的单位,backupCount 保留文件个数
        delay 是否开启 OutSteam缓存
            True 表示开启缓存，OutStream输出到缓存，待缓存区满后，刷新缓存区，并输出缓存数据到文件。
            False表示不缓存，OutStrea直接输出到文件"""
        self.prefix = filename
        self.backupCount = backupCount
        self.when = when.upper()

        # S 每秒建立一个新文件
        # M 每分钟建立一个新文件
        # H 每小时建立一个新文件
        # D 每天建立一个新文件

        # 正则匹配
        self.extMath_dict = {
            'S': r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}",
            'M': r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}",
            'H': r"^\d{4}-\d{2}-\d{2}_\d{2}",
            'D': r"^\d{4}-\d{2}-\d{2}"
        }

        self.when_dict = {
            'S': "%Y-%m-%d_%H-%M-%S",
            'M': "%Y-%m-%d_%H-%M",
            'H': "%Y-%m-%d_%H",
            'D': "%Y-%m-%d"
        }
        # self.interval = self.intervalUpate(interval)

        # 日志文件日期后缀
        self.suffix = self.when_dict.get(when)
        if not self.suffix:
            raise ValueError(u"指定的日期间隔单位无效: %s" % self.when)
        # 拼接文件路径 格式化字符串（日志文件的文件名格式）
        self.filefmt = os.path.join(filepath, "%s.%s.log" % (self.prefix, self.suffix))
        # 使用当前时间，格式化文件格式化字符串（初始化）
        self.start_time = datetime.datetime.now()
        self.filePath = self.start_time.strftime(self.filefmt)
        # 获得文件夹路径（去最后一个/之前的字符串）
        _dir = os.path.dirname(self.filefmt)
        try:
            # 如果日志文件夹不存在，则创建文件夹
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        except Exception:
            print(u"创建文件夹失败")
            print(u"文件夹路径：" + self.filePath)
            pass

        if codecs is None:
            encoding = None

        logging.FileHandler.__init__(self, self.filePath, 'a+', encoding, delay)

    def intervalUpate(self, raw_interval):
        # 间隔统一切换成s
        if self.when == 'S':
            interval = raw_interval
        elif self.when == 'M':
            interval = raw_interval * 60
        elif self.when == 'H':
            interval = raw_interval * 60 * 60
        elif self.when == 'D':
            interval = raw_interval * 60 * 60 * 24
        else:
            raise ValueError(u"指定的日期间隔单位无效: %s" % self.when)
        return interval

    def shouldChangeFileToWrite(self):
        """更改日志写入目的写入文件
        :return True 表示已更改，False 表示未更改"""
        # 以当前时间获得新日志文件路径
        _time_now = datetime.datetime.now()
        _filePath = _time_now.strftime(self.filefmt)
        # 新日志文件日期 不等于 旧日志文件日期，则表示已经到了日志切分的时候
        # 更换日志写入目的为新日志文件。
        # 例如 按 天 （D）来切分日志
        # 当前新日志日期等于旧日志日期，则表示在同一天内，还不到日志切分的时候
        # 当前新日志日期不等于旧日志日期，则表示不在
        # 同一天内，进行日志切分，将日志内容写入新日志内。
        if _filePath != self.filePath:
            self.filePath = _filePath
            return True
        return False

    def doChangeFile(self):
        """输出信息到日志文件，并删除多于保留个数的所有日志文件"""
        # 日志文件的绝对路径
        self.baseFilename = os.path.abspath(self.filePath)
        # stream == OutStream
        # stream is not None 表示 OutStream中还有未输出完的缓存数据
        if self.stream:
            # flush close 都会刷新缓冲区，flush不会关闭stream，close则关闭stream
            # self.stream.flush()
            self.stream.close()
            # 关闭stream后必须重新设置stream为None，否则会造成对已关闭文件进行IO操作。
            self.stream = None
        # delay 为False 表示 不OutStream不缓存数据 直接输出
        #   所有，只需要关闭OutStream即可
        if not self.delay:
            # 这个地方如果关闭colse那么就会造成进程往已关闭的文件中写数据，从而造成IO错误
            # delay == False 表示的就是 不缓存直接写入磁盘
            # 我们需要重新在打开一次stream
            # self.stream.close()
            self.stream = self._open()
        # 删除多于保留个数的所有日志文件
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                print(s)
                try:
                    os.remove(s)
                except Exception:
                    print("删除文件错误")

    def getFilesToDelete(self):
        """获得过期需要删除的日志文件"""
        # 分离出日志文件夹绝对路径
        # split返回一个元组（absFilePath,fileName)
        # 例如：split('I:\ScripPython\char4\mybook\util\logs\mylog.2017-03-19）
        # 返回（I:\ScripPython\char4\mybook\util\logs， mylog.2017-03-19）
        # _ 表示占位符，没什么实际意义，
        dirName, _ = os.path.split(self.baseFilename)
        fileNames = os.listdir(dirName)
        result = []
        # self.prefix 为日志文件名 列如：mylog.2017-03-19 中的 mylog
        # 加上 点号 . 方便获取点号后面的日期
        prefix = self.prefix + '.'
        plen = len(prefix)
        for fileName in fileNames:
            if fileName[:plen] == prefix:
                # 日期后缀 mylog.2017-03-19 中的 2017-03-19
                suffix = fileName[plen:]
                # 匹配符合规则的日志文件，添加到result列表中
                if re.compile(self.extMath_dict.get(self.when)).match(suffix):
                    result.append(os.path.join(dirName, fileName))
        # 对文件列表进行排序
        result.sort()

        # 返回待删除的日志文件
        # 多于保留文件个数 backupCount的所有前面的日志文件。
        if len(result) < self.backupCount:
            result = []
        else:
            result = result[:len(result) - self.backupCount]
        return result

    def emit(self, record):
        """发送一个日志记录
        覆盖FileHandler中的emit方法，logging会自动调用此方法"""
        try:
            if self.shouldChangeFileToWrite():
                self.doChangeFile()
            logging.FileHandler.emit(self, record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def setup_logger(log_path, log_name, log_type, log_backup):
    """
    功能：定义日志记录规则（包括格式，记录间隔，过期删除等）
    输入：
    1. 日志保存文件夹
    2. 日志名称
    返回：日志对象
    """
    logger = logging.getLogger(log_name)
    # 设置日志记录等级，小于设定级别的信息不会被记录到日志中（比如设定级别是INFO，则DEBUG和NOTSET级别的信息不会被记录）
    logger.setLevel(logging.DEBUG)  # 这里因为控制台输出需要debug等级（低于文件输出），故而这里设置为debug
    # 设置定期记录和删除日志
    # when="MIDNIGHT"，日志名变更时间单位
    # interval=1 间隔时间，是指等待N个when单位的时间后，自动重建文件
    # backupCount=30  保留日志最大文件数，超过限制，删除最先创建的文件；默认值0，表示不限制
    # when="MIDNIGHT", interval=1 表示每天0点为更新点，每天生成一个文件，只保留最近一个月的日志文件
    file_handler = MultiprocessHandler(log_path, log_name, when=log_type, backupCount=log_backup)
    # 定义日志输出格式
    formatter = logging.Formatter(
        '%(asctime)s %(filename)s %(funcName)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='[%Y-%m-%d %H:%M:%S]')
    # 添加日志文件文件输出控制器
    file_handler.setFormatter(formatter)
    # 日志文件不记录debug信息
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 添加控制台输出控制器
    console_handler = logging.StreamHandler()
    # 控制台输出debug信息
    console_handler.setLevel(logging.DEBUG)
    # 共用了一个格式
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


class MyLoggingHandler(TimedRotatingFileHandler):

    def __init__(self, filename, when='h', interval=1, backupCount=0, encoding=None, delay=False,
                 utc=False, atTime=None):
        TimedRotatingFileHandler.__init__(self, filename, when=when, interval=interval,
                                          backupCount=backupCount, encoding=encoding, delay=delay,
                                          utc=utc, atTime=atTime)

    def computeRollover(self, currentTime):
        # 将时间取整
        t_str = time.strftime(self.suffix, time.localtime(currentTime))
        t = time.mktime(time.strptime(t_str, self.suffix))
        return TimedRotatingFileHandler.computeRollover(self, t)

    def doRollover(self):
        """
        do a rollover; in this case, a date/time stamp is appended to the filename
        when the rollover happens. However, you want the file to be named for the
        start of the interval, not the current time. If there is a backup count,
        then we have to get a list of matching filenames, sort them and remove
        the one with the oldest suffix.
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        # get the time that this sequence started at and make it a TimeTuple
        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]
        t = self.rolloverAt - self.interval
        if self.utc:
            timeTuple = time.gmtime(t)
        else:
            timeTuple = time.localtime(t)
            dstThen = timeTuple[-1]
            if dstNow != dstThen:
                if dstNow:
                    addend = 3600
                else:
                    addend = -3600
                timeTuple = time.localtime(t + addend)
        dfn = self.rotation_filename(self.baseFilename + "." +
                                     time.strftime(self.suffix, timeTuple))
        # 修改内容--开始
        # 在多进程下，若发现dfn已经存在，则表示已经有其他进程将日志文件按时间切割了，只需重新打开新的日志文件，写入当前日志；
        # 若dfn不存在，则将当前日志文件重命名，并打开新的日志文件
        if not os.path.exists(dfn):
            try:
                self.rotate(self.baseFilename, dfn)
            except FileNotFoundError:
                # 这里会出异常：未找到日志文件，原因是其他进程对该日志文件重命名了，忽略即可，当前日志不会丢失
                pass
        # 修改内容--结束
        # 原内容如下：
        """
        if os.path.exists(dfn):
          os.remove(dfn)
        self.rotate(self.baseFilename, dfn)
        """

        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)
        if not self.delay:
            self.stream = self._open()
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval
        # If DST changes and midnight or weekly rollover, adjust for this.
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dstAtRollover = time.localtime(newRolloverAt)[-1]
            if dstNow != dstAtRollover:
                if not dstNow:  # DST kicks in before next rollover, so we need to deduct an hour
                    addend = -3600
                else:  # DST bows out before next rollover, so we need to add an hour
                    addend = 3600
                newRolloverAt += addend
        self.rolloverAt = newRolloverAt


# 定义日志
def setup_logger_sigleProcess(log_path, log_name):
    """
    功能：定义日志记录规则（包括格式，记录间隔，过期删除等）
    输入：
    1. 日志保存文件夹
    2. 日志名称
    返回：日志对象
    """
    logger = logging.getLogger(log_name)
    log_filepath = os.path.join(log_path, log_name)
    # 设置日志记录等级，小于设定级别的信息不会被记录到日志中（比如设定级别是INFO，则DEBUG和NOTSET级别的信息不会被记录）
    logger.setLevel(logging.DEBUG)  # 这里因为控制台输出需要debug等级（低于文件输出），故而这里设置为debug
    # 设置定期记录和删除日志
    # when="MIDNIGHT"，日志名变更时间单位
    # interval=1 间隔时间，是指等待N个when单位的时间后，自动重建文件
    # backupCount=30  保留日志最大文件数，超过限制，删除最先创建的文件；默认值0，表示不限制
    # when="MIDNIGHT", interval=1 表示每天0点为更新点，每天生成一个文件，只保留最近一个月的日志文件
    file_handler = TimedRotatingFileHandler(filename=log_filepath, when="M", interval=1,
                                            backupCount=3)
    # 后缀suffix设置，如filename="log"，suffix = "%Y-%m-%d_%H-%M.log"会生成文件名为log.2020-02-25.log
    file_handler.suffix = "%Y-%m-%d_%H-%M.log"
    # extMatch是编译好正则表达式，用于匹配日志文件名后缀
    # 需要注意的是suffix和extMatch一定要匹配的上，如果不匹配，过期日志不会被删除。
    file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}.log$")
    # 定义日志输出格式
    formatter = logging.Formatter(
        '%(asctime)s %(filename)s %(funcName)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='[%Y-%m-%d %H:%M:%S]')
    # 添加日志文件文件输出控制器
    file_handler.setFormatter(formatter)
    # 日志文件不记录debug信息
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 添加控制台输出控制器
    console_handler = logging.StreamHandler()
    # 控制台输出debug信息
    console_handler.setLevel(logging.DEBUG)
    # 共用了一个格式
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


try:
    # 定义公用信息（原本放在parameters中）

    # 设置加密解密flag（1表示需要解密，0表示不需要解密）
    encrypt_flag = 0

    # 是否切割type信息（1表示切割， 0表示不切割）
    type_cut_flag = 1

    # basic folder and file info
    platform_info = platform.platform().lower()
    if platform_info.startswith('windows'):
        baseFolder = "D:/qdaq"
    else:
        baseFolder = "/home/qdaq"
    # set xName, xUnit, include twodTD（二维时间域）, twodOC（二维阶次切片）, threedOS（三维阶次谱），时频分析"

    # log folder
    log_folder = os.path.join(baseFolder, "Log/qDAQ")
    # confirm log folder existed
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # read in the parameters save info to read the target parameters file
    # Config文件夹用于储存SigMA下发的配置信息
    config_folder = os.path.join(baseFolder, "Config")
    config_file = "paramReceived.json"

    # read in the limits save info to read the target limits file
    # Limit文件夹用于储存SigMA下发的界限值信息
    limit_folder = os.path.join(baseFolder, "Limit")
    limit_file = "limitReceived.json"

    # config.ini path
    # 读取配置文件config.ini里的信息
    # 定义配置文件路径
    basic_config_path = r'D:\qdaq\config.ini'
    with open(basic_config_path, 'r', encoding='utf-8') as fp:
        basic_config = configparser.ConfigParser()
        basic_config.read_file(fp)

    # 日志切割类型和日志保存天数
    log_type = basic_config['Log Info']['save_type'].upper()
    backup_count = int(basic_config['Log Info']['backup_count'])
    # 创建日志的实例
    qDAQ_logger = setup_logger(log_folder, 'qDAQlog', log_type, backup_count)

    # 获取采集设备名称
    ni_device = basic_config['Basic Info']['ni_device_name']
    # 获取服务器ip信息
    server_ip = basic_config['Basic Info']['server_ip']
    # 台架ip
    test_bench_ip = basic_config['Basic Info']['test_bench_ip']
    # 获取转速信号类型（可为ttl，resolver，resolver2）
    speed_signal = basic_config['Basic Info']['speed_signal'].lower()
    # 获取配置文件里的传感器数量信息（用于开辟共享内存）
    sensor_count = int(basic_config['Basic Info']['sensor_count'])
    # # 获取软件运行模式
    # running_mode = basic_config['Basic Info']["mode"].lower()
    # 获取开辟共享内存最大时长
    max_size = int(basic_config['Basic Info']["max_size"])

    # 获取板卡类型，NI 还是umic
    board = basic_config['Basic Info']['board']
    # 软件版本
    version = int(basic_config['Basic Info']['version'])

    sys_name = basic_config['Basic Info']["sys_name"]

    # qdaq版这样写
    if version in [1, 2]:
        xName = "Speed"
        xUnit = "rpm"
    # 恒速电机版这样写
    elif version in [3, 4, 5]:
        xName = "Time"
        xUnit = "s"

    # Umic的信息
    Umic_names = None
    Umic_hostapis = None
    bit_depth = None

    if "Umic Info" in basic_config:
        Umic_names = json.loads(basic_config["Umic Info"]["Umic_names"])
        Umic_hostapis = list(map(int, json.loads(basic_config["Umic Info"]["Umic_hostapis"])))
        bit_depth = int(basic_config["Umic Info"]["bit_depth"])

    # 获取原始数据读取和保存的信息
    save_type = basic_config['Raw Data']["save_type"].lower()
    read_type = basic_config['Raw Data']["read_type"].lower()

    # 服务器版单机版切换（即是否进行ftp上传）
    ftp_flag = int(basic_config['Raw Data']['ftp'])
    # 确定原始数据存放位置（包括服务器版和单机版）
    if ftp_flag:
        # 服务器版的原始数据存放位置
        basic_folder = "/home/data/ftp/Data"
    else:
        # 单机版的原始数据存放位置
        if platform_info.startswith('windows'):
            basic_folder = "D:/ftpData/Data"
        else:
            basic_folder = "/home/data/ftp/Data"
    # 定义共享内存中进程控制的flag索引的保存位置
    flag_index_dict = dict()
    # 这三个值用于在接收到stop指令后，停止speed，nvh以及datapack进程
    flag_index_dict['speedCalclation'] = 0
    flag_index_dict['nvhCalclation'] = 1
    flag_index_dict['dataPack'] = 2
    # 这两个判断speed和datapack进程是否在未出错的情况下已经计算完成,1代表完成，默认为0
    flag_index_dict["speed_finish"] = 3
    flag_index_dict["datapack_finish"] = 5
    # 这三个判断speed,nvh和datapack进程是否在运行过程中出现异常
    flag_index_dict["speed_error"] = 6
    flag_index_dict["nvh_error"] = 7
    flag_index_dict["datapack_error"] = 8
    flag_index_dict['calibration'] = 9
    # 下一个flag索引从10开始

    # 建立加密解密方法
    Cryptor = Aescrypt('Synovate212')

    # 计算机硬件信息
    cpu_serial = 'BFEBFBFF000A0660'
    # 是否进行硬件校验
    hardware_confirm_flag = 0

    # simu模式每帧等待时间（设置的时比例sleep_ratio, sleep_time = sleep_ratio * sampsPerChan / sampleRate)
    sleep_ratio = 0.2  # 帧长的1/5较为合适
    sleep_ratio = 0  # 帧长的1/5较为合适

    # 工况评估指标是否放入一维指标
    ramp_quality_flag = 0

    # 恒速段指标初始化评判结果
    constant_initial_indicator_diagnostic = int(basic_config['Indicator Factors']['constant_initial'])
    drive_initial_indicator_diagnostic = int(basic_config['Indicator Factors']['drive_initial'])
    coast_initial_indicator_diagnostic = int(basic_config['Indicator Factors']['coast_initial'])

    # 高通截止频率（滤除低阶成分，Hz）
    cut_off_freq = float(basic_config['Indicator Factors']['cut_off_freq'])

    # 初始测试段索引
    preRecogIndex = 0
except Exception:
    print("config info error")
    traceback.format_exc(Exception)
    os.system("Pause")
    sys.exit()
