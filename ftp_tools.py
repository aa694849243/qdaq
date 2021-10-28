#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2021/9/17 16:51
@Author : Wall@Synovate
@Email  : qxu@sonustc.com
@Version: 1.0.0
"""

from ftplib import FTP
import os
import socket


def ftp_connect(host, username='ftpuser1', password='ftpuser_shengteng', port=21):
    """
    功能：建立FTP连接
    输入：
    1. 目标服务器IP
    2. ftp服务的用户名，默认是'ftpuser1'
    3. ftp服务的密码， 默认是'ftpuser_shengteng'
    4. ftp服务的端口，默认是21
    返回：
    1. ftp实例
    """
    ftp = FTP()
    # 打开调试级别2，显示详细信息，1会产生适度的调试输出，通常每个请求只有一行，0表示无输出，默认是0
    # ftp.set_debuglevel(2)
    # 0主动模式 1被动模式
    # ftp.set_pasv(0)
    # 连接指定ip的ftp服务
    ftp.connect(host, port)
    # 登录，如果匿名登录则用空串代替即可
    ftp.login(username, password)
    return ftp


def file_upload(ftp, local_filepath, remote_filepath, buf_size=4096):
    """
    功能：ftp上传数据至目标路径并命名为目标名字
    输入：
    1. ftp实例
    2. 本地文件名字（全路径）
    3. 目标文件名字（全路径）
    返回：无
    """
    with open(local_filepath, 'rb') as fp:
        # 上传文件
        ftp.storbinary('STOR ' + remote_filepath, fp, buf_size)
        ftp.set_debuglevel(0)


def file_download(ftp, local_filepath, remote_filepath, buf_size=4096):
    """
    功能：ftp下载数据至目标路径并命名为目标名字
    输入：
    1. ftp实例
    2. 本地文件名字（全路径）
    3. 目标文件名字（全路径）
    返回：无
    """
    with open(local_filepath, 'wb') as fp:
        ftp.retrbinary('RETR ' + remote_filepath, fp, buf_size)  # 下载文件
        ftp.set_debuglevel(0)


def file_check(ftp, remote_filepath):
    """
    功能：确认目标路径下是否已存在该文件
    """
    file_list = ftp.nlst(os.path.split(remote_filepath)[0])
    if remote_filepath in file_list:
        return True
    else:
        return False


def folder_check(ftp, base_folder, system_no, type_no, serial_no, timestamp):
    """
    功能：确定远程路径
    输入：
    1.
    """
    file_list_level1 = ftp.nlst(base_folder)
    folder_level1 = base_folder + '/' + system_no
    if folder_level1 not in file_list_level1:
        # 不存在该系统的目录则创建
        ftp.mkd(folder_level1)
    file_list_level2 = ftp.nlst(folder_level1)
    folder_level2 = folder_level1 + '/' + type_no
    if folder_level2 not in file_list_level2:
        # 不存在该产品类型的目录则需要创建
        ftp.mkd(folder_level2)
    file_list_level3 = ftp.nlst(folder_level2)
    folder_level3 = folder_level2 + '/' + serial_no
    if folder_level3 not in file_list_level3:
        # 不存在该序列号的目录则需要创建
        ftp.mkd(folder_level3)
    file_list_level4 = ftp.nlst(folder_level3)
    folder_level4 = folder_level3 + '/' + timestamp
    if folder_level4 not in file_list_level4:
        # 不存在当前时间的目录则需要创建
        ftp.mkd(folder_level4)
    return folder_level4


if __name__ == '__main__':
    import time
    s_time = time.time()
    server_ip = '192.168.2.109'
    target_ftp = ftp_connect(server_ip)
    print(target_ftp.getwelcome())
    src_filename = r'D:\qdaq\test\bak\red-ant-002_200805090618.tdms'
    # target_ftp.mkd('/home/ftpuser1/wall1')
    target_folder = folder_check(target_ftp, '/home/data/ftp/Data', 'systemWall', 'redant-ttl-fan_0', 'red-ant-002', '20210927170646')

    target_filename = target_folder + '/' + 'red-ant-002_200805090618.tdms'
    if file_check(target_ftp, target_filename):
        print("target file already existed")
        target_ftp.delete(target_filename)
    file_upload(target_ftp, src_filename, target_filename)
    target_ftp.quit()
    print("Done")
    print(time.time() - s_time)
