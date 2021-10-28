#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2021/7/27 18:36
@Author : Wall@Synovate
@Email  : qxu@sonustc.com
@Version: 1.0.0
说明：获取当前运行电脑的硬件信息
"""

import socket
from psutil import net_if_addrs
import platform

if platform.platform().lower().startswith('windows'):
    from wmi import WMI


# 以下为windows下的方法
def get_computer_name_ip():
    """
    功能：获取 电脑名、ip地址
    输入：无
    输出：
    1. 本机电脑名
    2. ip地址
    """
    # 获取本机电脑名
    name = socket.getfqdn(socket.gethostname())
    # 获取本机ip
    addr = socket.gethostbyname(name)
    return name, addr


def get_mac_address():
    """
    功能：获取mac地址和对应的ip地址
    输入：无
    返回：
    1. mac地址列表
    2. IP地址列表
    """
    mac_list = []
    ip_list = []
    for adapter_name, adapter_value in net_if_addrs().items():
        for item in adapter_value:
            address = item[1]
            if '-' in address and len(address) == 17:
                # 记录mac地址信息（需要满足标准格式）
                mac_list.append(address)
            elif len(address.split('.')) == 4:
                ip_list.append(address)
    return mac_list, ip_list


def get_cpu_id():
    """
    功能：获取CPU序列号，唯一且无法修改
    输入：无
    返回：cpu序列号列表
    """
    c = WMI()
    id_list = []
    for cpu in c.Win32_Processor():
        id_list.append(cpu.ProcessorId.strip())
    return id_list


def get_disk_id():
    """
    功能：硬盘序列号，唯一且无法修改
    输入：无
    返回：硬盘序列号列表
    """
    c = WMI()
    id_list = []
    for physical_disk in c.Win32_DiskDrive():
        id_list.append(physical_disk.SerialNumber)
    return id_list


def get_board_id():
    """
    功能：主板序列号，唯一且无法修改，可能会不一样
    输入：无
    返回：主板序列号列表
    """
    c = WMI()
    id_list = []
    for board in c.Win32_BaseBoard():
        id_list.append(board.SerialNumber)
    return id_list


def get_bios_id():
    """
    功能：BIOS序列号，唯一且无法修改（组装机可能会获取不到该信息，应避免使用）
    输入：无
    返回：BIOS序列号列表
    """
    c = WMI()
    id_list = []
    for bios in c.Win32_BIOS():
        id_list.append(bios.SerialNumber)
    return id_list


# TODO: linux下的硬件信息获取方法
def get_cpu_id_linux():
    pass


if __name__ == '__main__':
    mac_addr, ip_addr = get_mac_address()
    print("MAC地址: ", mac_addr)
    print("ip地址：", ip_addr)
    pc_name, _ = get_computer_name_ip()
    print("电脑名：", pc_name)
    cpu_id = get_cpu_id()
    print("CPU 序列号：", cpu_id)
    disk_id = get_disk_id()
    print("Disk 序列号：", disk_id)
    board_id = get_board_id()
    print("主板序列号：", board_id)
    bios_id = get_bios_id()
    print("BIOS 序列号：", bios_id)
