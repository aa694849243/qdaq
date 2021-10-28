#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2021/7/27 15:56
@Author : Wall@Synovate
@Email  : qxu@sonustc.com
@Version: 1.0.0
uuid package to get mac address, wmi to get cpu_id and board_id
"""


import wmi
import socket
from psutil import net_if_addrs


def get_mac_addr():
    # MAC地址
    mac_list = []
    for k, v in net_if_addrs().items():
        for item in v:
            address = item[1]
            if '-' in address and len(address) == 17:
                mac_list.append(address)
    return mac_list


def get_computer_name_ip():
    """
    python获取 电脑名、ip地址
    :return:
    """
    # 获取本机电脑名
    name = socket.getfqdn(socket.gethostname())
    # 获取本机ip
    addr = socket.gethostbyname(name)
    return name, addr


def get_cpu_id():
    # CPU序列号，唯一且无法修改
    c = wmi.WMI()
    id_list = []
    for cpu in c.Win32_Processor():
        id_list.append(cpu.ProcessorId.strip())
    return id_list


def get_disk_id():
    # 硬盘序列号，唯一且无法修改
    c = wmi.WMI()
    id_list = []
    for physical_disk in c.Win32_DiskDrive():
        id_list.append(physical_disk.SerialNumber)
    return id_list


def get_board_id():
    # 主板序列号，唯一且无法修改，可能会不一样
    c = wmi.WMI()
    id_list = []
    for board in c.Win32_BaseBoard():
        id_list.append(board.SerialNumber)
    return id_list


def get_bios_id():
    # BIOS序列号，唯一且无法修改
    c = wmi.WMI()
    id_list = []
    for bios in c.Win32_BIOS():
        id_list.append(bios.SerialNumber)
    return id_list


if __name__ == '__main__':
    mac_addr = get_mac_addr()
    print("MAC地址: ", mac_addr)
    pc_name, ip_addr = get_computer_name_ip()
    print("电脑名：", pc_name)
    print("ip地址：", ip_addr)
    cpu_id = get_cpu_id()
    print("CPU 序列号：", cpu_id)
    disk_id = get_disk_id()
    print("Disk 序列号：", disk_id)
    board_id = get_board_id()
    print("主板序列号：", board_id)
    bios_id = get_bios_id()
    print("BIOS 序列号：", bios_id)
