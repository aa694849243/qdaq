#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2021/7/20 19:23
@Author : Wall@Synovate
@Email  : qxu@sonustc.com
@Version: 1.0.0
"""

#  des模式  填充方式  ECB加密方式
from pyDes import des, PAD_PKCS5, ECB
from binascii import b2a_base64, a2b_base64
import json
import traceback
import time


class Descrypt:
    def __init__(self, def_secret_key):
        self.DES_LENGTH = 8  # 只能是8
        self.pad_str = ' '
        self.key = self.pad_key(def_secret_key)
        self.cryptor = des(self.key, ECB, self.key, padmode=PAD_PKCS5)

    # 加密密钥需要长达16或者32位字符，不足用指定字符(默认是空格）拼接，更长则去除前面多出来的部分
    def pad_key(self, key):
        if len(key) < self.DES_LENGTH:
            key += self.pad_str * (self.DES_LENGTH - len(key))
        else:
            key = key[-self.DES_LENGTH:]
        return key

    def encrypt(self, text):
        try:
            encrypt_text = self.cryptor.encrypt(text)
        except ValueError:
            encrypt_text = self.cryptor.encrypt(text.encode())
        return b2a_base64(encrypt_text)

    def decrypt(self, text):
        return self.cryptor.decrypt(a2b_base64(text)).decode()


if __name__ == '__main__':
    # 读取参数配置文件
    config_filename = r'D:\qdaq\test\210721-1\AP4000_paramReceived.json'
    t1 = time.time()
    with open(config_filename, 'r') as f:
        config_data = json.load(f)
    print("读取未加密参数配置文件所需时间", time.time() - t1)
    # 读取界限值
    limit_filename = r'D:\qdaq\test\210721-1\PM61191_limitReceived.json'
    t1 = time.time()
    with open(limit_filename, 'r') as f:
        limit_data = json.load(f)
    print("读取未加密界限值文件所需时间", time.time() - t1)
    # 对比参数配置写入速度
    # 1. 直接写入json文件（旧模式）
    no_en_config_filename = r'D:\qdaq\test\210721-1\no_en_config.json'
    t1 = time.time()
    with open(no_en_config_filename, 'w') as f:
        json.dump(config_data, f)
    print('直接写入参数配置所需的时间：', time.time() - t1)

    # 2.1 加密后写入json文件
    en_config_filename = r'D:\qdaq\test\210721-1\des_1\en_config.json'
    t1 = time.time()
    # 构造密钥
    pc_1 = Descrypt('synovate')
    en_config_data = pc_1.encrypt(json.dumps(config_data))
    with open(en_config_filename, 'wb') as f:
        f.write(en_config_data)
    print("写入加密后参数配置文件所需的时间： ", time.time() - t1)

    # 2.2 读取加密后的文件
    t1 = time.time()
    # 构造密钥
    pc_1 = Descrypt('synovate')
    with open(en_config_filename, 'rb') as f:
        re_en_config_data = json.loads(pc_1.decrypt(f.read()))
    print("参数配置数据是否一致：", re_en_config_data == config_data)
    print("读取加密后参数配置文件所需的时间： ", time.time() - t1)

    # 对比界限值写入速度
    # 3. 直接写入json文件（旧模式）
    no_en_limit_filename = r'D:\qdaq\test\210721-1\no_en_limit.json'
    t1 = time.time()
    with open(no_en_limit_filename, 'w') as f:
        json.dump(limit_data, f)
    print('直接写入界限值所需时间：', time.time() - t1)

    # 4.1 加密后写入json文件
    en_limit_filename = r'D:\qdaq\test\210721-1\des_1\en_limit.json'
    t1 = time.time()
    # 构造密钥
    pc_2 = Descrypt('synovate')
    en_limit_data = pc_2.encrypt(json.dumps(limit_data))
    with open(en_limit_filename, 'wb') as f:
        f.write(en_limit_data)
    print("加密后写入界限值文件所需的时间： ", time.time() - t1)

    # 4.2 读取加密后的文件
    t1 = time.time()
    # 构造密钥
    pc_2 = Descrypt('synovate')
    with open(en_limit_filename, 'rb') as f:
        re_en_limit_data = json.loads(pc_2.decrypt(f.read()))
    print("界限值数据是否一致：", re_en_limit_data == limit_data)
    print("读取加密后界限值文件所需的时间： ", time.time() - t1)

    pc_3 = Descrypt('synovate1234')
    with open(en_config_filename, 'rb') as f:
        en_data = f.read()

    # 使用正确的密钥进行解密
    de_right_data = json.loads(pc_1.decrypt(en_data))
    print(de_right_data.keys())
    # 使用不正确的密钥进行解密
    try:
        de_wrong_data = pc_3.decrypt(en_data)
    except Exception:
        traceback.print_exc()
    # 字符串独立测试
    test_1_str = '1234567890'

    en_test_1_str = pc_1.encrypt(test_1_str)
    de_test_1_str = pc_1.decrypt(en_test_1_str)
    print(en_test_1_str, de_test_1_str)
    pass

