#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2021/7/20 18:11
@Author : Wall@Synovate
@Email  : qxu@sonustc.com
@Version: 1.0.0
reference: https://www.cnblogs.com/xiao-apple36/p/8744408.html
"""

from Cryptodome.Cipher import AES
from binascii import b2a_base64, a2b_base64
import time
import json
import traceback


class Aescrypt:
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
    en_config_filename = r'D:\qdaq\test\210721-1\aes\en_config.json'
    t1 = time.time()
    # 构造密钥
    pc_1 = Aescrypt('1234567890')
    en_config_data = pc_1.encrypt(json.dumps(config_data))
    with open(en_config_filename, 'wb') as f:
        f.write(en_config_data)
    print("写入加密后参数配置文件所需的时间： ", time.time() - t1)

    # 2.2 读取加密后的文件
    t1 = time.time()
    # 构造密钥
    pc_1 = Aescrypt('1234567890')
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
    en_limit_filename = r'D:\qdaq\test\210721-1\aes\en_limit.json'
    t1 = time.time()
    # 构造密钥
    pc_2 = Aescrypt('synovate1122')
    en_limit_data = pc_2.encrypt(json.dumps(limit_data))
    with open(en_limit_filename, 'wb') as f:
        f.write(en_limit_data)
    print("加密后写入界限值文件所需的时间： ", time.time() - t1)

    # 4.2 读取加密后的文件
    t1 = time.time()
    # 构造密钥
    pc_2 = Aescrypt('synovate1122')
    with open(en_limit_filename, 'rb') as f:
        re_en_limit_data = json.loads(pc_2.decrypt(f.read()))
    print("界限值数据是否一致：", re_en_limit_data == limit_data)
    print("读取加密后界限值文件所需的时间： ", time.time() - t1)

    pc_3 = Aescrypt('synovate')
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
