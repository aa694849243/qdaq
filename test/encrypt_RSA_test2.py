#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2021/7/20 19:26
@Author : Wall@Synovate
@Email  : qxu@sonustc.com
@Version: 1.0.0
reference: https://blog.csdn.net/weixin_30347335/article/details/99123821?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-2.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-2.control
"""
from Cryptodome import Random
from Cryptodome.PublicKey import RSA
from Cryptodome.Cipher import PKCS1_v1_5
from binascii import b2a_hex, a2b_hex
import json
import traceback
import time


class Rsaencrypt:
    # rsa加密
    def __init__(self, public_key):
        # 这里的1024是二进制位数, 也就是说他加密的内容只有1024 / 8 = 128个字节,
        # 但是里面又有着11个字节是它必须有的, 所以最长只能加密117个字节，加密时长度为117
        self.RSA_EN_LENGTH = 117
        self.cipher = PKCS1_v1_5.new(public_key)

    def raw_encrypt(self, text):
        self. ciphertext = self.cipher.encrypt(text.encode())
        # 因为rsa加密时候得到的字符串不一定是ascii字符集的，输出到终端或者保存时候可能存在问题
        # 所以这里统一把加密后的字符串转化为16进制字符串
        return b2a_hex(self.ciphertext)

    def encrypt(self, text):
        if len(text) <= self.RSA_EN_LENGTH:
            # 不需要分段
            return self.raw_encrypt(text)
        else:
            # 需要分段
            text_list = list()
            start_index = 0
            text_length = len(text)
            while start_index < text_length:
                if start_index + self.RSA_EN_LENGTH <= text_length:
                    # 如果长度依然不止一次可以处理完
                    text_list.append(self.raw_encrypt(text[start_index: start_index + self.RSA_EN_LENGTH]))
                else:
                    text_list.append(self.raw_encrypt(text[start_index:]))
                start_index += self.RSA_EN_LENGTH
            return b''.join(text_list)


class Rsadecrypt:
    # rsa解密
    def __init__(self, private_key):
        # 这里的1024是二进制位数, 也就是说他加密的内容只有1024 / 8 = 128个字节,
        # 但是里面又有着11个字节是它必须有的, 所以最长只能加密117个字节，解密时长度为256
        self.RSA_DE_LENGTH = 256
        self.cipher = PKCS1_v1_5.new(private_key)

    def raw_decrypt(self, text):
        decrypt_text = self.cipher.decrypt(a2b_hex(text), None)
        return decrypt_text.decode()

    def decrypt(self, text):
        text_list = list()
        for i in range(0, len(text)//self.RSA_DE_LENGTH):
            text_list.append(self.raw_decrypt(
                text[i * self.RSA_DE_LENGTH: (i + 1) * self.RSA_DE_LENGTH]))
        return ''.join(text_list)


def save_pem(p_pem, key_filename):
    """
    功能：保存公钥或私钥
    输入：
    1. 密钥
    2. 保存文件名（全路径）
    返回：无
    """
    with open(key_filename, 'w')as fp:
        fp.write(p_pem)


def read_key(key_filename, key_type=0):
    """
    功能：读取密钥（公钥或私钥）
    输入：
    1. 密钥文件（全路径）
    2. 密钥类型（0为公钥，1为私钥）
    返回：密钥
    """
    with open(key_filename, 'rb')as fp:
        p_info = fp.read()
    if key_type:
        # 读取私钥
        p_key = RSA.import_key(p_info)
    else:
        p_key = RSA.import_key(p_info)
    return p_key


if __name__ == '__main__':
    # 字符串独立测试
    # 生成密钥
    # 伪随机数生成器
    random_generator = Random.new().read
    # rsa算法生成实例
    rsa_1 = RSA.generate(1024, random_generator)
    private_pem = str(rsa_1.exportKey(), encoding="utf-8")
    public_pem = str(rsa_1.publickey().exportKey(), encoding="utf-8")
    # 保存密钥
    pub_key_filename = r'D:\qdaq\test\210721-1\rsa_2\public.pem'
    pri_key_filename = r'D:\qdaq\test\210721-1\rsa_2\private.pem'
    save_pem(public_pem, pub_key_filename)
    save_pem(private_pem, pri_key_filename)
    # 读取密钥
    pub_key = read_key(pub_key_filename)
    pri_key = read_key(pri_key_filename, 1)
    test_1_str = 'hello'*200
    rsa_en_1 = Rsaencrypt(pub_key)
    en_test_1_str = rsa_en_1.encrypt(test_1_str)
    print(len(en_test_1_str))
    rsa_de_1 = Rsadecrypt(pri_key)
    de_test_1_str = rsa_de_1.decrypt(en_test_1_str)
    print(en_test_1_str)
    print(de_test_1_str == test_1_str)

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
    print('直接写入参数配置所需时间：', time.time() - t1)

    # 2. 加密后写入json文件
    # 初始化密钥
    en_config_filename = r'D:\qdaq\test\210721-1\rsa_2\en_config.json'
    t2 = time.time()
    # 读取密钥并生成相关方法
    pub_key = read_key(pub_key_filename)
    rsa_en_2 = Rsaencrypt(pub_key)
    en_config_data = rsa_en_2.encrypt(json.dumps(config_data))
    with open(en_config_filename, 'wb') as f:
        f.write(en_config_data)
    print("加密后写入参数配置所需时间： ", time.time() - t2)

    # 对比界限值写入速度
    # 1. 直接写入json文件（旧模式）
    no_en_limit_filename = r'D:\qdaq\test\210721-1\no_en_limit.json'
    t1 = time.time()
    with open(no_en_limit_filename, 'w') as f:
        json.dump(limit_data, f)
    print('直接写入界限值所需时间：', time.time() - t1)

    # 2. 加密后写入json文件
    en_limit_filename = r'D:\qdaq\test\210721-1\rsa_2\en_limit.json'
    t2 = time.time()
    # 读取密钥并生成相关方法
    pub_key = read_key(pub_key_filename)
    rsa_en_2 = Rsaencrypt(pub_key)
    en_limit_data = rsa_en_2.encrypt(json.dumps(limit_data))
    with open(en_limit_filename, 'wb') as f:
        f.write(en_limit_data)
    print("加密后写入界限值所需时间： ", time.time() - t2)

    # 读取加密文件并解密
    # 1. 读取加密后的界限值
    t1 = time.time()
    # 读取密钥并生成相关方法
    pri_key = read_key(pri_key_filename, 1)
    rsa_de_2 = Rsadecrypt(pri_key)
    with open(en_config_filename, 'rb') as f:
        re_en_config_data = json.loads(rsa_de_2.decrypt(f.read()))
    print("参数配置数据是否一致：", re_en_config_data == config_data)
    print("读取加密后参数配置文件所需的时间： ", time.time() - t1)

    t1 = time.time()
    # 读取密钥并生成相关方法
    pri_key = read_key(pri_key_filename, 1)
    rsa_de_2 = Rsadecrypt(pri_key)
    with open(en_limit_filename, 'rb') as f:
        re_en_limit_data = json.loads(rsa_de_2.decrypt(f.read()))
    print("界限值数据是否一致：", re_en_limit_data == limit_data)
    print("读取加密后界限值文件所需的时间： ", time.time() - t1)

    # 使用不正确的密钥进行解密
    old_pub_key_filename = r'D:\qdaq\test\210721-1\rsa_2\public_old.pem'
    old_pri_key_filename = r'D:\qdaq\test\210721-1\rsa_2\private_old.pem'
    old_pub_key = read_key(old_pub_key_filename)
    old_pri_key = read_key(old_pri_key_filename, 1)
    rsa_de_3 = Rsadecrypt(old_pri_key)
    try:
        de_wrong_data = rsa_de_3.decrypt(en_limit_data)
    except Exception:
        traceback.print_exc()
    pass

