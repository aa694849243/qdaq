#config.ini
## Basic Info
### ni_device_name : ni板卡的名称
### server_ip : 服务器ip
### speed_signal : 转速信号
  - ttl:参数配置中转速通道名字为Speed
  - resolver:(单路旋变) 转速通道为Sin,暂不可配置Cos
  - resolver2:(双路旋变) 参数配置中转速通道为Sin,Cos
  振动通道固定为Vib或mic开头
### sensor_count : 传感器个数
### max_size : 最长采样时长*最大采样率，即最多有多少个数据，越大占用的的内存越大
### mode :
- rt   实采模式
- simu 仿真模式
### version： 
- 1:qdaq程序
- 2:qdaq比亚迪版
- 3:恒速电机转速无波动版
- 4:恒速电机转速波动不重采样版
- 5:恒速电机转速波动用转速重采样版
- 6:恒速电机转速波动用关键阶次重采样

### board: 
- NI:NI板卡采集
- Umic:Umic采集
- DT:dt板卡采集
## Raw Data
- save_type : hdf5/tdms 保存原始数据等文件时的类型
- read_type : simu时读取原始数据时文件的类型

## Umic Info
-  Umic_names:Umic的设备名，是一个list
-  Umic_hostapis:list类型