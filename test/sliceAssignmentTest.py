from multiprocessing import shared_memory
import numpy as np
from icecream import ic
import struct


shm_test=shared_memory.SharedMemory(name='test1',create=True,size=1)
a=np.ndarray(shape=(250,),dtype='f',buffer=shm_test.buf,offset=0)
b=np.ndarray(shape=(250,),dtype='f',buffer=shm_test.buf,offset=0)


ic(1)
shm_test2=shared_memory.SharedMemory(name='test2',create=True,size=6)
a=np.ndarray(shape=(1,),dtype='f',buffer=shm_test2.buf,offset=0)
struct_test=struct.Struct('f')
# struct_test.pack_into(shm_test2.buf, 6, *[1])
shm_test3=shared_memory.SharedMemory(name='test2',size=10)

ic(2)