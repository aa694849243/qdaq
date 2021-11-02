import multiprocessing
import time
import os

def fun():
    print('startfun')
    print(str(os.getppid())+'父进程')
    print(os.getpid())
    time.sleep(1)
    print('endfun')


if __name__ == '__main__':
    multiprocessing.freeze_support()  # 保护主进程
    p = multiprocessing.Process(target=fun, )
    print(str(os.getpid())+'main')
    # p.daemon = True
    p.start()
    print('123')
