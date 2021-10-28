import numpy as np

def get_revolution_by_speed(tn, tn1, vn, vn1, N):
    '''
    angular resampling
    *****************************************************
    parameters
           tn: time instance 1
          tn1: time instance 2
           vn: speed at tn
          vn1: speed at tn1
            N: angle in revolution
    return
          cnt: time at every N revolution
    '''
    cnt = []

    k = (vn1 - vn) / (tn1 - tn)
    if k == 0:
        T = vn * (tn1 - tn)
        cnt = tn + np.arange(1, T//N + 1) * (N / vn)
    else:
        a1 = tn - vn / k
        revolution = 0
        while True:
            revolution += N
            loc = a1 + np.sqrt(vn**2 + 2*k*revolution) / k
            if loc > tn1:
                break
            cnt.append(loc)
    return cnt
