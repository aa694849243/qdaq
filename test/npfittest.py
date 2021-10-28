import time
import numpy as np
from multiprocessing import shared_memory,Process
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt



def fit1d(loc,angle,d):
    time_for_son=0
    for i in range(1000):
        time_before_ploy_s=time.time()
        np.polyfit(loc,angle,d)
        time_after_ploy_s=time.time()
        time_for_son+=time_after_ploy_s-time_before_ploy_s

    print("time_for_ploy_son %.16f"%(time_for_son))




if __name__ == '__main__':
    angle = [34.288814280659274, 33.17760712428286, 33.63864304747154, 33.343361671971564, 31.85928893319166,
             32.97656889087063, 33.88876100496713, 31.521689627712604, 30.53346526711111, 30.974243655771563,
             29.82528634247947, 28.82862051437445, 27.81429574300292, 26.54256068784347, 27.524794432583082,
             28.208312204714485, 26.705686795333296, 25.703602842541663, 24.198544547623126, 22.80630505411347,
             24.750673252159864, 25.662944449983428, 23.119206517120503, 22.178629768778325, 22.77361932598782,
             22.414070572960973, 21.903719897760578, 20.226971987092806, 19.068752291050775, 19.53032823133926,
             19.039265383117446, 19.675332307382664, 20.75341041257832, 18.212851545299745, 16.187794923893026,
             16.976610338564775, 16.613616971387533, 15.882433073666848, 15.061097982815136, 14.066027657806705,
             14.798568025789017, 14.36050876623441, 12.763377305103694, 12.543334022145169, 11.231623136106524,
             10.525091303787057, 12.53793091231839, 12.125521636110108, 9.974340338552942, 9.600260863004896,
             8.516326072837073, 8.396926071421623, 8.685325691638033, 7.322174436768506, 7.009847411774293,
             7.056452647378381, 6.0474631708800475, 6.889679944254944]
    loc = [648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669,
           670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691,
           692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705]

    angle_array=np.array(angle)
    loc_array=np.array(loc)
    time_for_father=0
    time_for_scipy=0
    count=1
    for i in range(count):
        time_before_ploy_f = time.time()
        # t=np.polyfit(loc_array, angle_array, 1)
        t=np.interp(1,loc,angle)
        print(t) #print 语句可能要占运行时间
        time_after_ploy_f = time.time()
        time_for_father+=time_after_ploy_f - time_before_ploy_f

    print("time_for_ploy_father %.16f" % (time_for_father))

    for i in range(count):
        time_before_ploy_scipy=time.time()
        coff=interp1d(loc_array,angle_array,kind='linear', fill_value='extrapolate', assume_sorted=True)(1)

        # print(coff(1))
        print(coff)
        time_after_ploy_scipy=time.time()
        time_for_scipy += time_after_ploy_scipy - time_before_ploy_scipy

    print("time_for_ploy_scipy %.16f" % (time_for_scipy))

    x=range(10)
    y=[0,1,3,2,5,7,6,9,8,10]
    insert=np.array(range(9))+0.5
    npresult=np.interp(insert,x,y)
    scresult=interp1d(x,y,kind='cubic', fill_value='extrapolate', assume_sorted=True)(insert)

    plt.plot(x,y,color='r',label="origin")
    # plt.plot(insert,npresult,color='b',label="npresult")
    plt.scatter(insert,scresult,color='g',label="scresult")
    plt.legend()
    plt.show()

    # p1 = Process(target=fit1d, args=(loc, angle, 1))
    # p1.start()
    # p1.join()
    # print("done")
    # p1.terminate()
