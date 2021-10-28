import numpy as np
from scipy.optimize import curve_fit
import time

def func(x,a,b):
    return a*x+b

if __name__ == '__main__':

    angle_list = [34.288814280659274, 33.17760712428286, 33.63864304747154, 33.343361671971564, 31.85928893319166,
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
    loc_list = [648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669,
                670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691,
                692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705]



    angle_array=np.array(angle_list)
    loc_array=np.array(loc_list)

    time_for_np_list_fit=0
    time_for_np_array_fit=0
    time_for_sp_list_fit=0
    time_for_sp_array_fit=0

    time_before_np_list_fit=time.time()
    count=0
    while count<1000:
        count+=1
        coff=np.polyfit(loc_list, angle_list, 1)
    time_for_np_list_fit+=time.time()-time_before_np_list_fit

    time_before_np_array_fit=time.time()
    count=0
    while count<1000:
        count+=1
        coff=np.polyfit(loc_array, angle_array, 1)
    time_for_np_array_fit+=time.time()-time_before_np_array_fit


    time_before_sp_list_fit=time.time()
    count=0
    while count<1000:
        count+=1
        popt, pcov=curve_fit(func,loc_list, angle_list)
    time_for_sp_list_fit+=time.time()-time_before_sp_list_fit

    time_before_sp_array_fit=time.time()
    count=0
    while count<1000:
        count+=1
        popt, pcov=curve_fit(func,loc_array, angle_array)
    time_for_sp_array_fit+=time.time()-time_before_sp_array_fit


    print("time_for_np_list_fit:{},time_for_np_array_fit:{},time_for_sp_list_fit:{},time_for_sp_array_fit:{}".format(
        time_for_np_list_fit,
        time_for_np_array_fit,
        time_for_sp_list_fit,
        time_for_sp_array_fit
    ))