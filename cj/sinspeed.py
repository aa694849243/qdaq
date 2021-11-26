import matplotlib.pyplot as plt
from utils import read_raw_data

filename = "D:\\qdaq\\Simu\\byd-alltests-1mic.h5"
data = read_raw_data(filename, ['Cos', 'Mic1', 'Sin', 'Vib1'],'hdf5')
plt.plot(data['Sin'])
plt.show()