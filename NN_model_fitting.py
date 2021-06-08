#Step_3
#DNN Fitting
import h5py
import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fnum = h5py.File('ACC10k_fitting_frames.h5', 'r')
intens = fnum['intens'][:]
fft_maps = fnum['radial_fft_intens'][:]
fnum.close()

x,y = np.indices((130,40))
x1 = x.ravel(); y1 = y.ravel()
nn = np.where(y1 < 10+0.25*x1)[0]
nn_0 = np.where(y1 > 10+0.25*x1)[0]
nnn = nn.shape[0]

fftmap0 = fft_maps[0][:,0::2][20:150,:40]
fftmap1 = fftmap0.ravel()[nn]

fitting_data = np.zeros((fft_maps.shape[0],nnn))
for i in range(fft_maps.shape[0]):
    fftmap0 = fft_maps[i][:,0::2][20:150,:40]
    fitting_data[i] = fftmap0.ravel()[nn]

plt.figure()
plt.imshow(np.log10(fitting_data))
plt.colorbar()
plt.savefig('input_data.png')

savfilename = "fitting_data.npz"
np.savez(savfilename, fitting_data=fitting_data)

#fnum = h5py.File('ACC10k_fitting_data.h5', 'w')
#fnum['intens'] = intens
#fnum['fitting_data'] = fitting_data
#fnum['radial_fft_intens'] = fft_maps
#fnum.close()





fd = np.load('fitting_data.npz')
fitting_data = fd['fitting_data']





model_path = '/home/zhuangyu/p002160/scratch/yulong/cub42_0001/data/scorr/CC_ACC/'
model = tf.keras.models.load_model(model_path + 'ACC_model_X80')

predict_PCA_coor = model.predict(fitting_data)

plt.scatter(predict_PCA_coor[:,0], predict_PCA_coor[:,1], s=0.1)
plt.savefig('PPCA_0_1.png')

fnum = h5py.File('ACC10k_PCA_coor.h5', 'w')
fnum['intens'] = intens
fnum['fitting_data'] = fitting_data
fnum['predict_PCA_coor'] = predict_PCA_coor
fnum.close()


#/gpfs/exfel/u/scratch/SPB/201802/p002160/yulong/cub42_0001/data/ACC10K/Resutls/ACC10k_PCA_coor.h5
