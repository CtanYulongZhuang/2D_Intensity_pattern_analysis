#Working ok max-exfl to extract the 2d dense pattern from 2d intens patterns per classes
#using the script phasing
import sys
#sys.path.append('/Users/yulongzhuang/Documents/Code/My_libs/SPI')
sys.path.append('/home/zhuangyu/Libs/SPI')
import phasing
from scipy import ndimage
import multiprocessing as mp
import ctypes
import numpy as np
import h5py
import matplotlib.pyplot as plt

input_path = ''
filename = input_path + 'output_050.h5'
h5 = h5py.File(filename,'r')
#h5.keys() #<KeysViewHDF5 ['intens', 'inter_weight', 'likelihood', 'mutual_info', 'occupancies', 'orientations', 'scale']>
intens = np.array(h5['intens'])
orientations = np.array(h5['orientations'])
h5.close()

nclass = intens.shape[0]
x, y = np.indices(intens.shape[1:]); x-=220; y -= 220
#x, y = np.indices(np.array(intens.shape)[[1,2]]); x-=220; y -= 220
rad = np.sqrt(x*x + y*y)

alg_strings = ['ER']*50 +['DM']*100 + ['ER']*50

res_den = np.zeros([nclass,intens.shape[1],intens.shape[1]])
res_su = np.zeros([nclass,intens.shape[1],intens.shape[1]])
res_hist = np.zeros([nclass,50])
hist_factor = np.zeros([nclass,16])
max_factor = np.zeros(nclass)

for j in range(nclass):
    intens_i = intens[j]
    size = intens_i.shape[1]
    intens_i = intens_i - ndimage.minimum_filter(intens_i, (18,18))
    intens_i[(rad < 50) & (intens_i == 0)] = -1
    #intens_i[(rad < 35)] = -1
    intens_i[(rad > 205) & (rad < 219)] = -1

    print('Class:', j)

    res_den_i = mp.Array(ctypes.c_double, 16*size**2)
    res_su_i = mp.Array(ctypes.c_double, 16*size**2)
    res_hist_i = mp.Array(ctypes.c_double, 16*50)
    hist_factor_i = mp.Array(ctypes.c_double, 16)

    def mp_worker(rank, indices, res_den_i, res_su_i, res_hist_i, hist_factor_i):
        irange = indices[rank::nproc]

        for i in irange:
            num_supp = 600 + 100*i
            phaser = phasing.ModePhaser(intens_i, num_supp = num_supp)
            phaser.phase(alg_strings)
            res_den_i[i*size**2:(i+1)*size**2] = phaser.current.ravel()
            res_su_i[i*size**2:(i+1)*size**2] = phaser.support.ravel()
            hi_i = np.histogram(phaser.current[phaser.support], bins=50)[0]
            res_hist_i[i*50:(i+1)*50] = hi_i
            hist_factor_i[i] = hi_i[0]/np.sum(hi_i[10:])
            if rank == 0:
                print("PP (%d):"%i, hist_factor_i[i])

    nproc = 8
    indices = np.arange(16)
    jobs = [mp.Process(target=mp_worker, args=(rank, indices, res_den_i, res_su_i, res_hist_i, hist_factor_i)) for rank in range(nproc)]
    [j.start() for j in jobs]
    [j.join() for j in jobs]

    res_den_i = np.frombuffer(res_den_i.get_obj()).reshape(16,size, size)
    res_su_i = np.frombuffer(res_su_i.get_obj()).reshape(16,size, size)
    res_hist_i = np.frombuffer(res_hist_i.get_obj()).reshape(16,50)
    hist_factor_i = np.frombuffer(hist_factor_i.get_obj())

    n_max = np.where(hist_factor_i == np.max(hist_factor_i))[0][0]
    res_den[j] = res_den_i[n_max]
    res_su[j] = res_su_i[n_max]
    res_hist[j] = res_hist_i[n_max]
    hist_factor[j] = hist_factor_i
    max_factor[j] = n_max


intens_r = intens*0
for i in range(nclass):
    intens_r[i] = np.abs(np.fft.fftshift(np.fft.fftn(res_den[i])))

savfilename = input_path + "Gold_rec2D.npz"
np.savez(savfilename, res_den=res_den,res_su=res_su, res_hist=res_hist, \
hist_factor=hist_factor, intens_r=intens_r, intens=intens, max_factor=max_factor)



np.histogram(phaser.current[phaser.support], bins=50)[0]
