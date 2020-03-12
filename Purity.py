import numpy as np
import h5py
import matplotlib.pyplot as plt
from random import randrange
from scipy import interpolate
from scipy import signal
from scipy import fft, ifft
from scipy.signal import argrelextrema
import multiprocessing as mp
import ctypes

from matplotlib import cm


def spatial_contrast(intens, region):
    #region = [50, 100]
    n_models = intens.shape[0]
    size_1D = intens.shape[1]
    centre_p = (0 + size_1D - 1)/2.
    coordinate = np.array([(x,y) for x in range(size_1D) for y in range(size_1D)])
    x_c = coordinate[:,0] - centre_p
    y_c = coordinate[:,1] - centre_p
    filter = np.where(((x_c**2 + y_c**2)**(0.5) > region[0]) & ((x_c**2 + y_c**2)**(0.5) < region[1]))



    contrast_models = np.zeros(intens.shape[0])
    for i in range(n_models):

        z_c = intens[i].ravel()/(np.sum(intens[i].ravel())/(size_1D**2))
        #plt.scatter(x_c[nHQ], y_c[nHQ], c=np.log10(z_c[nHQ]), s=1)
        i_map = z_c[filter]
        i_map = i_map/np.sum(i_map)*len(i_map)
        ave_map = np.sum(i_map)/len(i_map)
        contrast = (np.sum([(i_map[k] - ave_map)**2 for k in range(len(i_map))])/len(i_map))**(0.5)
        contrast_models[i] = contrast
        print("model:", i, "contrast = ", contrast)

    return contrast_models, filter


def radial_azimuthal_variation(intens, n_angbin, rbin, mask, binsize):
    #rbins= np.arange(65)*2+60
    #n_angbin=60

    def binned_sted( xx,yy,bins,binsize ):
        nb=len(bins)
        yy_mean=bins*0.0
        yy_std=bins*0.0
        for i in range(0,nb):
            n_in_bin=np.where((xx< bins[i]+binsize/2) & (xx> bins[i]-binsize/2))
            #print(bins[i]-binsize/2,bins[i]+binsize/2)
            yy_mean[i]=np.mean(yy[n_in_bin])
            yy_std[i]=np.std(yy[n_in_bin])
        return bins,yy_mean,yy_std

    def HQ_std_RA(x0, y0, intens_i, n_angbin, r, bins, binsize):
        angbin = 3.1416/(n_angbin)
        z0 = intens_i/(np.sum(intens_i.ravel())/(size_1D**2))
        fp0 = interpolate.interp2d(x0, y0, z0, kind='cubic')

        x_matrix = np.array([r * np.cos(i*angbin) for i in range(n_angbin)])
        y_matrix = np.array([r * np.sin(i*angbin) for i in range(n_angbin)])

        z_matrix = np.array([fp0(x_matrix[i,j],y_matrix[i,j])[0] for i in range(n_angbin) for j in range(size_1D)])
        z_matrix = z_matrix.reshape(n_angbin, size_1D)

        radii_vri = [np.std(np.concatenate((z_matrix[:,k], z_matrix[:,size_1D - k-1]), axis = 0)) for k in range(220)]
        radii_vri = radii_vri[::-1]

        BB=binned_sted(np.arange(220),np.log10(radii_vri),rbins,binsize)
        HQ_std_RA = np.sum(BB[2])/n_bins

        return HQ_std_RA



    n_models = intens.shape[0]
    size_1D = intens.shape[1]
    centre_p = (0 + size_1D - 1)/2.
    radil_size = int(centre_p)

    r = np.arange(size_1D) - (size_1D-1)/2
    x_matrix = np.zeros([n_angbin, size_1D])

    x0 = np.arange(size_1D) - (size_1D-1)/2
    y0 = np.arange(size_1D) - (size_1D-1)/2

    angbin = 3.1416/(n_angbin)
    radii_vri = np.zeros([n_models, radil_size])
    radii_vri_fft = np.zeros([n_models, radil_size - mask])

    nproc = 4
    HQ_std_RA = mp.Array(ctypes.c_double, n_models)
    HQ_std_models = mp.Array(ctypes.c_double, n_models)

    def mp_worker(rank, models, HQ_std_models): #models = np.range(n_models)
        irange = models[rank::nproc]
        for i in irange:
            HQ_std_models[i] = HQ_std_RA(x0, y0, intens[i], n_angbin, r, rbins, binsize)
            if rank == 0:
                print("PP (%d):"%i, HQ_std_models[i])

    # Get pairs to be processed
    ind = np.arange(n_models)
    jobs = [mp.Process(target=mp_worker, args=(rank, ind, HQ_std_RA)) for rank in range(nproc)]
    [j.start() for j in jobs]
    [j.join() for j in jobs]

    HQ_std_models = np.frombuffer(HQ_std_models.get_obj())

    return HQ_std_models





filename =  'output_099.h5'
h5 = h5py.File(filename,'r')
#h5.keys() #<KeysViewHDF5 ['intens', 'inter_weight', 'likelihood', 'mutual_info', 'occupancies', 'orientations', 'scale']>
intens = np.array(h5['intens'])
orientations = np.array(h5['orientations'])
h5.close()

rbins = np.arange(65)*2+60
n_angbin = 60
mask = 50
binsize = 20
HQ_std_models = radial_azimuthal_variation(intens, n_angbin, rbins, mask, binsize)

region = [50,200]
contrast_models, filter = spatial_contrast(intens, region)



nHQ = filter

cmpmst = (contrast_models-min(contrast_models))/(max(contrast_models)-min(contrast_models))
colormp = cm.viridis(cmpmst)
