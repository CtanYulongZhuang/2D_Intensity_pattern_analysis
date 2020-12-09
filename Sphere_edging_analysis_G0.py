import sys
import multiprocessing as mp
import ctypes
import numpy as np
import h5py
from scipy import ndimage

import matplotlib.pyplot as plt

sys.path.append('/home/ayyerkar/.local/dragonfly/utils/py_src/')
import detector, reademc


det_fname = '../cub42_0001/det_2160_lowq8.h5'
emc_fname = '../../emc/scorr/cub42_lowq.emc'

det = detector.Detector(det_fname, mask_flag=True)
emc = reademc.EMCReader(emc_fname, det)

frame0 = emc.get_frame(5, sym=True)

def read_blacklist(blacklist_name):
    file0 = open(blacklist_name, 'r')
    lines = file0.readlines()
    file0.close()
    blacklist = np.array([int(i.split()[0]) for i in lines])
    return blacklist

blacklist = read_blacklist('NN_BL03_gp2.txt')

n_round = np.where(blacklist == 0)[0]
n_frames = n_round.shape[0]




x, y = np.indices((345,355)); x -= 345//2; y -= 355//2
intrad = np.sqrt(x*x + y*y)
radii = np.arange(120)*1
intens_RP_x = np.zeros([n_frames,120])

for i in range(n_frames):
    print(i)
    xxx = emc.get_frame(n_round[i], sym=True)
    xdata = xxx.data
    xmask = xxx.mask
    xdata[xmask] = -1
    #plt.imshow(xxx)
    #nv = np.where(xxx > 0)[0]
    #plt.scatter(intrad[nv], xxx[nv],s=1)

    intens_r = radii*0.0
    flat_xdata = xdata.ravel()
    flat_intrad = intrad.ravel()
    for k in range(len(radii)):
        #print(k)
        nr = np.where((flat_intrad < radii[k]+2) & (flat_intrad > radii[k]) & (flat_xdata > -0.5))[0]
        if (len(nr) != 0):
            intens_r[k] = np.sum(flat_xdata[nr])/len(nr)
        if (len(nr) == 0):
            intens_r[k] = -1
        #print(len(nr))

    intens_RP_x[i] = intens_r

savfilename = "RP_BL03_gp2.npz"
np.savez(savfilename, intens_RP_x=intens_RP_x, blacklist=blacklist)



ps = np.load('RP_BL03_gp2.npz')
intens_RP_x = ps['intens_RP_x']


# Generate sinc function models
radsamples = np.arange(10,50,0.1)
q = 2*np.sin(0.5*np.arctan(np.arange(221)*0.2/705)) * 6/12.3984 * 10
sincmodels_sinc = np.array([np.sinc(q*r)**2 for r in radsamples])
print('Generated models')


#radsamples = np.arange(30,80,0.1)
#q = 2*np.sin(0.5*np.arctan(np.arange(221)*0.2/705)) * 6/12.3984 * 10
#spheremodels = np.array([np.abs((np.sin(np.pi*r*q) - np.pi*r*q*np.cos(np.pi*r*q))/(np.pi*r*q)**3)**2 for r in radsamples])

qvals = 2. * np.sin(0.5 * np.arctan(np.arange(221)*0.2/705)) * 6/12.3984 * 10
diameters = np.arange(30,80,0.1)
svals = np.outer(diameters, qvals) * np.pi
svals[svals==0] = 1e-6
spheremodels = (np.sin(svals) - svals * np.cos(svals))**2 / svals**6 * diameters[:,np.newaxis]**6


psizes = np.zeros(intens_RP_x.shape[0])
pcorr = np.zeros(intens_RP_x.shape[0])
G0 = np.zeros(intens_RP_x.shape[0])

for i in range(intens_RP_x.shape[0]):
    line = intens_RP_x[i][20:120]
    corrs = np.corrcoef(line, spheremodels[:,20:120])[0,1:]
    psizes[i] = diameters[corrs.argmax()]
    pcorr[i] = corrs.max()
    G0[i] = np.sum(line)/np.sum(spheremodels[corrs.argmax(),20:120])
    print(i, '/', intens_RP_x.shape[0])


sizes_output_fname = 'data/MC/cub_ss/Sphere_G0_new.h5'
print('Saving output to ', sizes_output_fname)
with h5py.File(sizes_output_fname, 'w') as f:
    f['size'] = psizes
    f['corr'] = pcorr
    f['G0'] = G0


savfilename = "Sphere_G0_new.npz"
np.savez(savfilename, psizes=psizes, pcorr=pcorr, G0=G0)
