import sys
sys.path.append('/home/ayyerkar/.local/dragonfly/utils/py_src/')
import detector
import reademc
import numpy as np
import ctypes
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import multiprocessing as mp
import h5py
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
%matplotlib


##mask_emc = np.array([ not i for i in mask_emc0]).reshape(319, 331)                        #mask file

#poisson
det = detector.Detector('det_2160_lowq8.h5', mask_flag=True)
det_mask = det.unassembled_mask
rad_map = np.ceil(np.sqrt(det.cx**2+det.cy**2))
rad_masked = (rad_map[det_mask == 1])
rad_masked = rad_masked.astype(int)
#with open('photons_corr.txt', 'r') as f:
#     emc_flist = [i.strip() for i in f.readlines()]


emc_flist = '../../emc/scorr/cub42_lowq.emc'

emc = reademc.EMCReader(emc_flist,det)
n_frames = emc.num_frames

h5 = h5py.File('/gpfs/exfel/u/scratch/SPB/201802/p002160/yulong/cub42_0001/data/cub_sph/Whole_sample7.h5', 'r')
pm0 = h5['PCA_mean_0'][:]
pm1 = h5['PCA_mean_1'][:]
pm2 = h5['PCA_mean_2'][:]
h5.close()
nsp = np.where((pm0 < 0) & (pm1 > 1.35))[0]



frame_p = emc.get_frame(0, raw=True)*0+1
radcount = np.zeros(rad_masked.max() + 1)
np.add.at(radcount, rad_masked, frame_p[det_mask==1])
radmask = (radcount > 10)
radcount[radcount == 0] = 1
radii = np.arange(220)



qvals = 2. * np.sin(0.5 * np.arctan(np.arange(220)*0.2/705)) * 6/12.3984 * 10
diameters = np.arange(30,80,0.1)
svals = np.outer(diameters, qvals) * np.pi
svals[svals==0] = 1e-6
spheremodels = (np.sin(svals) - svals * np.cos(svals))**2 / svals**6 * diameters[:,np.newaxis]**6

n_frames = len(nsp)
cc = np.zeros(n_frames)
dia = np.zeros(n_frames)
Rq = np.zeros((n_frames, 100))
Profile_obs = np.zeros((n_frames, 100))
Profile_ideal = np.zeros((n_frames, 100))
evenlop_obs = np.zeros((n_frames, 100))
evenlop_ideal = np.zeros((n_frames, 100))
index = np.zeros(n_frames)
index0 = np.zeros(n_frames)
for i in range(100):
    f_i = emc.get_frame(nsp[i], raw=True)
    data_i = f_i[det_mask==1]
    radavg = np.zeros_like(radcount)
    tdata = data_i
    np.add.at(radavg, rad_masked, tdata)
    radavg /= radcount

    corrs = np.corrcoef(radavg[20:120], spheremodels[:,20:120])[0, 1:]
    Profile_obs[i] = radavg[20:120]/np.max(radavg[20:120])
    Profile_ideal[i] = spheremodels[corrs.argmax(),20:120]/np.max(spheremodels[corrs.argmax(),20:120])
    pf_obs_max = argrelextrema(Profile_obs[i], np.greater)[0]
    pf_ideal_max = argrelextrema(Profile_ideal[i], np.greater)[0]
    evenlop_obs[i] =  np.interp(radii[20:120],pf_obs_max,Profile_obs[i][pf_obs_max])
    evenlop_ideal[i] =  np.interp(radii[20:120],pf_ideal_max,Profile_ideal[i][pf_ideal_max])
    cc[i] = corrs.max()
    dia[i] = diameters[corrs.argmax()]
    Rq[i] = Profile_obs[i]/ Profile_ideal[i]
    index[i] = np.sum(Rq[i]*radii[20:120]**2)/np.sum(Rq[i])
    index0[i] = np.sum(Profile_obs[i])/np.sum(Profile_ideal[i])

plt.plot(evenlop_obs[i], color='blue')
plt.plot(Profile_obs[i], color='blue')

plt.plot(evenlop_ideal[i], color='red')
plt.plot(Profile_ideal[i], color='red')
plt.yscale('log')

plt.plot(Rq[i])



print(index[i])


#np.add.at(radcount, rad_masked, view1[det_mask==1])
###############################################
intens_tot = mp.Array(ctypes.c_double, n_frames)
cc = mp.Array(ctypes.c_double, n_frames)
error = mp.Array(ctypes.c_double, n_frames)
dia = mp.Array(ctypes.c_double, n_frames)
Rq = np.Array(ctypes.c_double, n_frames, )
def mp_worker(rank, indices, dia, intens_tot, cc, error):
    irange = indices[rank::nproc]

    for i in irange:
        f_i = emc.get_frame(i, raw=True)
        data_i = f_i.data[mask_center]
        radavg = np.zeros_like(radcount)
        tdata = data_i
        np.add.at(radavg, flotrad, tdata)
        radavg /= radcount

        corrs = np.corrcoef(((radcount)[20:120]), (spheremodels)[:,20*120])[0, 1:]
        cc[i] = corrs.max()
        dia[i] = diameters[corrs.argmax()]
        intens_tot[i] = np.sum(tdata)
        if (intens_tot[i] >=10):
            error[i] = fis(np.log10(intens_tot[i]))
        if (intens_tot[i] < 10):
            error[i] = 100

        if rank == 0:
            print("CC (%d):"%i, ' intens = ', intens_tot[i], ' dia = ', dia[i], ' err = ', error[i])



nproc = 16
ind = np.arange(n_frames)
jobs = [mp.Process(target=mp_worker, args=(rank, ind, dia, intens_tot, cc, error)) for rank in range(nproc)]
[i.start() for i in jobs]
[i.join() for i in jobs]
dia = np.frombuffer(dia.get_obj())
intens_tot = np.frombuffer(intens_tot.get_obj())
cc = np.frombuffer(cc.get_obj())
error = np.frombuffer(error.get_obj())
###############################################
with h5py.File('diameter_weighted_error_mask0.h5', 'w') as f:
    f['Run_num'] = running_num
    f['diameter'] = dia
    f['intens_tot'] = intens_tot
    f['error'] = error
    f['cc'] = cc


np.savez('diameter_weighted_error_mask0.npz', dia=dia, intens=intens_tot, error=error, cc=cc)
