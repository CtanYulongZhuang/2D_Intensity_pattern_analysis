import sys
sys.path.append('/home/ayyerkar/.local/dragonfly/utils/py_src/')
import detector
import reademc
import numpy as np
import ctypes
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import multiprocessing as mp

#poisson
det = detector.Detector('det_2160_lowq8.h5', mask_flag=True)
#with open('photons_corr.txt', 'r') as f:
#     emc_flist = [i.strip() for i in f.readlines()]

emc_flist = '../../emc/corr/cub42_lowq.emc'

emc = reademc.EMCReader(emc_flist,det)
n_frames = emc.num_frames

clist = np.array([f['num_data'] for f in emc.flist])
running_num = np.zeros(np.max(clist))
running_num[0:clist[0]] = np.int( emc_flist[0][75:78] )
for i in range(len(clist)-1):
    running_num[clist[i]:clist[i+1]] = np.int( emc_flist[i+1][75:78] )



mask_emc0 = emc.get_frame(1).mask.ravel()
mask_emc0 = np.array([ not i for i in mask_emc0]).reshape(512, 256)
mask_center = np.load('../aux/mask_center_cmc_0.npy')                           #mask file

intrad = np.load('../aux/intrad_004.npy')
intrad = intrad[256:768,256:512]
flotrad = intrad[mask_center]
s_s = np.load('../aux/int2std_model_r30_mask0_weighted.npz')
std_models = s_s['std_model']
int_models = s_s['intens_model']
std_smooth = savgol_filter(std_models, 51, 3)
fis = interp1d(np.log10(int_models), std_smooth,'cubic')


radcount = np.zeros(intrad.max() + 1)
np.add.at(radcount, intrad, mask_center)
radmask = (radcount > 10)
radcount[radcount == 0] = 1

det_dist = 100 # mm
qvals = 2. * np.sin(0.5 * np.arctan(np.arange(int(intrad.max())+1) * 0.075 / det_dist)) / 4.5
diameters = np.linspace(10, 50, 1200)
svals = np.outer(diameters, qvals) * np.pi
svals[svals==0] = 1e-6
spheremodels = (np.sin(svals) - svals * np.cos(svals))**2 / svals**6 * diameters[:,np.newaxis]**6


###############################################
intens_tot = mp.Array(ctypes.c_double, n_frames)
cc = mp.Array(ctypes.c_double, n_frames)
error = mp.Array(ctypes.c_double, n_frames)
dia = mp.Array(ctypes.c_double, n_frames)

def mp_worker(rank, indices, dia, intens_tot, cc, error):
    irange = indices[rank::nproc]

    for i in irange:
        f_i = emc.get_frame(i)
        data_i = f_i.data[mask_center]
        radavg = np.zeros_like(radcount)
        tdata = data_i
        np.add.at(radavg, flotrad, tdata)
        radavg /= radcount

        corrs = np.corrcoef(((radcount*radavg)[radmask]), (radcount*spheremodels)[:,radmask])[0, 1:]
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
