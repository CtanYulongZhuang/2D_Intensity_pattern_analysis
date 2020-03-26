
import multiprocessing as mp
import ctypes
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import interpolate

def overall_zimuthal_average(intens, n_angbin):

    n_models = intens.shape[0]
    size_1D = intens.shape[1]

    #n_angbin = 20
    angbin = 3.1416/(n_angbin)
    r = np.arange(size_1D) - (size_1D-1)/2
    x0 = np.arange(size_1D) - (size_1D-1)/2
    y0 = np.arange(size_1D) - (size_1D-1)/2
    sum_line_model_r = r*0.0

    for i in range(n_models):
        z0 = intens[i]
        fp0 = interpolate.interp2d(x0, y0, z0, kind='cubic')
        sum_line_r = r*0.0
        for j in range(n_angbin):
            thita_1 = j*angbin
            line_1_x = r * np.cos(thita_1)
            line_1_y = r * np.sin(thita_1)
            line_1_z = r * 0.0
            line_1_z = np.array([fp0(line_1_x[k],line_1_y[k])[0] for k in range(size_1D)])
            sum_line_r = sum_line_r + line_1_z

        ave_line_r = sum_line_r/n_angbin                                        #average profile of image i
        sum_line_model_r = sum_line_model_r + ave_line_r
        #print("model:", i, "Ave radial profile:", ave_line_r)

    ave_line_model_r = sum_line_model_r/n_models                                #average profile of the whole data
    return ave_line_model_r


def cross_correlation(signal1, signal2):

    length = len(signal1)
    CC_results = np.zeros(length*2)
    for i in range(length):
        sn1 = length - i - 1
        sn2 = i + 1
        #print('sig1: ', signal1[sn1:], 'sig2: ', signal2[:sn2])
        #print('sig1: ', signal2[sn1:], 'sig2: ', signal1[:sn2])
        CC_results[i] = np.sum(signal1[sn1:] * signal2[:sn2])
        CC_results[2*length-i-1] = np.sum(signal2[sn1:] * signal1[:sn2])

    return CC_results




def RotCC_2_images(intens1, intens2, n_angbin, mask_radius):

    #@jit
    def CTY(z_matrix1, z_matrix2):
        n_angbin = z_matrix1.shape[0]
        CC_value = np.empty((n_angbin, n_angbin)) # <--- np.zeros() is rewritten as np.empty()
        CC_value[:] = 0
        STD_value = np.empty((n_angbin, n_angbin)) # <--- np.zeros() is rewritten as np.empty()
        STD_value[:] = 0
        logCC_value = np.empty((n_angbin, n_angbin)) # <--- np.zeros() is rewritten as np.empty()
        logCC_value[:] = 0

        for i in range(n_angbin):
            line_1_z0 = z_matrix1[i]
            line_1_z0[np.where(line_1_z0 < 0.000001)] = 0.000001
            line_1_z1 = line_1_z0/np.sum(line_1_z0)*len(line_1_z0)
            for j in range(n_angbin):
                line_2_z0 = z_matrix2[j]
                line_2_z0[np.where(line_2_z0 < 0.000001)] = 0.000001
                line_2_z1 = line_2_z0/np.sum(line_2_z0)*len(line_2_z0)

                line_ave = (line_1_z1 + line_2_z1)/2.
                line_1_z = line_1_z1/line_ave
                line_2_z = line_2_z1/line_ave


                CC_value[i,j] = np.sum(line_1_z*line_2_z)/(np.sum(line_1_z**2)*np.sum(line_2_z**2))**0.5
                STD_value[i,j] = np.sum((line_1_z - line_2_z)**2)
                log_line_1 = np.log10(line_1_z0+0.001)
                log_line_2 = np.log10(line_2_z0+0.001)
                npos = np.where((log_line_1 > 0) & (log_line_2 > 0))
                logCC_value[i,j] = np.sum(log_line_1[npos] * log_line_2[npos])

        return  CC_value, logCC_value, STD_value

    size_1D = intens1.shape[0]
    centre_p = (0 + size_1D - 1)/2.

    #coordinate = np.array([(x,y) for x in range(size_1D) for y in range(size_1D)])
    #x = coordinate[:,0] - centre_p
    #y = coordinate[:,1] - centre_p
    #r = np.arange(size_1D) - (size_1D-1)/2
    r0 = np.arange(size_1D) - (size_1D-1)/2
    n_mask = np.where(abs(r0) > mask_radius)
    r = r0[n_mask]
    x_matrix = np.zeros([n_angbin, size_1D])


    x0 = np.arange(size_1D) - (size_1D-1)/2
    y0 = np.arange(size_1D) - (size_1D-1)/2

    angbin = 3.1416/(n_angbin)

    x_matrix = np.array([r * np.cos(i*angbin) for i in range(n_angbin)])
    y_matrix = np.array([r * np.sin(i*angbin) for i in range(n_angbin)])



    z1 = intens1
    z2 = intens2
    fp1 = interpolate.interp2d(x0, y0, z1, kind='cubic')
    fp2 = interpolate.interp2d(x0, y0, z2, kind='cubic')

    z_matrix1 = np.array([fp1(x_matrix[i,j],y_matrix[i,j])[0] for i in range(n_angbin) for j in range(len(r))])
    z_matrix1 = z_matrix1.reshape(n_angbin, len(r))
    z_matrix2 = np.array([fp2(x_matrix[i,j],y_matrix[i,j])[0] for i in range(n_angbin) for j in range(len(r))])
    z_matrix2 = z_matrix2.reshape(n_angbin, len(r))

    #CC_value, logCC_value, STD_value = CTY(z_matrix1, z_matrix2)

    return  CTY(z_matrix1, z_matrix2)


# compute using the R language
#norm_corr_ab = sum(a*b) / sqrt(sum(a^2)*sum(b^2)) #equal 0.947
#norm_corr_ac = sum(a*c) / sqrt(sum(a^2)*sum(c^2))


#rec_name = 'cub42_0001'
#foldername = '/gpfs/exfel/exp/SPB/201802/p002160/scratch/yulong/'+rec_name+'/data/corr/'
#foldername = '/gpfs/exfel/exp/SPB/201802/p002160/scratch/ayyerkar/'+rec_name+'/data/corr/'
#filename = foldername + 'output_044.h5'
filename = 'output_100.h5'
h5 = h5py.File(filename,'r')
#h5.keys() #<KeysViewHDF5 ['intens', 'inter_weight', 'likelihood', 'mutual_info', 'occupancies', 'orientations', 'scale']>
intens = np.array(h5['intens'])
orientations = np.array(h5['orientations'])
h5.close()

#intens = np.concatenate((intens1, intens2), axis = 0)


n_models = intens.shape[0]
CC_models = mp.Array(ctypes.c_double, n_models**2)
logCC_models = mp.Array(ctypes.c_double, n_models**2)
aveCC_models = mp.Array(ctypes.c_double, n_models**2)
STD_models = mp.Array(ctypes.c_double, n_models**2)

def mp_worker(rank, indices, CC_models, logCC_models, STD_models, aveCC_models):
    irange = indices[rank::nproc, 0]
    jrange = indices[rank::nproc, 1]

    for i, j in zip(irange, jrange):
        CC_value, logCC_value, STD_value = RotCC_2_images(intens[i], intens[j], 40, 20)

        CC_models[i*n_models + j] = max(CC_value.ravel())
        logCC_models[i*n_models + j] = max(logCC_value.ravel())
        CC_models[j*n_models + i] = max(CC_value.ravel())
        logCC_models[j*n_models + i] = max(logCC_value.ravel())
        STD_models[i*n_models + j] = min(STD_value.ravel())
        STD_models[j*n_models + i] = min(STD_value.ravel())
        aveCC_models[i*n_models + j] = np.mean([max(CC_value[l]) for l in range(CC_value.shape[0])])
        aveCC_models[j*n_models + i] = np.mean([max(CC_value[l]) for l in range(CC_value.shape[0])])

        if rank == 0:
            print("CC (%d, %d):"%(i,j), max(CC_value.ravel()), max(logCC_value.ravel()), min(STD_value.ravel()))

nproc = 16
ind = []
# Get pairs to be processed
for i in range(n_models):
    for j in range(i+1, n_models):
        ind.append([i,j])
ind = np.array(ind)
jobs = [mp.Process(target=mp_worker, args=(rank, ind, CC_models, logCC_models, STD_models, aveCC_models)) for rank in range(nproc)]
[j.start() for j in jobs]
[j.join() for j in jobs]

CC_models = np.frombuffer(CC_models.get_obj()).reshape(n_models, n_models)
logCC_models = np.frombuffer(logCC_models.get_obj()).reshape(n_models, n_models)
aveCC_models = np.frombuffer(aveCC_models.get_obj()).reshape(n_models, n_models)
STD_models = np.frombuffer(STD_models.get_obj()).reshape(n_models, n_models)

CC_models[np.where(CC_models == 0)] = 1
logCC_models[np.where(logCC_models == 0)] = 1
aveCC_models[np.where(aveCC_models == 0)] = 1

savfilename = "Gold_Oct40_Cub42_45_CC_mask20_reg.npz"
np.savez(savfilename, CC_models=CC_models,logCC_models=logCC_models, STD_models=STD_models, aveCC_models=aveCC_models)
