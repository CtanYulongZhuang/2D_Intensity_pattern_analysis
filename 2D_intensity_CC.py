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




def RotCC_2_images(intens1, intens2, n_angbin):

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
            line_1_z = line_1_z0/np.sum(line_1_z0) 
            for j in range(n_angbin): 
                line_2_z0 = z_matrix2[j]
                line_2_z = line_2_z0/np.sum(line_2_z0) 
                
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
    r = np.arange(size_1D) - (size_1D-1)/2
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

    z_matrix1 = np.array([fp1(x_matrix[i,j],y_matrix[i,j])[0] for i in range(n_angbin) for j in range(size_1D)])
    z_matrix1 = z_matrix1.reshape(n_angbin, size_1D)
    z_matrix2 = np.array([fp2(x_matrix[i,j],y_matrix[i,j])[0] for i in range(n_angbin) for j in range(size_1D)])
    z_matrix2 = z_matrix2.reshape(n_angbin, size_1D)

    #CC_value, logCC_value, STD_value = CTY(z_matrix1, z_matrix2)

    return  CTY(z_matrix1, z_matrix2)

    
# compute using the R language
#norm_corr_ab = sum(a*b) / sqrt(sum(a^2)*sum(b^2)) #equal 0.947
#norm_corr_ac = sum(a*c) / sqrt(sum(a^2)*sum(c^2))


rec_name = 'cub42_0001'
#foldername = '/gpfs/exfel/exp/SPB/201802/p002160/scratch/yulong/'+rec_name+'/data/corr/'
foldername = '/gpfs/exfel/exp/SPB/201802/p002160/scratch/ayyerkar/'+rec_name+'/data/corr/'
filename = foldername + 'output_044.h5'
h5 = h5py.File(filename,'r')
#h5.keys() #<KeysViewHDF5 ['intens', 'inter_weight', 'likelihood', 'mutual_info', 'occupancies', 'orientations', 'scale']>
intens2 = np.array(h5['intens'])
orientations2 = np.array(h5['orientations'])
h5.close()

#intens = np.concatenate((intens1, intens2), axis = 0)


n_models = intens.shape[0]
CC_models = np.zeros([n_models,n_models])+1.0
STD_models = np.zeros([n_models,n_models])
logCC_models = np.zeros([n_models,n_models])

for i in range(n_models):
    z1 = intens[i]
    print("main image = ", i)

    for j in range(i+1, n_models):
        z2 = intens[j]

        CC_value, logCC_value, STD_value = RotCC_2_images(z1, z2, 20)
        CC_models[i,j] = max(CC_value.ravel())
        logCC_models[i,j] = max(logCC_value.ravel())
        CC_models[j,i] = max(CC_value.ravel())
        logCC_models[j,i] = max(logCC_value.ravel())
        STD_models[i,j] = min(STD_value.ravel())
        STD_models[j,i] = min(STD_value.ravel())
        print("CC image = ", j, ':', max(CC_value.ravel()), max(logCC_value.ravel()), min(STD_value.ravel()))

savfilename = rec_name+"_CC_analysis.npz"
np.savez(savfilename, CC_models=CC_models,logCC_models=logCC_models, STD_models=STD_models)




























































































#
