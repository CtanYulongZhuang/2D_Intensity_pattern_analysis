# Preparing training data: original intens, radially fft intens, flattened low frequency training array, plus CC-PCA embedded labels
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.decomposition import PCA

#filename = 'CC01/output_050.h5'
#h5 = h5py.File(filename,'r')
#h5.keys() #<KeysViewHDF5 ['intens', 'inter_weight', 'likelihood', 'mutual_info', 'occupancies', 'orientations', 'scale']>
#intens = np.array(h5['intens'])
#h5.close()

p = np.load("ACC_intens.npz")
intens0 = p['intens']
n_models = intens0.shape[0]
intens = intens0*0
for i in range(n_models):
    intens[i] = intens0[i]/np.sum(intens0[i])

size_1D = intens.shape[1]
centre = int((size_1D-1)/2)
n_angbin = 180


#ang_maps = []
fft_maps = np.zeros([n_models,centre,n_angbin])
for i in range(n_models):
    z0 = intens[i]
    x0,y0 = np.indices((size_1D,size_1D)) ; x0-=centre; y0-=centre

    x0 = np.arange(size_1D) - (size_1D-1)/2
    y0 = np.arange(size_1D) - (size_1D-1)/2
    fp0 = interpolate.interp2d(x0, y0, z0, kind='cubic')


    angbin = 2*np.pi/(n_angbin)
    r = np.arange(centre)
    z_mat0 = []
    x00,y00 = np.indices((size_1D,size_1D)) ; x00-=centre; y00-=centre
    plt.scatter(x00,y00,c = z0,s=1)
    for j in range(n_angbin):
        thita_1 = j*angbin #- 180*angbin
        line_x = r * np.cos(thita_1)
        line_y = r * np.sin(thita_1)
        line_z = [fp0(line_x[k],line_y[k])[0] for k in range(centre)]
        z_mat0 = z_mat0 + line_z
        #plt.scatter(line_x,line_y,s=0.1,alpha=0.5)


    z_mat = np.array(z_mat0)
    z_mat[np.where(z_mat<0)[0]] = 0.0
    z_mat = z_mat.reshape(n_angbin, centre ).T

    fft_z_mat = np.array([[np.abs(np.fft.fft(z_mat[i]))][0] for i in range(centre)])
    fft_maps[i] = fft_z_mat
    print(i)


x,y = np.indices((130,40))
x1 = x.ravel(); y1 = y.ravel()
nn = np.where(y1 < 10+0.25*x1)[0]
nnn = nn.shape[0]

pp_data = np.zeros([n_models, nnn])
for i in range(n_models):
    fftmap_0 = fft_maps[i][:,0::2][20:150,:40]
    fftmap_1 = fftmap_0.ravel()[nn]
    #plt.scatter(x1[nn],y1[nn],fftmap_1)
    pp_data[i] = fftmap_1



p = np.load("ACC_training_CC_matrix.npz")
CC_models = p['CC_models']
n_models = CC_models.shape[0]

embedding = PCA(n_components=3)
X_t = embedding.fit_transform(CC_models)
labels = X_t


#savfilename = "ACC_training_input.npz"
#np.savez(savfilename, fft_maps=fft_maps, pp_data=pp_data, labels=labels)

fnum = h5py.File('ACC_1000_frames.h5', 'w')
fnum['intens'] = intens 
fnum['train_data'] = train_data 
fnum['labels'] = labels 
fnum['radial_fft_intens'] = fft_maps 
fnum.close()
