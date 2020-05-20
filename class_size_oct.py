import sys
import multiprocessing as mp
import ctypes
import numpy as np
import h5py
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import interpolate


sys.path.append('utils/py_src/')
#import detector, reademc



def get_angavg(arr):
    angavg = np.zeros(360)
    np.add.at(angavg, intang[radsel], arr[radsel])
    return angavg / angcounts

def get_line(marr, angle):
    tcoords = np.dot([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], zcoords).T + [345//2, 355//2]
    dmap = ndimage.map_coordinates(marr.data, tcoords.T, order=1).reshape(170,21)
    mmap = ndimage.map_coordinates(marr.mask, tcoords.T, order=0).reshape(170,21)
    dcounts = (~mmap).sum(1)
    dcounts[dcounts==0] = 1
    dmap[mmap] = 0
    return dmap.sum(1) / dcounts, mmap.all(1)


# Get detector and emc file
#det = detector.Detector(det_fname, mask_flag=True)
#emc = reademc.EMCReader(emc_fname, det)


filename = 'CC200/output_050.h5'
h5 = h5py.File(filename,'r')
#h5.keys() #<KeysViewHDF5 ['intens', 'inter_weight', 'likelihood', 'mutual_info', 'occupancies', 'orientations', 'scale']>
intens = np.array(h5['intens'])
likelihood = np.array(h5['likelihood'])
occupancies_class = np.array(h5['occupancies'])
orientations = np.array(h5['orientations'])
n_models = intens.shape[0]
size_1D = intens.shape[1]



# Initialize angular averaging
x, y = np.indices((441,441)); x -= 220; y -= 220
intang = ((np.arctan2(y, x) + np.pi)*360/2./np.pi).astype('i4')
intang[intang==360] = 0
intrad = np.sqrt(x*x + y*y).astype('i4')
radsel = (intrad > 50)
angcounts = np.zeros(360)
np.add.at(angcounts, intang[radsel], 1) #Counts how many picels per angle (later will be used to average)
angcounts[angcounts==0] = 1
#zx, zy = np.indices((170,21), dtype='f8')
#zy -= 10.
#zcoords = np.array([zx.ravel(), zy.ravel()])
print('Initialized angular averaging')

# Get angle for every frame via orientation of each class
angavgs = np.array([get_angavg(i) for i in intens])
maxang = np.argmax(angavgs, axis=1)
#pang = (angmax*2 - maxang[modes])%360
# ---- For octahedra with two streaks
shift = (angavgs[np.arange(200),(maxang+71)%360] > angavgs[np.arange(200), (maxang-71+360)%360])*71
#pang = (angmax*2 - (maxang+shift)[modes])%360
#pang2 = (angmax*2 - (maxang+shift-71)[modes])%360
pang1 = ((maxang+shift))%360
pang2 = ((maxang+shift-71))%360
print('Calculated angle for each frame')



#plt.figure(figsize=(12,8))
#for i in range(50):
#    plt.subplot(5, 10, i-0+1)
#    intens_m = intens[i].ravel()+0.00001
#    intens_m = intens_m.reshape(size_1D, size_1D)
#    plt.imshow(np.log10(intens_m), vmin=-2)
#    line_x = - r * np.sin(pang_pi1[i])
#    line_y = - r * np.cos(pang_pi1[i])
#    line_x2 = - r * np.sin(pang_pi2[i])
#    line_y2 = - r * np.cos(pang_pi2[i])
#    plt.plot(line_x+220,line_y+220)
#    plt.plot(line_x2+220,line_y2+220)
#    plt.text(0, 100, '  No.:'+str( maxang[i]), fontsize=8, color ='Cyan')



# Generate sinc function models
radsamples = np.arange(10,50,0.1)
q = 2*np.sin(0.5*np.arctan(np.arange(221)*0.2/705)) * 6/12.3984 * 10
sincmodels = np.array([np.sinc(q*r)**2 for r in radsamples])
print('Generated models')



pang_pi = (maxang/360*2*np.pi)
pang_pi1 = (pang1/360*2*np.pi)
pang_pi2 = (pang2/360*2*np.pi)
r = np.arange(221)

psizes1 = pang_pi1*0
psizes2 = pang_pi2*0

for i in range(200):
    z0 = intens[i]
    x0 = np.arange(441)-220
    y0 = np.arange(441)-220
    fp0 = interpolate.interp2d(x0, y0, z0, kind='cubic')
    line_x1 = - r * np.sin(pang_pi1[i])
    line_y1 = - r * np.cos(pang_pi1[i])
    line_x2 = - r * np.sin(pang_pi2[i])
    line_y2 = - r * np.cos(pang_pi2[i])

    x_matrix = np.vstack((line_x1, line_x2))
    y_matrix = np.vstack((line_y1, line_y2))

    z_matrix0 = np.array([fp0(x_matrix[i,j],y_matrix[i,j])[0] for i in range(2) for j in range(len(r))])
    z_matrix = z_matrix0.reshape(2, len(r))

    corrs = np.array([np.corrcoef(z_matrix[0][51:]/np.sum(z_matrix[0][51:]), sincmodels[k][51:]/np.sum(sincmodels[k][51:]))[0,1:] for k in range(400)])
    psizes1[i] = radsamples[corrs.argmax()]
    corrs = np.array([np.corrcoef(z_matrix[1][51:]/np.sum(z_matrix[1][51:]), sincmodels[k][51:]/np.sum(sincmodels[k][51:]))[0,1:] for k in range(400)])
    psizes2[i] = radsamples[corrs.argmax()]
    #
savfilename = "CC200/Class_size.npz"
np.savez(savfilename, psizes1=psizes1,psizes2=psizes2)
