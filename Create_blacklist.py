import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm

def create_blacklist(bad_modes, filename, percentage, blacklist_new, blacklist_old):
    #filename =  'output_100.h5'
    h5 = h5py.File(filename,'r')
    #h5.keys() #<KeysViewHDF5 ['intens', 'inter_weight', 'likelihood', 'mutual_info', 'occupancies', 'orientations', 'scale']>
    intens = np.array(h5['intens'])
    orientations = np.array(h5['orientations'])
    likelihood = np.array(h5['likelihood'])
    h5.close()

    n_models = intens.shape[0]
    size_1D = intens.shape[1]
    modes = np.array([int(orientations[i]/180) for i in range(orientations.shape[0])] )


    try:
        file0 = open(blacklist_old,"r")
        Lines = file0.readlines()
        file0.close() #to change file access modes
        blacklist = np.array([int(i.split()[0]) for i in Lines])

    except:
        blacklist = modes * 0


    nb = []
    #bad_modes = [9,10,35,38,41]
    for i in bad_modes:
        n_mode1 = np.where(modes == i)[0]
        lli = likelihood[n_mode1]
        npXX = np.where(lli > np.percentile(lli, percentage))[0]
        #plt.hist(likelihood[n_mode1], bins=100, range=(-10000,-10))
        #plt.hist(likelihood[n_mode1[np35]], bins=100, range=(-10000,-10))
        nb.extend(n_mode1[npXX])

    #plt.hist(likelihood[nb], bins=100, range=(-10000,-10))


    blacklist[nb] = 1
    file1 = open(blacklist_new,"w")
    for i in  blacklist:
        file1.write(str(i)+"\n")

    file1.close() #to change file access modes





filename =  'output_100.h5'
bad_modes = [9,10,35,38,41]
percentage = 35
blacklist_old = 'blacklist_cc_00.txt'
blacklist_new = 'blacklist_cc_01.txt'
create_blacklist(bad_modes, filename, percentagee, blacklist_new, blacklist_old)


