
import numpy as np
import h5py
 
from sklearn.cluster import KMeans
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA


class decomposition_analysis:

    def __init__(self, input_CC_matrix, filename):
        input_CC_matrix = CC_models
        mean_CC = [np.mean(input_CC_matrix[i][np.where(input_CC_matrix[i] < 1)[0]]) for i in range(input_CC_matrix.shape[0])]

        self.input_CC_matrix = input_CC_matrix
        self.mean_CC = mean_CC


        filename = 'CC200/output_050.h5'
        h5 = h5py.File(filename,'r')
        #h5.keys() #<KeysViewHDF5 ['intens', 'inter_weight', 'likelihood', 'mutual_info', 'occupancies', 'orientations', 'scale']>
        self.intens = np.array(h5['intens'])
        self.likelihood = np.array(h5['likelihood'])
        self.occupancies_class = np.array(h5['occupancies'])
        self.orientations = np.array(h5['orientations'])
        self.n_models = intens.shape[0]
        self.size_1D = intens.shape[1]



    def grouping(self, n_clusters, methods=0):
        if (methods == 0):
            embedding = PCA(n_components=3)

        if (methods == 1):
            embedding = Isomap(n_components=3)

        X_t = embedding.fit_transform(self.input_CC_matrix)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_t)
        group_ind_ori = kmeans.labels_
        group_position = kmeans.cluster_centers_
        group_dis = group_position[:,0]**2 + group_position[:,1]**2 + group_position[:,2]**2
        group_ind_sort = np.argsort(group_dis)
        
        #arrange groups with the group-central-distance to the centre
        group_ind = group_ind_ori*0
        for i in range(len(group_ind_sort)):
            gnsi = group_ind_sort[i]
            class_i = np.where(group_ind_ori == gnsi)[0]
            group_ind[class_i] = i

        self.group_ind = group_ind
        self.group_position = group_position

        group_list = []
        for i in range(n_clusters):
            group_list.append(np.where(group_ind == i)[0])

        self.group_list = group_list
        self.n_clusters = n_clusters
        self.X_t = X_t

        occupancies_group = np.zeros([self.occupancies_class.shape[0],15])
        for i in range(n_clusters):
            nkt = np.where(group_ind == i)[0]
            occupancies_group[:,i] = np.sum(self.occupancies_class[:,nkt], axis=1)

        self.occupancies_group = occupancies_group

        return group_ind, group_list

    def create_group_blacklist(self,list_cluster, blacklist_name):

        def create_blacklist(good_modes, blacklist_new):

            modes = np.array([int(self.orientations[i]/180) for i in range(self.orientations.shape[0])] )
            blacklist = modes * 0 + 1

            nb = []
            for i in good_modes:
                n_mode1 = np.where(modes == i)[0]
                lli = self.likelihood[n_mode1]
                npXX = np.where(lli > np.percentile(lli, 0))[0]
                nb.extend(n_mode1[npXX])

            blacklist[nb] = 0
            file1 = open(blacklist_new,"w")
            for i in  blacklist:
                file1.write(str(i)+"\n")

            file1.close() #to change file access modes

        list_class = []
        for i in list_cluster:
            sub = self.group_list[i]
            print(sub)
            list_class.extend(sub)

        filename =  blacklist_name
        good_modes = list_class
        percentage = 0
        create_blacklist(good_modes, filename)


da = decomposition_analysis(CC_models,'CC200/output_050.h5')
group_ind, group_list = da.grouping(15,0)
da.create_group_blacklist([0,2], "hahaha.txt")

X_t = da.X_t
