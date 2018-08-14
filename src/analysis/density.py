import numpy as np
from sklearn.neighbors import KernelDensity
import sys
import os

from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist

src_dir = os.path.abspath(  os.path.join(os.pardir,'src')  )
sys.path=[src_dir]+ sys.path#sys.path[0] = src_dir

from analysis.data_sample import data_bootstrap, data_randomize

def kerneldensity(X,arr_x0,bw,n_rep = 0):
    
    n = len(arr_x0)
    ## fit the density of the real data at each point contained in arr_x0
    arr_rho = np.zeros(n)
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X)
    arr_rho = np.exp(kde.score_samples(arr_x0))


    ## fit the density of the randomized dataset (n_rep repetitions)
    arr_rho_rand = np.zeros((n_rep,n))
    for i_nrep in range(n_rep):
        X_rand = data_randomize(X)
        kde_rand = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X_rand)
        arr_rho_rand[i_nrep,:] = np.exp(kde_rand.score_samples(arr_x0))

    if n_rep > 0:
        return arr_rho,arr_rho_rand
    else:
        return arr_rho

def avg_shortest_dist(X, N_samples):
    '''Calculate the average distance of he nearest neighbor of a randomly sampled point.
    IN:
    - X, arr, shape = samples x features
    - N_samples, int, how many points to sample randomly and get their nn
    OUT:
    - d_mu, float, average distance to nearest neighbor
    '''
    N = len(X)
    kdt = KDTree(X, metric='euclidean')

    list_d = []
    for i in range(N_samples):
        i1 = np.random.choice(N,replace=False,size=1)
        x1 = X[i1,:]
        d = kdt.query(x1, k=2, return_distance=True)[0][0][1]
        list_d += [d]
    return np.mean(list_d)



def cdf_dist_arrid_vec(arr_pd, x_vec, arr_d , n_rand, n_boot):
    arr_p_data = 0.0*arr_d
    arr_p_rand = np.zeros((n_rand,len(arr_d)))
    arr_p_boot = np.zeros((n_boot,len(arr_d)))

    arr_p_dist = cdist([x_vec],arr_pd)[0]
    
    N = len(arr_pd)
    
    for i_d,d in enumerate(arr_d):
        arr_p_data[i_d] = np.sum( arr_p_dist<=d )/float(N)

    for i in range(n_boot):
        arr_pd_boot = data_bootstrap(arr_pd)
        arr_p_dist_boot = cdist([x_vec],arr_pd_boot)[0]
        for i_d,d in enumerate(arr_d):
            arr_p_boot[i,i_d] = np.sum( arr_p_dist_boot<=d )/float(N)

    arr_p_rand = np.zeros((n_rand,len(arr_d)))
    for i in range(n_rand):
        arr_pd_rand = data_randomize(arr_pd)
        arr_p_dist_rand = cdist([x_vec],arr_pd_rand)[0]
        for i_d,d in enumerate(arr_d):
            arr_p_rand[i,i_d] = np.sum( arr_p_dist_rand<=d )/float(N)  
    return arr_p_data, arr_p_rand, arr_p_boot

## pvalue and effect size from density
def rho_pval(arr_rho,arr_rho_rand):
    n_rep_kd,N_c = np.shape(arr_rho_rand)
    arr_rho_rel = arr_rho/arr_rho_rand
    arr_pval = np.sum(arr_rho_rel<1.0,axis=0)/n_rep_kd
    arr_pval = 1.0/n_rep_kd*(arr_pval==0.0) + arr_pval
    return arr_pval

def rho_eff(arr_rho,arr_rho_rand):
    ## simple ratio of the means
    arr_rho_rel =  arr_rho/arr_rho_rand
    arr_eff = np.mean(arr_rho_rel,axis=0)
#   ##cohens d
#     arr_eff = (arr_rho - np.mean(arr_rho_rand,axis=0) )/np.std(arr_rho_rand,axis=0)
    return arr_eff