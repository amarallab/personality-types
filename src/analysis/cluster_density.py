import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean,cdist

from sklearn.mixture import GaussianMixture



import sys
import os

from analysis.clustering import gmm_labels
from analysis.density import kerneldensity, avg_shortest_dist

# src_dir = os.path.abspath(  os.path.join(os.pardir,'src')  )
# sys.path=[src_dir]+ sys.path#sys.path[0] = src_dir

def gmm_kd(N_c,arr_xy, n_rep, n_rep_kd, bw = -1, N_samples = 1000, arr_w = None ):
    '''
    '''
    dict_result = {}
    ## gmm -fitting
    # arr_cd,arr_cov,arr_weights,likelihood,aic,bic = gmm_multi(N_c,arr_xy,n_rep)
    arr_cd,arr_cov,arr_weights,likelihood,aic,bic,arr_plabels = gmm_labels(N_c,arr_xy,n_rep, arr_w = arr_w)

    dict_result['cluster'] = arr_cd
    dict_result['cov'] = arr_cov
    dict_result['weights'] = arr_weights
    dict_result['L'] = likelihood
    dict_result['AIC'] = aic
    dict_result['BIC'] = bic
    dict_result['labels'] = arr_plabels
    ## kernel-density estimation

    ## default -- we take the average shortest distance
    if bw < 0.0:
        N_samples = 1000
        bw_c = avg_shortest_dist(arr_xy, N_samples)
    else:
        bw_c = 1.0*bw
    dict_result['bw'] = bw
    dict_result['bw_c'] = bw_c

    if n_rep_kd > 0:
        arr_rho,arr_rho_rand = kerneldensity(arr_xy,arr_cd, bw_c ,n_rep = n_rep_kd)
    else:
        arr_rho = []
        arr_rho_rand = []
    dict_result['rho'] = arr_rho
    dict_result['rho_rand'] = arr_rho_rand

    return dict_result