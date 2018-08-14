import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean,cdist

from sklearn.mixture import GaussianMixture

import sys
import os

# src_dir = os.path.abspath(  os.path.join(os.pardir,'src')  )
# sys.path=[src_dir]+ sys.path#sys.path[0] = src_dir

def kmeans_clusters(N_c,arr_xy, cores = 1, ninit = 10, maxiter = 300 ):
    '''k-means clustering 
    IN:
    - N_c: number of clusters, int
    - arr_xy: array to be clustered (shape: samples x features)

    - cores: number of cores to use, int (default 1)
    - n_init: number of initial conditions for search, int (default: 10), more is better
    - maxiter: maximum number of iterations
    OUT:
    - cluster centers, array of floats (shape: N_c x features)
    '''

    km = KMeans(n_clusters=N_c, n_init=ninit, max_iter=maxiter,n_jobs=cores)
    km.fit(arr_xy)
    
    return km.cluster_centers_

def kmeans_clusters_dist(N_c,arr_xy, cores = 1, ninit = 10, maxiter = 300 ):
    '''k-means clustering 
    IN:
    - N_c: number of clusters, int
    - arr_xy: array to be clustered (shape: samples x features)

    - cores: number of cores to use, int (default 1)
    - n_init: number of initial conditions for search, int (default: 10), more is better
    - maxiter: maximum number of iterations
    OUT:
    - cluster centers, array of floats (shape: N_c x features)
    - total distance function
    '''

    km = KMeans(n_clusters=N_c, n_init=ninit, max_iter=maxiter,n_jobs=cores)
    km.fit(arr_xy)
    
    return km.cluster_centers_, km.inertia_ 

def gmm_clusters(N_c,arr_xy,n_rep=10, n_init = 1):
    '''calculate the Gaussian Mixture centers.
    We do n_rep different runs of the model (with n_init initialization): 
    - we save the aic and bic for each run 
    - we return the cluster center only for the one with the best likelihood from all runs
    For this we run n_rep times the model with (tyically n_init=1) initializatins

    IN:
    - N_c, int, number of components
    - arr_xy: array to be clustered (shape: samples x features)
    - n_rep, int: number of different runs (with n_init initializations) for which we report the aic/bic;
        the cluster centers will be the best from all n_rep runs (default=10)
    - n_init, int: number of initial conditions for search, int (default: 1),that gives a result (an aic and a bic)
    OUT:
    - arr_cd_tmp, arr, shape=N_c x features
    - arr_aic, arr, shape = n_rep; AIC-values for the n_rep runs 
    - arr_bic, arr, shape = n_rep; BIC-values for the n_rep runs 
    - arr_weights, arr, shape = N_c, weights for each cluster


    '''
    arr_aic = np.zeros(n_rep)
    arr_bic = np.zeros(n_rep)
    arr_weights = np.zeros(N_c)
    likelihood = -np.inf
    N,n_comp = np.shape(arr_xy)
    arr_cd_tmp = np.zeros((N_c,n_comp))
    for i_nrep in range(n_rep):
        gmm = GaussianMixture(N_c,n_init=n_init)
        gmm.fit(arr_xy)
        aic = gmm.aic(arr_xy)
        arr_aic[i_nrep] = aic
        bic = gmm.bic(arr_xy)
        arr_bic[i_nrep] = bic
        p_ = gmm.lower_bound_
        if p_ >likelihood:
            likelihood = p_
            arr_cd_tmp = gmm.means_
            arr_weights = gmm.weights_
    return arr_cd_tmp, arr_aic, arr_bic, arr_weights

def gmm_single(N_c,arr_xy,n_rep = 1):
    gmm = GaussianMixture(N_c)
    ## fit the model
    gmm.fit(arr_xy)

    ## get the output
    # the cluster-centers and covariances
    arr_cd = gmm.means_
    arr_cov = gmm.covariances_
    # the weights of the clusters
    arr_weights = gmm.weights_
    # lower bound from em
    likelihood = gmm.lower_bound_
    # aic and bic on data
    aic = gmm.aic(arr_xy)
    bic = gmm.bic(arr_xy)
    return arr_cd,arr_cov,arr_weights,likelihood,aic,bic

def gmm_multi(N_c,arr_xy,n_rep):
    likelihood = -np.inf
    arr_cd = np.zeros(( N_c,np.shape(arr_xy)[1] ))
    aic = 0.0
    bic = 0.0
    arr_weights = np.zeros(N_c)
    arr_cov = 0.0

    for i_nrep in range(n_rep):
        arr_cd_tmp,arr_cov_tmp,arr_weights_tmp,likelihood_tmp,aic_tmp,bic_tmp = gmm_single(N_c,arr_xy)
        if likelihood_tmp>likelihood:
            likelihood = likelihood_tmp
            arr_cd = arr_cd_tmp
            aic = aic_tmp
            bic = bic_tmp
            arr_weights = arr_weights_tmp
            arr_cov = arr_cov_tmp

    return arr_cd,arr_cov,arr_weights,likelihood,aic,bic

def gmm_labels(N_c,arr_xy,n_rep, arr_w = None):
    ## fit the GMM
    likelihood = -np.inf
    for i_nrep in range(n_rep):
        ## we can select the initial parameters for the weights via arr_w
        if arr_w == None:
            gmm = GaussianMixture(N_c)
        else:
            arr_w = arr_w/np.sum(arr_w)
            gmm = GaussianMixture(N_c, weights_init = arr_w)
        gmm.fit(arr_xy)
        likelihood_tmp = gmm.lower_bound_
        if likelihood_tmp > likelihood:
            likelihood = likelihood_tmp
            arr_cd,arr_cov,arr_weights,likelihood,aic,bic,arr_plabels = gmm_get_params(gmm,arr_xy)
    return arr_cd,arr_cov,arr_weights,likelihood,aic,bic,arr_plabels

def gmm_get_params(gmm,arr_xy):
    arr_cd = gmm.means_
    arr_cov = gmm.covariances_
    arr_weights = gmm.weights_
    likelihood = gmm.lower_bound_
    aic = gmm.aic(arr_xy)
    bic = gmm.bic(arr_xy)
    arr_plabels = gmm.predict_proba(arr_xy)
    return arr_cd,arr_cov,arr_weights,likelihood,aic,bic,arr_plabels