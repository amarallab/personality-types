import numpy as np
from sklearn.decomposition import FactorAnalysis

import sys
import os


path_project = '/DRIVE/REPOS/personality-types-shared/'

varimax_dir =  os.path.join( path_project,'src','external') 
sys.path=[varimax_dir]+ sys.path
varimax_dir =  os.path.join( path_project,'src','external','factor_rotation') 
sys.path=[varimax_dir]+ sys.path


import factor_rotation as fr


def rotated_scaled_fa(n_comp, arr_pq,varimax_=True):
    '''Perform factor analysis on a matrix
    IN:
    - n_comp, int, number of latent dimensions
    - arr_pq, arr, shape:  samples (persons) x features (questions)
    - varimax_, bool, whether to perform a varimax rotation (default=True)
    OUT:
    - arr_qd, arr, shape: features x latent-dimension
    - arr_pd, arr, shape: samples x latent dimensions
    '''
    fa = FactorAnalysis(n_comp)
    fa.fit(arr_pq)
    
    arr_pd = fa.transform(arr_pq)
    arr_qd = fa.components_.T

    ## do the varimax-rotation
    if varimax_ == True:
	    arr_dp = np.transpose(arr_pd)
	    
	    L1,T= fr.rotate_factors(arr_qd,'varimax')
	    arr_qd_new = np.dot(arr_qd,T)
	    
	    T_m1 = np.linalg.inv(T)

	    arr_pd_new = np.dot(T_m1,arr_dp)
	    arr_pd_new = np.transpose(arr_pd_new)
	    
	    return arr_qd_new, arr_pd_new
    else:
    	return arr_qd, arr_pd



