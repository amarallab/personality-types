import numpy as np
from scipy.linalg import svd
import random

def get_subset(arr_p_d_,N_p_c_):
    '''return a random subsample of a dataset
    IN:
    - arr_p_d_, arr, shape: samples x features
    - N_p_c_, int, size of subsample. if N_p_c_ <0 or >#samples, we return the original array
    Out:
    - arr_p_d_subsample: arr, shape N_p_c_ x features 
    '''
    N_P_ = len(arr_p_d_[:,0])
    if N_p_c_ >= N_P_ or N_p_c_<0:
        return arr_p_d_
    else:
        return arr_p_d_[np.random.choice(N_P_,size=N_p_c_, replace=False),:]

def data_bootstrap(arr_p_d_c_):
    '''return a bootstrap-dataset of the same size, i.e. draw data points with replacement
    IN:
    - arr_p_d_c_, arr, shape: samples x features
    OUT:
    - arr_p_d_c_boot_, arr, shape: samples x features
    '''
    N_d_ = len(arr_p_d_c_[0,:])
    N_p_ = len(arr_p_d_c_[:,0])

    arr_p_d_c_boot_ = arr_p_d_c_[ np.random.choice( N_p_,size=N_p_,replace=True)  ,:]

    return arr_p_d_c_boot_

def data_randomize(arr_p_d_c_,m_replace = False):
	'''Randomize a dataset (with or without replacement) by all entries in a given dimension. 
    (i.e. we randomize but keep the marginal distributions const)
    IN:
    - arr_p_d_c_, arr, shape: samples x features
    - m_replace, randomize with or without replacement (default=False)
    OUT:
    - arr_p_d_c_rand_, arr, shape: samples x features; randomized version of original dataset
	'''
	N_d_ = len(arr_p_d_c_[0,:])
	N_p_ = len(arr_p_d_c_[:,0])	
	arr_p_d_c_rand_ = 0.0*arr_p_d_c_
	for i_d in range(N_d_):
		p_ = 1.0*arr_p_d_c_[:,i_d]
		if m_replace == False:
			np.random.shuffle(p_)
		else:
			p_ = np.random.choice(p_,size=len(p_),replace=True)
		arr_p_d_c_rand_[:,i_d] = p_
	return arr_p_d_c_rand_