"""
This script contains functions used to calculate the entropy of transition matrices 
and fit first order Markov models to sequence data.
@author: bridgetsmart
@notebook date: 6 Jun 2022
"""

import numpy as np, pandas as pd
from scipy.sparse import dok_matrix
import math, string, re, pickle, json, time, os, sys, datetime, itertools
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import eigs
from os import listdir

from other_functions import *


def fit_to_markov_sparse(data_mapped, n):
    '''
    All states are mapped to a 0,n-1 alphabet,
    a first-order Markov Chain is fit to the data,
    and a sparse matrix is used to store the 
    transition probabilities.

    It returns the transition matrix and the number of states. 
    The returned transition matrix contains an additional state
    which is used to force ergodicity later.
    '''

    # get all one step transitions
    one_step_trans = [(x[i], x[i+1]) for x in data_mapped for i in range(len(x)-1)]

    # preallocate transition matrix + a state used when we force ergodicity
    P = dok_matrix((n+1, n+1), dtype=np.float64) # using 64 bit precision

    # get unique transitions and counts of these transitions
    inds, counts = np.unique(one_step_trans, axis = 0, return_counts = True)

    # allocate these values to their respective position within the P matrix
    # each element i,j now contains the number of transitions from i to j in the data.
    P[inds[:, 0], inds[:, 1]] = counts

    # Normalise the P matrix to change these counts to transiiton probabilities.
    P = normalize(P, norm='l1', axis = 1) # use sklearn to normalise
    
    return P, n

def fit_to_markov(data_mapped, n):

    '''
    All states are mapped to a 0,n-1 alphabet,
    a first-order Markov Chain is fit to the data,
    and a dense matrix is used to store the 
    transition probabilities.

    It returns the transition matrix and the number of states. 
    The returned transition matrix contains an additional state
    which is used to force ergodicity later.
    '''

    # get all one step transitions   
    one_step_trans = [(x[i], x[i+1]) for x in data_mapped for i in range(len(x)-1)]

    # preallocate transition matrix + a state used when we force ergodicity
    P = np.ones((n+1, n+1)) 

    # get unique transitions and counts of these transitions
    inds, counts = np.unique(one_step_trans, axis = 0, return_counts = True)

    # allocate these values to their respective position within the P matrix
    # each element i,j now contains the number of transitions from i to j in the data.
    P[inds[:, 0], inds[:, 1]] += counts

    # Normalise the P matrix to change these counts to transiiton probabilities.
    P = normalize(P, norm='l1', axis = 1) # use sklearn to normalise

    return P, n

def get_ent(pi,A,n):
    '''
    This function returns the entropy estimate using the 
    stationary distribution and the transition matrix of a Markov Chain.
    '''

    ent = 0
    for i in range(n):
        for j in range(n):
            if A[i,j] ==0:
                pass
            else:
                if pi[i] == 0:
                    pass
                else:
                    ent += - pi[i]*A[i,j]*math.log(A[i,j])
    return ent    
    

def get_stationary_sparse(A):
    '''
    Given a sparse transition matrix, A, this function returns the stationary
    distribution using the eigenvectors.
    '''
    e_vals, eig_vecs = eigs(A.T) # gets left e vectors
    l = np.where(abs(e_vals-1) == np.min(abs(e_vals-1)))[0][0]

    # get real component
    v = eig_vecs[:,l].real

    return v/sum(v)

    # function to get entropy estimates from LAMP fit


def get_stationary(A):
    '''
    Given a dense transition matrix, A, this function returns the stationary
    distribution using the eigenvectors.
    '''
    e_vals, eig_vecs = np.linalg.eig(A.T) # gets left e vectors
    l = np.where(abs(e_vals-1) == np.min(abs(e_vals-1)))[0][0]
    
    # get real component
    v = eig_vecs[:,l].real

    return v/sum(v)
