"""
Functions to calculate LAMP Entropies.
@author: bridgetsmart
@notebook date: 6 Jun 2022
"""

import numpy as np, pandas as pd

import math, string, re, pickle, json, time, os, sys, datetime, itertools
from os import listdir
from discreteMarkovChain import markovChain 
import networkx as nx
from sklearn.preprocessing import normalize

from entropy_markov_function import *
from helpful_functions import *
from data_processing import *


def get_trans(fn):
    '''
    Function to read in the transition matrices saved using the code from the original LAMP
    paper.

    Returns the transition matrix and the number of states (n).
    '''

    # read in transition matrix
    with open('transition_matrices/'+fn, 'r') as f:
        lamp_transitions = np.array([y.replace("\n","").split(',') for y in f.readlines()])[1:-1]

    # process transitions
    lamp_transitions = np.array([x[0].split(" ") for x in lamp_transitions])
    lamp_transitions = np.array([[int(y[0]), int(y[1]), float(y[2])] for y in lamp_transitions])
    n_lamp = int(np.max(list(set(lamp_transitions[:,0])) + list(set(lamp_transitions[:,1]))))+1
    lamp_A = np.zeros((n_lamp,n_lamp))
    lamp_A[lamp_transitions[:,0].astype(int), lamp_transitions[:,1].astype(int)] = lamp_transitions[:,2]

    return lamp_A, n_lamp-1


def get_stat(P,n, p=False):
    '''
    This function returns the entropy estimate for a given transition matrix
    which may not be ergodic. It forces ergodicity by adding extra links
    to an artificial state with weight p.

    If p is not defined it is assumed to be equal to 
    half of the smallest existing transition probability.

    Returns entropy estimate and the stationary distribution.
    '''

    g = nx.from_numpy_array(P, create_using = nx.DiGraph)
    d = sorted(nx.strongly_connected_components(g), key=len, reverse=True) # use this function as graph is directed
    print(f'There are originally {len(d)} components with median size {np.median([len(x) for x in d])}.') # gives number of connected components
    
    # now add in extra links
    if p:
        delta = p
    else:
        delta = np.min(P[np.nonzero(P)])/2 #P.shape[0]
        print(f'delta is {delta}')

    to = n # artificial state (true states go from 0 to n.)
    if len(d)>1: # add a link from each connected component to this new state
        for i in range(n):
            P[to,i] += delta
            P[i, to] += delta
    P[to,to] +=delta

    # re normalise P
    P = normalize(P, norm='l1')
    mc = markovChain(P)
    mc.computePi('power')
    
    # normalise pi
    pi= mc.pi/np.sum(mc.pi)

    return get_ent(pi, P, P.shape[0]), pi



def get_stat_largest_cc(P, tag, results_filepath):
    '''
    This function returns the entropy estimate for the largest connected component
    of a given transition matrix which may not be ergodic. 

    Returns entropy estimate and the stationary distribution.
    '''

    g = nx.from_numpy_array(P, create_using = nx.DiGraph)
    d = sorted(nx.strongly_connected_components(g), key=len, reverse=True) # use this function as graph is directed
    print(f'There are originally {len(d)} components with median size {np.median([len(x) for x in d])}.') # gives number of connected components
    print(f'Originally there are {P.shape[0]} nodes')

    orig = P.shape[0]
    P = P[:,list(d[0])]
    P = P[list(d[0]), :] # select only largest cc
    print(f'Now there are {P.shape[0]} nodes.')
    new = P.shape[0]

    # saves the number of states removed
    print(f'{[orig-new]} states have been removed.')
    np.save( results_filepath+f"{tag}_nodes_removed_using_largest_cc.npy", [[orig-new]])

    # re normalise
    P = normalize(P, norm='l1')
    mc = markovChain(P)
    mc.computePi('power')
    
    # normalise pi
    pi= mc.pi/np.sum(mc.pi)

    return get_ent(pi, P, P.shape[0]), pi