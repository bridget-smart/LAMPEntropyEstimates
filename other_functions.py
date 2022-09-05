"""
Useful functions.
@author: bridgetsmart
@notebook date: 6 Jun 2022
"""

from os import listdir
import numpy as np

def check_function_existance(path, fn):
    '''
    Function to check if file exists at a given path.
    '''
    return fn in listdir(path)


def flatten(data):
    '''
    Function to flatten iterable datatypes.
    '''

    if type(data[0]) == list:
        return [y for x in data for y in x]
    if type(data) == type(np.zeros((1,1))):
        return [y for x in data for y in x]
    else:
        return data


def get_shannon(seq):
    '''
    Function to get a Shannon Entropy estimate from a stationary distribution 
    using - \sum_x p(x) log p(x).
    '''
    v, c = np.unique(seq, return_counts=True)
    n = len(seq)
    p = c / n

    non_zero = p[p != 0]
    return np.sum(-non_zero*np.log2(non_zero))





