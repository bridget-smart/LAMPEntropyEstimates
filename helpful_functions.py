"""
Useful functions.
@author: bridgetsmart
@notebook date: 6 Jun 2022
"""

from os import listdir
import numpy as np

def check_function_existence(path, fn):
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


def get_shannon(list_values):
    '''
    Function to get a Shannon Entropy estimate from a distribution 
    using - \sum_x p(x) log p(x).

    These probabilities are calculated using frequencies in the input
    values given in list_values.
    '''
    v, c = np.unique(list_values, return_counts=True)
    n = len(list_values)
    p = c / n

    non_zero = p[p != 0]
    return np.sum(-non_zero*np.log2(non_zero))





