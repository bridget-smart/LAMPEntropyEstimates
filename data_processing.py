"""
Functions to process data.
@author: bridgetsmart
@notebook date: 6 Jun 2022
"""

import numpy as np
import pandas as pd
import pickle
from itertools import groupby

from helpful_functions import *
from lamp_specific_ent_functions import *
from entropy_markov_function import *


def map_data(states, data):
    '''
    Function to map data (passes as an array of lists or any 2D iterable 
    data structure, with some pool of n possible values (states) which 
    is passed as a list which may contain duplicates to an alphabet of [0,n-1].

    Returns the mapped data and the size of the alphabet n.
    '''
    # number of states

    # first, get the alphabet
    alphabet = list(set(states))
    try:
        alphabet.sort()
    except:
        pass
    n = len(alphabet)
    # now, map the data to the alphabet
    data_mapped = np.array([[alphabet.index(y) for y in x] for x in data], dtype=object)

    return data_mapped, n

def process_l(list_of_values, to_replace, replacement_value = None):
    '''
    Function to remove a set of values from a list and replace them with the replacement value.
    If replacement value is None, then the elements will be removed.

    to_replace is a set.
    '''

    return [val_ret(x, to_replace, replacement_value) for x in list_of_values if val_ret(x, to_replace, replacement_value) is not None]

def val_ret(y, to_replace, replacement_value): 
    '''
    Function to return y if y is not in the set to_replace.
    '''
    if y in to_replace:
        return replacement_value
    else:
        return y 



def filter_data(data_mapped, data_flag, min_occurances,checkpoint_filepath):
    '''
    Given some data and a flag, this function checks for the existance of the data which has been mapped,
    and if the file exists it reads it in to avoid double calculating,
    if it doesn't yet exist it flattens and maps the data_mapped to an alphabet of [0,n-1].

    It returns the mapped data and the size of the alphabet (n).
    '''

    # file doesn't exist
    if not check_function_existence(checkpoint_filepath,f'{data_flag}_filtered_mapped_data.pkl'):
        v,c = np.unique(flatten(data_mapped), return_counts=True)
        replacement_val = 2*np.max(v) # we set this to be outside the vocab size so it is an additional character
        to_replace = set(v[c<=min_occurances]) # we replace values which occur less than 10 times for all datasets except for  the lastfm dataset where min_occurances = 50.

        for i in range(len(data_mapped)):
            data_mapped[i] = process_l(data_mapped[i], to_replace, replacement_val)
            

        s = flatten(data_mapped)

        data_mapped, n = map_data(s, data_mapped) # reset the alphabet to be [0,n].

        # save the mapped data
        with open(f'{checkpoint_filepath}{data_flag}_filtered_mapped_data.pkl','wb') as f:
            pickle.dump(data_mapped, f)


    # file does exist - just load in
    else:
        print('loading')
        with open(f'{checkpoint_filepath}{data_flag}_filtered_mapped_data.pkl','rb') as f:
            data_mapped = pickle.load(f)
            n = len(np.unique([x for y in data_mapped for x in y]))

    return data_mapped, n


# Dataset specific functions

def read_brightkite(data_path, min_occurances, checkpoint_filepath):
    '''
    Function to read in and process the brightkite dataset.
    Process:
    - filtering to remove values which appear less than 10 times
    - mapping to [0,n-1] dictionary

    Returns the processed data in a numpy array.
    Each element in the array represents a single user, 
    with items representing the locations
    which the user has checked into.
    '''

    if not check_function_existence(checkpoint_filepath,'brightkite_filtered_mapped_data.pkl'):
        data_bk = pd.read_csv(f'{data_path}loc-brightkite_totalCheckins.txt', sep='\t', header=None)
        data_bk.columns = ['users','time','lat','lon','loc_id']

        paths_bk = []
        for l, df_ in data_bk.groupby('users').loc_id:
            paths_bk.append([x for x in df_.values])

        data_mapped, n= map_data(flatten(paths_bk), paths_bk)

        # filter values
        data_mapped, n= filter_data(data_mapped, 'brightkite', min_occurances,checkpoint_filepath)

    else:
        with open(f'{checkpoint_filepath}brightkite_filtered_mapped_data.pkl','rb') as f:
            data_mapped = pickle.load(f)
            n = len(np.unique([x for y in data_mapped for x in y]))

    return data_mapped, n



def read_lastfm(data_path, min_occurances, checkpoint_filepath):
    if not check_function_existence(checkpoint_filepath,'lastfm_filtered_mapped_data.pkl'):
        data_lfm = pd.read_csv(f'{data_path}lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv', sep='\t',
                on_bad_lines = 'skip', header=None)
        data_lfm.columns = ['userid','timestamp','musicbrainz_artist_id','artist_name','musicbrainz_track_id','track_name']

        paths_lfm = []
        for user, artists in data_lfm.groupby('userid').artist_name:
            paths_lfm.append([x for x in artists.values])

        data_mapped, n= map_data(flatten(paths_lfm), paths_lfm)


        # filter values
        data_mapped, n= filter_data(data_mapped, 'lastfm', min_occurances,checkpoint_filepath)

    else:
        with open(f'{checkpoint_filepath}lastfm_filtered_mapped_data.pkl','rb') as f:
            data_mapped = pickle.load(f)
            n = len(np.unique([x for y in data_mapped for x in y]))

    return data_mapped, n


def read_wikispeedia(data_path, min_occurances, checkpoint_filepath):
    if not check_function_existence(checkpoint_filepath,'wikispeedia_filtered_mapped_data.pkl'):
        with open('data/wiki_paths.txt', 'r') as f:
            wiki_paths = [x.replace("\n","").split(";") for x in f.readlines()]

        data_mapped, n= map_data(flatten(wiki_paths), wiki_paths)


        # filter values
        data_mapped, n= filter_data(data_mapped, 'wikispeedia', min_occurances,checkpoint_filepath)

    else:
        with open(f'{checkpoint_filepath}wikispeedia_filtered_mapped_data.pkl','rb') as f:
            data_mapped = pickle.load(f)
            n = len(np.unique([x for y in data_mapped for x in y]))

    return data_mapped, n


def read_reuters(data_path, min_occurances, checkpoint_filepath):
    if not check_function_existence(checkpoint_filepath,'reuters_filtered_mapped_data.pkl'):
        data_path_r = data_path+'reuters/training/'
        files_train = [data_path_r+f for f in listdir(data_path_r)]

        data_path_r_test = data_path+'reuters/test/'
        files_test = [data_path_r_test+f for f in listdir(data_path_r_test)]

        files = files_train + files_test

        data_re = []
        for fn in files:
            with open(fn, 'r',  errors='ignore') as f:
                temp = [x.replace("\n","").lower() for x in f.readlines()]
                data_re.append(" ".join(temp).replace("  ",""))

        data_re = [x.split(" ") for x in data_re]

        data_mapped, n= map_data(flatten(data_re), data_re)


        # filter values
        data_mapped, n= filter_data(data_mapped, 'reuters', min_occurances,checkpoint_filepath)
    

    else:
        with open(f'{checkpoint_filepath}reuters_filtered_mapped_data.pkl','rb') as f:
            data_mapped = pickle.load(f)
            n = len(np.unique([x for y in data_mapped for x in y]))
    return data_mapped, n

def read_data(tag, data_path, min_occurances, checkpoint_filepath):
    if tag == 'lastfm':
        data_mapped, n = read_lastfm(data_path, min_occurances, checkpoint_filepath)
    elif tag == 'brightkite':
        data_mapped, n = read_brightkite(data_path, min_occurances, checkpoint_filepath)
    elif tag == 'wikispeedia':
        data_mapped, n = read_wikispeedia(data_path, min_occurances, checkpoint_filepath)
    elif tag == 'reuters':
        data_mapped, n = read_reuters(data_path, min_occurances, checkpoint_filepath)
    else:
        print('invalid tag')
        return 0

    return data_mapped, n

def run_estimates(tag, results_filepath, data_path, min_occurances, checkpoint_filepath, p_markov, p_lamp):

    lamp_transitions, n_lamp = get_trans(f"{tag}_transition_matrix.txt")

    # largest cc estimate
    LAMP_ent, _ =  get_stat_largest_cc(lamp_transitions.copy(), tag, results_filepath)
    np.save(f"{results_filepath}{tag}_lamp_est_only_lcc.npy", [[LAMP_ent]])

    # adding additional state
    LAMP_ent2, _ =  get_stat(lamp_transitions.copy(), n_lamp, p_lamp)
    np.save( f"{results_filepath}{tag}_{p_lamp}_lamp_est_additional_state.npy", [[LAMP_ent2]])

    # read data
    data_mapped, n = read_data(tag,data_path, min_occurances, checkpoint_filepath)

    ## SHANNON ESTIMATORS
    seq = [y for x in data_mapped for y in x]
    shann_sequence = get_shannon(seq)
    np.save( f"{results_filepath}{tag}_shann_sequence_level.npy", [[shann_sequence]])


    shann_av_path = np.mean([get_shannon(x) for x in data_mapped])
    np.save( f"{results_filepath}{tag}_shann_av_path.npy", [[shann_av_path]])


    ## MARKOV ESTIMATORS 
    A,n = fit_to_markov_sparse(data_mapped, n)
    fo_largestcc, pi = get_stat_largest_cc(A.copy(), tag, results_filepath)
    np.save( f"{results_filepath}{tag}_markovfirstorder.npy", [[fo_largestcc]])

    shann_markov = np.sum(-pi*np.log(pi))
    np.save( f"{results_filepath}{tag}_shann_markov_stationary.npy", [[shann_markov]])

    A,n = fit_to_markov_sparse(data_mapped, n)
    markov_addit_state, pi = get_stat(A.copy(), n, p_markov)
    np.save( f"{results_filepath}{tag}_{p_markov}_markov_notcc.npy", [[markov_addit_state]])
    
    return 0