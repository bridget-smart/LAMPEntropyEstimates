"""
Functions to get estimates from LAMP Entropy Paper.
@author: bridgetsmart
@notebook date: 6 Jun 2022
"""

from data_processing import *
from tqdm.notebook import tqdm


data_path = 'data/'
results_filepath = 'results/'
checkpoint_filepath = 'checkpoint_data/'


tags = ['lastfm','brightkite','wikispeedia','reuters']
min_occurances = [50,10,10,10]
p_markov = [2**(-15),2**(-15),2**(-10),2**(-15)] # choice of weight for artifically added edges chosen from simulations for first-order markov approach
p_lamp = [2**(-20),2**(-20),2**(-20),2**(-20)]# choice of weight for artifically added edges chosen from simulations for LAMP approach

for i in tqdm(range(len(tags))):
    run_estimates(tags[i], results_filepath, data_path, min_occurances[i], checkpoint_filepath, p_markov[i], p_lamp[i])

