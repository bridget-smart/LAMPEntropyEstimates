import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from data_processing import *

data_path = 'data/'
results_filepath = 'results/'
checkpoint_filepath = 'checkpoint_data/'

tags = ['lastfm','brightkite','wikispeedia','reuters']
min_occurances = [50,10,10,10]

data = {}

for i in tqdm(range(len(tags))):
    data[tags[i]] = read_data(tags[i],data_path, min_occurances[i], checkpoint_filepath)

lamp_trans = {}
for i in tqdm(range(len(tags))):
    lamp_trans[tags[i]] = get_trans(f"{tag}_transition_matrix.txt")



# to get all values
p_vals_markov = {}
p_vals_markov['p'] = [-i for i in range(25)]

# first order markov
for tag in tags:
    data_mapped, n = data[tag]
    p_vals_markov_list = []
    for i in range(25): 
        A,n = fit_to_markov_sparse(data_mapped, n)
        p= 2**(-i)
        markov_addit_state, pi = get_stat(A.copy(), n, p)
        p_vals_markov_list.append(markov_addit_state)
        np.save( f"{results_filepath}{tag}_{p}_markov_notcc.npy", [[markov_addit_state]])

    p_vals_markov[tag] = p_vals_markov_list


# lamps
# first order markov
p_vals_lamp = {}
for tag in tags:
    data_mapped, n = data[tag]
    p_vals_lamp_list = []
    for i in range(25): 
        p= 2**(-i)
        LAMP_ent2, _ =  get_stat(lamp_trans[tag][0], lamp_trans[tag][1], p)

        p_vals_lamp_list.append(LAMP_ent2)
        np.save( f"{results_filepath}{tag}_{p}_lamp_notcc.npy", [[LAMP_ent2]])

    p_vals_lamp[tag] = p_vals_lamp_list



# plotting
fig, axs = plt.subplots(2,2)
fig.suptitle('Entropy Estimates for Various values of p - first order Markov Chain model')
axes = axs.ravel()

for i in range(len(tags)):
    axes[i].scatter(p_vals_markov['p'],p_vals_markov[tags[i]])
    axes[i].set_xlabel('$\log_2{p}$')
    axes[i].set_title(tags[i])

plt.tight_layout()


# plotting
fig, axs = plt.subplots(2,2)
fig.suptitle('Entropy Estimates for Various values of p - LAMP model')
axes = axs.ravel()

for i in range(len(tags)):
    axes[i].scatter(p_vals_lamp['p'],p_vals_lamp[tags[i]])
    axes[i].set_xlabel('$\log_2{p}$')
    axes[i].set_title(tags[i])

plt.tight_layout()
