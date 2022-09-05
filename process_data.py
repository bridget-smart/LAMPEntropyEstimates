import numpy as np
import pandas as pd
from data_processing import *



import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-nu', type=int)
    args = parser.parse_args()

    job_number = args.job_nu
    print(job_number)

    data_path = 'data/'
    results_filepath = 'results/'
    checkpoint_filepath = 'checkpoint_data/'


    tags = ['lastfm','brightkite','wikispeedia','reuters']
    min_occurances = [50,10,10,10]

    data = {}


    # for i in range(len(tags)):
    print(f'processing {tags[job_number]}')
    data[tags[job_number]] = read_data(tags[job_number],data_path, min_occurances[job_number], checkpoint_filepath)