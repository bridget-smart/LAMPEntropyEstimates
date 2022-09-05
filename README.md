#  Linear Additive Markov Process (LAMP) Entropy Evaluations

This repo contains implementation of code to get entropy estimates from the paper **PAPER LINK TO BE ADDED**.

To run this code you need to:

1. Download the datasets and save to your data folder. The links to download each are available in the paper on page 5.

2. Get the transition matrices using the LAMP fit. One approach to do so is to save the transition matrices fit using the code from [Ravi Kumar, Maithra Raghu, Tamás Sarlós and Andrew Tomkins. "Linear Additive Markov Processes", WWW 2017.](https://arxiv.org/1704.01255) [here](https://github.com/google-research/google-research/tree/master/lamp). These should be saved in a folder `transition_matrices`, with the names `lastfm_transition_matrix.txt, brightkite_transition_matrix.txt, reuters_transition_matrix.txt and wikispeedia_transition_matrix.txt`. A modified version of the original code which saves these transition matrices is available at https://github.com/bridget-smart/modified_lamp.

3. Set up a folder to save checkpoint_data, results in. Set all relevant file paths in `run.py`. It is reccommended you use a file structure as follows: 

   ```
   folder >
   		code
   		transition_matrices/
   		data/
    		results/
   		checkpoint_data/
   ```

   

   

### Notes

These calculations take a while to run, especially those to simulate the estimates across multiple values of $p$. When running, it is advisable to break the script up into smaller jobs which can be run in parallel. You should also check the code runs on your system using a subset of the original datasets before running over the entire sequences. 

`process_data.py` is a script which was used to preprocess the datasets remotely and calculate the first-order Markov Chain transition matrices. This step is optional and is included for completeness.

`run.py` contains the weights for the artificial links used to force ergodicity in the Markov Chains to allow the stationary distribution to be calculated, as described in the paper. 

To reproduce the results to select these values, see `choosing_p.py`.

#### Wikispeedia Dataset
Once you download the wikispeedia paths from [snap.stanford.edu/data/wikispeedia.htm](snap.stanford.edu/data/wikispeedia.htm), the file `paths_finished.tsv` is renamed to `wiki_paths.txt` and this file is used.

#### Lastfm Dataset and Reuters Dataset
Can be simply downloaded and unzipped from the link in the paper.

#### Brightkite Dataset
Download the file `loc-brightkite_totalCheckins.txt.gz` from the link provided and unzip.
