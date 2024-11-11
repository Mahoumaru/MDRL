# MDRL

Code repository for the paper "Domains as Objectives: Domain-Uncertainty-Aware Policy Optimization through Explicit Multi-Domain Convex Coverage Set Learning"

## How to use

See the files sac_arguments_file.py and launcher.py to see the full list of arguments.
For example, to run the eMDSAC algorithm on the Hopper-v3 environment, use:
'''
python launcher.py --algos emdsac --env_name Hopper-v3 --n_runs 1 --continuous --n_regions 1 --no_encoding --explicit_scal --use_ut --first_run_idx 0
'''

This will run only one random seed (n_runs = 1) using only one environment in the environment set (n_regions = 1) with explicit scalarization of the critic function (i.e. the critic takes the environment parameter $\kappa$ as input along with the state $s$ and uncertainty $\varpi$).
