import numpy as np
import torch
import gym
import argparse
from itertools import count
#from copy import deepcopy

def sac_algorithm():
    ## + Randomly initialize actor and the two critic networks

    ## + Initialize entropy temperature parameter \alpha

    ## + Initialize target networks and affect to them the same weights as the two critics
    ## networks previously initialized

    ## + Initialize the replay buffer

    ## + With M = total_num_episodes
    ## for episode = 1 to M do:
    for i_episode in range(1, total_num_episodes+1):
        ## + Receive initial observation (with reset of course)

        ## + With T = number of steps per episode
        ## (note that T can be different for each episode, so we use an infinite loop and break when done)
        ## for t = 1 to T do:
        done = False
        for t in count(0):
            ## + Select action according to the actor or
            ## randomly during the warmup steps.

            ## + Execute the action and observe reward r_t, new state and done signal

            ## + Store transition (s_t, a_t, r_t, s_{t+1}) in replay buffer R

            ## + With K the number of updates
            ## for k = 1 to K do:
            for k in range(number_of_updates):
                ## + Sample a random minibatch of N transitions (s_i, a_i, r_i, s_{i+1}) from the replay buffer R
                ## (This is only possible if there are at least N transitions in the replay buffer)

                ## + Set y_i = r_i + \gamma (min(Q_1'(s_{i+1}, \mu(s_{i+1})), Q_2'(s_{i+1}, \mu(s_{i+1}))) - \alpha * \log{\mu(s_{i+1})})
                ## (y_i is the temporal difference target)

                ## + Update both critics
                ## (Using the mean squared TD error, \frac{1}{N} \sum\limits_{i=1}^{N} [y_i - Q(s_i, a_i)]^2)

                ## + Update the actor policy using the sampled policy gradient

                ## + Update the target networks

            if done:
                break
