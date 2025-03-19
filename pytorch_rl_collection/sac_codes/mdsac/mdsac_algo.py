"""
Joint algorithm file for all MDSAC algorithms, i.e.
DRSAC, sDRSAC, (c,e,u)-MDSAC and uMD-SIRSA
"""
import os
import torch
import pandas as pd
from itertools import count
from pytorch_rl_collection.utils import evaluate, md_evaluate, make_Dirs
from datetime import datetime

from copy import deepcopy

from pytorch_rl_collection.sac_codes.mdsac.mdsac_agent import *
from pytorch_rl_collection.replay_buffers.replay_memory import MDReplayMemorySet
from pytorch_rl_collection.envs_utils.env_set import SIRSAMDEnvSet, MDEnvSet

#########################################################################################
from hilbertcurve.hilbertcurve import HilbertCurve
from collections import defaultdict

# Map samples to Hilbert indices
def get_hilbert_indices(hilbert_curve, samples, bin_size):
    hilbert_indices = []
    for sample in samples:
        scaled_sample = [int(coord // bin_size) for coord in sample]
        hilbert_index = hilbert_curve.distance_from_point(scaled_sample)
        hilbert_indices.append(hilbert_index)
    return hilbert_indices

# Update counters for the samples
def update_counters(hilbert_curve, samples, bin_size, counters):
    hilbert_indices = get_hilbert_indices(hilbert_curve, samples, bin_size)
    for index in hilbert_indices:
        counters[index] += 1
    return counters

#########################################################################################
def get_agent(algo_name):
    return {
        "drsac": DRSAC_AGENT,
        "sdrsac": SDRSAC_AGENT,
        "cmdsac": CMDSAC_AGENT,
        "emdsac": EMDSAC_AGENT,
        "umdsac": UMDSAC_AGENT,
        "umd_sirsa": UMD_SIRSA_AGENT,
    }[algo_name]

#########################################################################################
def get_env_set(algo_name):
    if algo_name == "sirsa":
        return SIRSAMDEnvSet
    else:
        return MDEnvSet

#########################################################################################
def make_eval_envs(algo_name, core_env, seed):
    n_eval_envs = 1 if "drsac" in algo_name else 2
    eval_envs = []
    for j in range(n_eval_envs):
        eval_env = deepcopy(core_env)
        try:
            eval_env.reset(seed=seed)
        except TypeError as err:
            eval_env.seed(seed=seed)
        ####
        eval_envs.append(eval_env)
    #####
    return eval_envs, n_eval_envs

#########################################################################################
def mdsac_algorithm(agent, env, args, np, use_encoder, output_path):
    algo_name_upper = args.algo.upper()
    if algo_name_upper == "UMDSAC":
        algo_name_upper += "-v2" if args.use_solver else "-v1"
    if args.algo != "drsac":
        algo_name_upper += ("_UT" if args.use_ut else "_MC") + ("_EX" if args.explicit_scal else "_IMP")
    #####
    make_Dirs(output_path + "checkpoint/")
    if args.save_best:
        make_Dirs(output_path + "best/")
    #####
    eval_envs, n_eval_envs = make_eval_envs(args.algo, env.env_set[0], args.seed)
    #####
    train_csv_writing_mode = 'w'
    eval_csv_writing_mode = 'w'
    sampled_params_writing_mode = 'w'
    #####
    ## + Initialize entropy temperature parameters \alpha
    agent.initialize_entropy_temperature()
    ## + Set the uncertainty space
    agent.set_varpi_ranges(env.subdomains.ranges)
    ## + Randomly initialize actor
    agent.initialize_actor()
    ## + Randomly initialize the two critic networks
    agent.initialize_critics()
    ## + Initialize target networks and affect to them the same weights as the two critics
    ## networks previously initialized
    agent.initialize_targets()
    ## + Randomly initialize solver network (only relevant for eMDSAC and uMDSAC)
    agent.initialize_solver(use_solver=args.use_solver)
    ## + Randomly initialize dynamics network
    agent.initialize_dynamics_net()
    ## + Randomly initialize OSI (relevant for SIRSA and (c,e,u)-MDSAC algos)
    agent.initialize_OSI()
    ## + Initialize the replay buffer
    agent.initialize_replay_buffer_set(size=args.n_regions)
    #########
    p = 4  # Order of the Hilbert curve (2^p bins per dimension)
    bin_size = 1/(2**p)  # Size of each bin
    hilbert_curve = HilbertCurve(p, agent.varpi_size // 2)
    # Initialize counters
    counters = defaultdict(int)
    ###
    if not args.just_run_test_target:
        ## + With M = total_num_episodes
        ## for episode = 1 to M do:
        i_episode = 0
        while agent.steps < args.total_num_steps:
            i_episode += 1
            ## + Receive initial observation
            state = env.reset()
            ## + Sample environment parameters
            # if agent.continuous: (IF condition not needed, since we assume continuous uncertainty spaces)
            I = env.sample_env_params(use_varpi=False)
            ## + Sample varpi using the sampled parameters (i.e. varpi is sampled such that
            ## I belongs to its support). Note that we could also do the reverse (i.e., sample varpi
            ## first and then sample the parameters, but there is no theoretical difference between the two)
            orig_varpi = agent.sample_varpis(kappas=I)
            ## + Set initial timestep's varpi to the original one
            varpi = deepcopy(orig_varpi)
            ## + With T = number of steps per episode
            ## (note that T can be different for each episode, so we use an infinite loop and break when done)
            ## for t = 1 to T do:
            episode_rewards = np.zeros(env.n_regions)
            episode_loss = 0.
            done = False
            for t in count(0):
                pre_not_dones = ~env.get_dones()
                valid_idxs = env.get_stillActiveIdxs()
                ## + Select action according to the actor, or
                ## randomly during the warmup steps.
                action = agent.select_action(state, varpi, kappa=None)
                ## + Execute the action and observe reward r_t, new state and done signal
                new_state, reward, done, _, _ = env.step(action)
                ## + Update the episode return
                episode_rewards[pre_not_dones] += reward
                ## + Store transition (s_t, a_t, r_t, s_{t+1}, kappa, varpi) in replay buffer R
                # Ignore the "done" signal if it comes from hitting the time horizon.
                # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = np.ones(done.shape) if t+1 == env._max_episode_steps else (~done).astype(float)
                agent.rmset_store_transition((state, action, reward, new_state, mask, I, orig_varpi.cpu().numpy()), valid_idxs=valid_idxs)
                ## + With K the number of updates
                ## for k = 1 to K do:
                for k in range(args.number_of_updates):
                    ## + Sample a random minibatch of N transitions (s_i, a_i, r_i, s_{i+1}) from the replay buffer R
                    ## (This is only possible if there are at least N transitions in the replay buffer)
                    minibatch_states = None
                    for mb_s, mb_a, mb_r, mb_ns, mb_m, mb_p, mb_v in agent.rmset_sample_minibatch(batch_size=args.batch_size):
                        if minibatch_states is None:
                            minibatch_states, minibatch_actions, minibatch_rewards, minibatch_new_states, \
                            minibatch_masks, minibatch_kappas, minibatch_varpis = mb_s, mb_a, \
                            mb_r, mb_ns, mb_m, mb_p, mb_v
                        else:
                            minibatch_states = torch.cat((minibatch_states, mb_s), dim=0)
                            minibatch_actions = torch.cat((minibatch_actions, mb_a), dim=0)
                            minibatch_rewards = torch.cat((minibatch_rewards, mb_r), dim=0)
                            minibatch_new_states = torch.cat((minibatch_new_states, mb_ns), dim=0)
                            minibatch_masks = torch.cat((minibatch_masks, mb_m), dim=0)
                            minibatch_kappas = torch.cat((minibatch_kappas, mb_p), dim=0)
                            minibatch_varpis = torch.cat((minibatch_varpis, mb_v), dim=0)
                    ####
                    ## + If use Hindsight Experience Replay.
                    if minibatch_states is not None and args.use_her:
                        mb_k = ((minibatch_kappas - agent.min_rel_param) / (agent.max_rel_param - agent.min_rel_param)).cpu().numpy()
                        ######
                        counters = update_counters(hilbert_curve, mb_k, bin_size, counters)
                        pd.DataFrame(counters, index=[0]).to_csv(
                            output_path + "checkpoint/minibatch_params_count.csv",
                            mode='w',
                            header=True
                        )
                        ##
                        assert minibatch_states.shape[0] == args.batch_size
                        minibatch_varpis = agent.sample_varpis(size=args.batch_size, her=True, kappas=minibatch_kappas)
                    elif minibatch_states is None:
                        minibatch_varpis = None
                    ####
                    ## + Update the dynamics network
                    agent.update_dynamics_net(states=minibatch_states, actions=minibatch_actions, next_states=minibatch_new_states,
                        kappas=minibatch_kappas
                    )
                    ## + Update the system identifier models (all elements of the ensemble)
                    ## (relevant for SIRSA and (c,e,u)-MDSAC algos)
                    agent.update_systemID(states=minibatch_states, actions=minibatch_actions, next_states=minibatch_new_states,
                        kappas=minibatch_kappas, varpis=minibatch_varpis
                    )
                    ## + Set y_i = r_i + \gamma (min(Q_1'(s_{i+1}, \mu(s_{i+1})), Q_2'(s_{i+1}, \mu(s_{i+1}))) - \alpha * \log{\mu(s_{i+1})})
                    ## (y_i is the temporal difference target)
                    y = agent.get_td_target(
                        states=minibatch_states,
                        actions=minibatch_actions,
                        rewards=minibatch_rewards,
                        next_states=minibatch_new_states,
                        masks=minibatch_masks,
                        kappas=minibatch_kappas,
                        varpis=minibatch_varpis
                    )
                    ## + Update both critics
                    ## (Using the mean squared TD error, \frac{1}{N} \sum\limits_{i=1}^{N} [y_i - Q(s_i, a_i)]^2)
                    agent.update_critics(td_target=y,
                        states=minibatch_states,
                        actions=minibatch_actions,
                        kappas=minibatch_kappas,
                        varpis=minibatch_varpis
                    )
                    ## + Update the actor policy using the sampled policy gradient
                    agent.update_actor(states=minibatch_states, kappas=minibatch_kappas, varpis=minibatch_varpis)
                    ## + Update the envelope solver (only relevant for eMDSAC and uMDSAC)
                    agent.update_solver(states=minibatch_states, kappas=minibatch_kappas,
                        varpis=minibatch_varpis
                    )
                    ## + Update the target networks
                    agent.update_target_networks(minibatch_states is not None)
                    ## + Update the entropy temperature parameter
                    agent.update_temperature(states=minibatch_states, kappas=minibatch_kappas, varpis=minibatch_varpis)
                ###
                orig_varpi = orig_varpi[~done]
                ### Update varpi (relevant for SIRSA and (c,e,u)-MDSAC algos)
                varpi = agent.update_uncertainty(state[~done], action[~done], new_state[~done], varpi[~done])
                ##
                state = new_state[~done]
                ##
                if (done == True).all() or (t+1) == env._max_episode_steps:
                    min_ret = (np.round(np.min(episode_rewards), 3), np.argmin(episode_rewards)+1)
                    max_ret = (np.round(np.max(episode_rewards), 3), np.argmax(episode_rewards)+1)
                    mean_ret = np.round(np.mean(episode_rewards), 3)
                    median_ret = np.round(np.median(episode_rewards), 3)
                    print("[{}] Episode: {}; Steps {}; Episode Acc. Reward: Min = {}, Max = {}, Mean = {}, Median = {}; Loss = {}".format(algo_name_upper, i_episode, agent.steps,
                        min_ret, max_ret, mean_ret, median_ret,
                        np.round(episode_loss, 4)
                    ))
                    pd.DataFrame({"Episodes": [i_episode], "Steps": [agent.steps], "Scores": [episode_rewards.reshape(-1)]}).to_csv(output_path + "checkpoint/train_returns.csv",
                                 mode=train_csv_writing_mode, header=(train_csv_writing_mode == 'w' or not os.path.isfile(output_path + "checkpoint/train_returns.csv"))
                    )
                    train_csv_writing_mode = 'a'
                    ######
                    pd.DataFrame({"x": I[:, 0], "y": I[:, 1], "T": [t+1]}).to_csv(
                        output_path + "checkpoint/sampled_params.csv",
                        mode=sampled_params_writing_mode,
                        header=(sampled_params_writing_mode == 'w' or not os.path.isfile(output_path + "checkpoint/sampled_params.csv"))
                    )
                    sampled_params_writing_mode = 'a'
                    break
                else:
                    ##
                    I = I[~done]
            ####
            if i_episode % 10 == 0:
                #if args.continuous:
                nominal_varpi = np.ones_like(env.latent_ranges[:, 1])
                nominal_varpi = np.concatenate((nominal_varpi, np.zeros_like(nominal_varpi)), axis=-1)
                fullrange_varpi = np.concatenate(
                   (0.5 * (env.latent_ranges[:, 1] + env.latent_ranges[:, 0]),
                    np.sqrt(1./12.) * (env.latent_ranges[:, 1] - env.latent_ranges[:, 0])), axis=-1
                )
                ######
                agent.save_model(output_path + "checkpoint/")
                ######
                avg_reward = np.zeros(n_eval_envs)
                for _ in range(args.num_evals):
                    episode_reward, _ = md_evaluate(
                        envs=eval_envs,
                        varpis=None if "drsac" in args.algo else [nominal_varpi, fullrange_varpi],
                        agent=agent,
                        algo_name=algo_name_upper,
                        osi_agent=None
                    )
                    avg_reward += episode_reward
                avg_reward /= args.num_evals
                print("--------------------------------")
                print("[{}] Evaluation over {} Episodes Avg. Acc. Reward: {}".format(algo_name_upper, args.num_evals, avg_reward))
                print("--------------------------------")
                ######
                if args.save_best:
                    if 'best_reward' not in locals():
                        best_reward = -np.inf
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        agent.save_model(output_path + "best/")
                ######
                pd.DataFrame({"Scores": [avg_reward.reshape(-1)]}).to_csv(
                    output_path + "checkpoint/eval_returns.csv",
                    mode=eval_csv_writing_mode,
                    header=(eval_csv_writing_mode == 'w' or not os.path.isfile(output_path + "checkpoint/eval_returns.csv"))
                )
                eval_csv_writing_mode = 'a'
    else:
        agent.load_weights(output_path)

#########################################################################################
def run_mdsac_algorithm(args):
    from torchoptim.optimizers import TAdam
    from torch.optim import Adam
    import numpy as np
    from pytorch_rl_collection.replay_buffers.replay_memory import MDReplayMemorySet
    ######
    print("#########################################")
    print("++ THIS IS THE JOINT ALGORITHM RUNNER.")
    print("#########################################")
    ######
    use_encoder = not args.no_encoding
    # - Initialize the environment
    env = get_env_set(args.algo)(env_name=args.env_name, multi_dim=args.multi_dim, n_regions=args.n_regions, seed=args.seed, use_encoder=use_encoder,
        add_noise=False, use_quasi_random=False
    )
    #####
    np.random.seed(args.seed)
    env.seed(args.seed)
    #####
    states_dim = env.observation_space.shape[0]
    actions_dim = env.action_space.shape[0]
    domains_dim = env.subdomains.domain_dim
    #####
    if "DClaw" in args.env_name:
        value_range = env.env_set[0].robot.get_config('dclaw').qpos_range
        add_observation_noise = False
    else:
        value_range = None
        add_observation_noise = False
    # - Initialize the agent
    ### The agent carries all the functions/operations inherent to the algorithm
    ### such as update_actor(), store_transition() or even initialize_replay_buffer().
    ### It is therefore just a function holder.
    agent = get_agent(args.algo)(states_dim, actions_dim, domains_dim, args, OptimCls=TAdam,
                       ReplayBufferCls=MDReplayMemorySet,
                       value_range=value_range,
                       add_noise=add_observation_noise)
    #####
    # - Run the algorithm
    date = datetime.now()
    date = [str(date.year), str(date.month), str(date.day)]
    date[2] = date[2] if len(date[2]) > 1 else "0"+date[2]
    date[1] = date[1] if len(date[1]) > 1 else "0"+date[1]
    if not args.just_run_test_target:
        output_path_suffix = "_" + "".join(date)
    else:
        ## sdrsac CVAR
        #assert args.multi_dim == False
        #output_path_suffix = "_20250223"
        ## Both umdsac CVAR
        assert args.multi_dim == False
        output_path_suffix = "_20250224"
        #output_path_suffix = "_20240710" if args.multi_dim else "_20240802"
    ###
    algo = args.algo
    if algo == "umdsac":
        algo += "_v2" if args.use_solver else "_v1"
    if args.algo != "drsac":
        algo += ("_UT" if args.use_ut else "_MC") + ("_EX" if args.explicit_scal else "_IMP")
    ###
    env_key_list = list(env.params_idxs.keys())
    cvar_suffix = "_cvar{}".format(str(args.cvar_alpha).replace(".", "p")) if args.use_cvar else ""
    output_path = "./trained_agents/{}/{}_{}/{}/".format(
        algo + ("_her" if args.use_her else "") + cvar_suffix,
        args.env_name + ("_ObsNoise" if add_observation_noise else ""),
        ("all" if len(env_key_list) > 2 else "_".join(env_key_list)) + output_path_suffix,
        args.seed
    )
    ####
    mdsac_algorithm(agent, env, args, np, use_encoder, output_path)
    if not args.just_run_test_target:
        # - Save the agent actor and critic
        make_Dirs(output_path)
        agent.save_model(output_path)
    #######
    # - Close the environment
    env.close()
    ################
    ################
    from pytorch_rl_collection.test_target_env import run_test_target
    use_osi = False
    run_test_target(actor=agent.actor, output_path=output_path, env_name=args.env_name, algo_name=args.algo,
            actor_seed=args.seed, dim=domains_dim, osi_net=agent.get_osi(),
            ccs=True, continuous=args.continuous, use_osi=use_osi, save_results=True, use_encoder=use_encoder,
            seed=1234, n_runs=50, cuda=args.cuda, cuda_id=args.cuda_id)
