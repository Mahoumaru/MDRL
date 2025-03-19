import os
import torch
import pandas as pd
from itertools import count
from pytorch_rl_collection.utils import evaluate, md_evaluate, make_Dirs
from datetime import datetime

from copy import deepcopy

def sirsa_algorithm(agent, env, args, np, use_encoder, output_path):
    algo_name_upper = (args.algo + ("_UT" if args.use_ut else "_MC") + ("_EX" if args.explicit_scal else "_IMP")).upper()
    make_Dirs(output_path + "checkpoint/")
    if args.save_best:
        make_Dirs(output_path + "best/")
    #####
    eval_envs = []
    for j in range(2):
        eval_env = deepcopy(env.env_set[0])
        try:
            eval_env.reset(seed=args.seed)
        except TypeError as err:
            eval_env.seed(seed=args.seed)
        #print(eval_env.reset())
        eval_envs.append(eval_env)
    #####
    train_csv_writing_mode = 'w'
    eval_csv_writing_mode = 'w'
    temp_csv_writing_mode = 'w'
    #####
    ## + Randomly initialize actor, disturber and the two critic networks
    agent.initialize_critics()
    agent.initialize_actor()
    ## + Initialize entropy temperature parameters \alpha
    agent.initialize_entropy_temperature()
    ## + Initialize target networks and affect to them the same weights as the two critics
    ## networks previously initialized
    agent.initialize_targets()
    ## +
    agent.initialize_systemID()
    ## + Initialize the replay buffer
    agent.initialize_replay_buffer_set(size=args.n_regions)
    ## +
    agent.set_varpi_ranges(env.subdomains.ranges)
    ###
    if not args.just_run_test_target:
        ## + With M = total_num_episodes
        ## for episode = 1 to M do:
        i_episode = 0
        n_actors = 1
        #total_num_episodes = 2 * (args.total_num_steps // env._max_episode_steps)
        #for i_episode in range(1, total_num_episodes+1):
        while agent.steps < args.total_num_steps:
            i_episode += 1
            ## + Receive initial observation (with reset of course)
            state = env.reset()
            ## + Sample degen varpi or param
            if agent.continuous:
                I, varpi, idx = env.sample_env_params()
            else:
                raise NotImplementedError
            ## + Sample varpi
            varpi = torch.FloatTensor(varpi).to(agent.device)
            print(varpi, "; idx =", idx)
            ## + Preserve this original varpi
            orig_varpi = deepcopy(varpi)
            ## + With T = number of steps per episode
            ## (note that T can be different for each episode, so we use an infinite loop and break when done)
            ## for t = 1 to T do:
            episode_rewards = np.zeros(1)
            done = False
            for t in count(0):
                ## + Sample alpha from varpi
                pre_not_dones = ~env.get_dones()
                valid_idxs = env.get_stillActiveIdxs()
                ## + Select action according to the actor and disturber, or
                ## randomly during the warmup steps.
                action = agent.select_action(state, varpi)
                ## + Execute the action and observe reward r_t, new state and done signal
                new_state, reward, done, _, _ = env.step(action)
                #print(new_state.shape, done.shape, varpi.shape, reward.shape, episode_rewards.shape, pre_not_dones.shape)
                episode_rewards[pre_not_dones] += reward
                ## + Store transition (s_t, a_t, r_t, s_{t+1}) in replay buffer R
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
                    minibatch_states, minibatch_actions, minibatch_rewards, minibatch_new_states, \
                    minibatch_masks, minibatch_kappas, minibatch_varpis, minibatch_idxs = agent.rmset_sample_minibatch(batch_size=args.batch_size)
                    ##
                    if agent.steps > agent.warmup:
                        assert minibatch_states is not None
                    ## + Update the system identifier models (all elements of the ensemble)
                    agent.update_systemID(states=minibatch_states, actions=minibatch_actions, next_states=minibatch_new_states,
                        kappas=minibatch_kappas, varpis=minibatch_varpis, idxs=minibatch_idxs
                    )
                    ## + Set y_i = r_i + \gamma (min(Q_1'(s_{i+1}, \mu(s_{i+1})), Q_2'(s_{i+1}, \mu(s_{i+1}))) - \alpha * \log{\mu(s_{i+1})})
                    ## (y_i is the temporal difference target)
                    y = agent.get_td_target(states=minibatch_states, actions=minibatch_actions, rewards=minibatch_rewards,
                        next_states=minibatch_new_states, masks=minibatch_masks, kappas=minibatch_kappas,
                        varpis=minibatch_varpis
                    )
                    ## + Update both critics
                    ## (Using the mean squared TD error, \frac{1}{N} \sum\limits_{i=1}^{N} [y_i - Q(s_i, a_i)]^2)
                    agent.update_critics(td_target=y, states=minibatch_states, actions=minibatch_actions, kappas=minibatch_kappas)
                    ## + Update the actor policy using the sampled policy gradient
                    agent.update_actor(states=minibatch_states, kappas=minibatch_kappas, varpis=minibatch_varpis)
                    ## + Update the entropy temperature parameter
                    _ = agent.update_temperature(states=minibatch_states, varpis=minibatch_varpis)
                    temp_csv_writing_mode = 'a'
                    ## + Update the target networks
                    agent.update_target_networks(minibatch_states is not None)
                ##
                orig_varpi = orig_varpi[~done]
                ### Update varpi
                varpi = agent.update_uncertainty(state[~done], action[~done], new_state[~done], varpi[~done])
                ##
                state = new_state[~done]
                I = I[~done]
                if (done == True).all() or (t+1) == env._max_episode_steps:
                    min_ret = (np.round(np.min(episode_rewards), 3), np.argmin(episode_rewards)+1)
                    max_ret = (np.round(np.max(episode_rewards), 3), np.argmax(episode_rewards)+1)
                    mean_ret = np.round(np.mean(episode_rewards), 3)
                    median_ret = np.round(np.median(episode_rewards), 3)
                    print("[{}] Episode: {}; Steps {}; Episode Acc. Reward: Min = {}, Max = {}, Mean = {}, Median = {}".format(algo_name_upper, i_episode, agent.steps,
                        min_ret, max_ret, mean_ret, median_ret
                    ))
                    pd.DataFrame(episode_rewards.reshape(1, -1)).to_csv(output_path + "checkpoint/train_returns.csv",
                                 mode=train_csv_writing_mode,
                                 header=(train_csv_writing_mode == 'w' or not os.path.isfile(output_path + "checkpoint/train_returns.csv"))
                    )
                    train_csv_writing_mode = 'a'
                    break
            ####
            if agent.steps >= (n_actors * args.total_num_steps):
                agent.save_model(output_path, id=n_actors)
                n_actors += 1
            ####
            if i_episode % 10 == 0:
                if args.continuous:
                    nominal_varpi = np.ones_like(env.latent_ranges[:, 1])
                    nominal_varpi = np.concatenate((nominal_varpi, np.zeros_like(nominal_varpi)), axis=-1)
                    fullrange_varpi = np.concatenate(
                       (0.5 * (env.latent_ranges[:, 1] + env.latent_ranges[:, 0]),
                        np.sqrt(1./12.) * (env.latent_ranges[:, 1] - env.latent_ranges[:, 0])), axis=-1
                    )
                else:
                    nominal_varpi, _, _ = env.subdomains.get_point_domain_id(point=np.ones_like(env.latent_ranges[:, 1]))#env.nominal_params)
                ######
                agent.save_model(output_path + "checkpoint/")
                ######
                avg_reward = np.zeros(2)
                for _ in range(args.num_evals):
                    episode_reward, _ = md_evaluate(
                        envs=eval_envs,
                        varpis=[nominal_varpi, fullrange_varpi],
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
                        import math
                        best_reward = -math.inf
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        agent.save_model(output_path + "best/")
                ######
                pd.DataFrame(avg_reward.reshape(1, -1)).to_csv(
                    output_path + "checkpoint/eval_returns.csv",
                    mode=eval_csv_writing_mode,
                    header=(eval_csv_writing_mode == 'w' or not os.path.isfile(output_path + "checkpoint/eval_returns.csv"))
                )
                eval_csv_writing_mode = 'a'
    else:
        agent.load_weights(output_path)

###########################################
def run_sirsa_algorithm(args):
    #import robel
    #import gym
    #import pybulletgym
    from torchoptim.optimizers import TAdam
    from torch.optim import Adam
    import numpy as np
    from pytorch_rl_collection.replay_buffers.replay_memory import MDReplayMemorySet
    from pytorch_rl_collection.sac_codes.sirsa.sirsa_agent import SIRSA_AGENT
    from pytorch_rl_collection.envs_utils.env_set import SIRSAMDEnvSet
    ######
    use_encoder = not args.no_encoding
    # - Initialize the environment
    env = SIRSAMDEnvSet(env_name=args.env_name, multi_dim=args.multi_dim, n_regions=1, n_varpis=args.n_regions, seed=args.seed, use_encoder=use_encoder,
        add_noise=False, use_quasi_random=False
    )
    #print(env._max_episode_steps)
    np.random.seed(args.seed)
    env.seed(args.seed)
    _ = env.sample_varpis()

    states_dim = env.observation_space.shape[0]
    actions_dim = env.action_space.shape[0]
    domains_dim = env.subdomains.domain_dim
    if "DClaw" in args.env_name:
        value_range = env.env_set[0].robot.get_config('dclaw').qpos_range
        add_observation_noise = False
    else:
        value_range = None
        add_observation_noise = False
    # - Initialize the agent
    model_type = ""
    ### The agent carries all the functions/operations inherent to the algorithm
    ### such as update_actor(), store_transition() or even initialize_replay_buffer().
    ### It is therefore just a function holder.
    agent = SIRSA_AGENT(states_dim, actions_dim, domains_dim, args, OptimCls=TAdam,
                       ReplayBufferCls=MDReplayMemorySet, value_range=value_range,
                       add_noise=add_observation_noise,
                       model_type=model_type, threshold=args.total_num_steps//2)

    # - Run the SIRSA algorithm
    date = datetime.now()
    date = [str(date.year), str(date.month), str(date.day)]
    date[2] = date[2] if len(date[2]) > 1 else "0"+date[2]
    date[1] = date[1] if len(date[1]) > 1 else "0"+date[1]
    if not args.just_run_test_target:
        output_path_suffix = "_20240710" if args.multi_dim else "_20240802"
    else:
        output_path_suffix = "_20240710" if args.multi_dim else "_20240802"
    ###
    algo = args.algo + ("_UT" if args.use_ut else "_MC") + ("_EX" if args.explicit_scal else "_IMP") + ("_resample" if args.resample else "") + ("_cont" if args.continuous else "_disc")
    env_key_list = list(env.params_idxs.keys())
    cvar_suffix = "_cvar{}".format(str(args.cvar_alpha).replace(".", "p")) if args.use_cvar else ""
    output_path = "./trained_agents/{}/{}_{}/{}/".format(
        algo + (("_" + model_type) if model_type != "" else "") + cvar_suffix,
        args.env_name + ("_ObsNoise" if add_observation_noise else ""),
        ("all" if len(env_key_list) > 2 else "_".join(env_key_list)) + output_path_suffix,# +"_2000ep_true_truesirsa",
        args.seed
    )
    ###
    sirsa_algorithm(agent, env, args, np, use_encoder, output_path)
    if not args.just_run_test_target:
        # - Save the agent actor and critic
        make_Dirs(output_path)
        agent.save_model(output_path)
    # - Close the environment
    env.close()
    ################
    ################
    from pytorch_rl_collection.test_target_env import run_test_target
    use_osi = True
    run_test_target(actor=agent.actor, output_path=output_path, env_name=args.env_name, algo_name=args.algo,
            actor_seed=args.seed, dim=domains_dim, osi_net=agent.systemID,
            ccs=True, continuous=args.continuous, use_osi=use_osi, save_results=True, use_encoder=use_encoder,
            seed=1234, n_runs=50, cuda=args.cuda, cuda_id=args.cuda_id)
