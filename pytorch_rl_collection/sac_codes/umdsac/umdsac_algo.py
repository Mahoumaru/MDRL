import os
import torch
import pandas as pd
from itertools import count
from pytorch_rl_collection.utils import evaluate, md_evaluate, make_Dirs
from datetime import datetime

from copy import deepcopy

USE_SOLVER = False

def umdsac_algorithm(agent, env, args, np, use_encoder, output_path):
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
    #####
    ## + Randomly initialize actor, disturber and the two critic networks
    agent.initialize_critics()
    agent.initialize_actor()
    agent.initialize_solver(USE_SOLVER)
    ## + Initialize entropy temperature parameters \alpha
    agent.initialize_entropy_temperature()
    ## + Initialize target networks and affect to them the same weights as the two critics
    ## networks previously initialized
    agent.initialize_targets()
    ## +
    agent.initialize_systemID()
    ## +
    #agent.initialize_dynamics_net()
    ## + Initialize the replay buffer
    agent.initialize_replay_buffer_set(size=args.n_regions)
    ## +
    agent.set_varpi_ranges(env.subdomains.ranges)
    ## + Initialize OSI and Dynamics networks
    #agent.initialize_osi_dyn()
    ## + Initialize homotopy parameter
    #agent.initialize_lambda()
    ###
    if not args.just_run_test_target:
        ## + With M = total_num_episodes
        ## for episode = 1 to M do:
        i_episode = 0
        #total_num_episodes = 2 * (args.total_num_steps // env._max_episode_steps)
        #for i_episode in range(1, total_num_episodes+1):
        while agent.steps < args.total_num_steps:
            i_episode += 1
            ## + Receive initial observation (with reset of course)
            state = env.reset()
            ## + Sample degen varpi or param
            if agent.continuous:
                I = env.sample_env_params(use_varpi=False)
            else:
                I = np.eye(env.n_regions)
                _ = env.sample_env_params(use_varpi=False)
            ## + Sample varpi
            varpi = agent.sample_varpis(kappas=I)#(size=1)#
            ## + Preserve this original varpi
            orig_varpi = deepcopy(varpi)
            ## + With T = number of steps per episode
            ## (note that T can be different for each episode, so we use an infinite loop and break when done)
            ## for t = 1 to T do:
            episode_rewards = np.zeros(env.n_regions)
            episode_loss = 0.
            done = False
            for t in count(0):
                ## + Sample alpha from varpi
                pre_not_dones = ~env.get_dones()
                valid_idxs = env.get_stillActiveIdxs()
                if args.resample:
                    I = env.sample_env_params(use_varpi=False)
                ## + Select action according to the actor and disturber, or
                ## randomly during the warmup steps.
                #action = agent.select_action(state, np.concatenate((I, np.zeros_like(I)), axis=-1))#
                action = agent.select_action(state, varpi, kappa=None)
                ## + Execute the action and observe reward r_t, new state and done signal
                new_state, reward, done, _, _ = env.step(action)
                #print(new_state.shape, done.shape, varpi.shape, reward.shape, episode_rewards.shape, pre_not_dones.shape)
                episode_rewards[pre_not_dones] += reward
                #print(state.shape, new_state.shape, reward.shape, done.shape, alpha.shape, varpi.shape, valid_idxs)
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
                    for mb_s, mb_a, mb_r, mb_ns, mb_m, mb_p, mb_v in agent.rmset_sample_minibatch(batch_size=args.batch_size):
                        if minibatch_states is None:
                            minibatch_states, minibatch_actions, minibatch_rewards, minibatch_new_states, \
                            minibatch_masks, minibatch_kappas, HER_varpis = mb_s, mb_a, \
                            mb_r, mb_ns, mb_m, mb_p, mb_v
                        else:
                            minibatch_states = torch.cat((minibatch_states, mb_s), dim=0)
                            minibatch_actions = torch.cat((minibatch_actions, mb_a), dim=0)
                            minibatch_rewards = torch.cat((minibatch_rewards, mb_r), dim=0)
                            minibatch_new_states = torch.cat((minibatch_new_states, mb_ns), dim=0)
                            minibatch_masks = torch.cat((minibatch_masks, mb_m), dim=0)
                            minibatch_kappas = torch.cat((minibatch_kappas, mb_p), dim=0)
                            HER_varpis = torch.cat((HER_varpis, mb_v), dim=0)
                            #utopia_varpis = torch.cat((utopia_varpis, mb_utop_v), dim=0)
                            #minibatch_idxs = np.concatenate((minibatch_idxs, mb_idxs), axis=0)
                    ##
                    #"""
                    if minibatch_states is not None:
                        assert minibatch_states.shape[0] == args.batch_size
                        HER_varpis = agent.sample_varpis(size=args.batch_size, her=True, kappas=minibatch_kappas)
                    else:
                        HER_varpis = None
                    #"""
                    ##
                    #utopia_varpis = agent.update_utopia_varpis(next_states=minibatch_new_states, kappas=minibatch_kappas, utopia_varpis=utopia_varpis, replay_memory_indices=minibatch_idxs)
                    ## + Set y_i = r_i + \gamma (min(Q_1'(s_{i+1}, \mu(s_{i+1})), Q_2'(s_{i+1}, \mu(s_{i+1}))) - \alpha * \log{\mu(s_{i+1})})
                    ## (y_i is the temporal difference target)
                    y = agent.get_td_target(states=minibatch_states, actions=minibatch_actions, rewards=minibatch_rewards, next_states=minibatch_new_states, masks=minibatch_masks, kappas=minibatch_kappas, her_varpis=HER_varpis)
                    ## + Update both critics
                    ## (Using the mean squared TD error, \frac{1}{N} \sum\limits_{i=1}^{N} [y_i - Q(s_i, a_i)]^2)
                    agent.update_critics(td_target=y, states=minibatch_states, actions=minibatch_actions, kappas=minibatch_kappas, her_varpis=HER_varpis)
                    ## + Update the actor policy using the sampled policy gradient
                    agent.update_actor(states=minibatch_states, kappas=minibatch_kappas, her_varpis=HER_varpis)
                    ## + Update the envelope solver
                    agent.update_solver(states=minibatch_states,
                        kappas=minibatch_kappas
                    )
                    ## + Update the system identifier models (all elements of the ensemble)
                    agent.update_systemID(states=minibatch_states, actions=minibatch_actions, next_states=minibatch_new_states,
                        kappas=minibatch_kappas, varpis=HER_varpis
                    )
                    ## + Update the entropy temperature parameter
                    agent.update_temperature(states=minibatch_states, kappas=minibatch_kappas, her_varpis=HER_varpis)
                    ## + Update the target networks
                    agent.update_target_networks(minibatch_states is not None)
                ###
                orig_varpi = orig_varpi[~done]
                ### Update varpi
                varpi = agent.update_uncertainty(state[~done], action[~done], new_state[~done], varpi[~done])
                ##
                state = new_state[~done]
                I = I[~done]
                #varpi = agent.sample_varpis(kappas=I, her=True)
                if (done == True).all() or (t+1) == env._max_episode_steps:
                    min_ret = (np.round(np.min(episode_rewards), 3), np.argmin(episode_rewards)+1)
                    max_ret = (np.round(np.max(episode_rewards), 3), np.argmax(episode_rewards)+1)
                    mean_ret = np.round(np.mean(episode_rewards), 3)
                    median_ret = np.round(np.median(episode_rewards), 3)
                    print("[{}] Episode: {}; Steps {}; Episode Acc. Reward: Min = {}, Max = {}, Mean = {}, Median = {}; Loss = {}".format(algo_name_upper, i_episode, agent.steps,
                        min_ret, max_ret, mean_ret, median_ret,
                        np.round(episode_loss, 4)
                    ))
                    pd.DataFrame(episode_rewards.reshape(1, -1)).to_csv(output_path + "checkpoint/train_returns.csv",
                                 mode=train_csv_writing_mode, header=(train_csv_writing_mode == 'w' or not os.path.isfile(output_path + "checkpoint/train_returns.csv"))
                    )
                    train_csv_writing_mode = 'a'
                    break
            ####
            if agent.steps % args.total_num_steps == 0:
                agent.save_model(output_path, id=agent.steps//args.total_num_steps)
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
                avg_reward = np.zeros(2)#0.
                for _ in range(args.num_evals):
                    #episode_reward = evaluate(env.env_set[0], agent, algo_name=algo_name_upper, varpi=nominal_varpi)
                    episode_reward, _ = md_evaluate(
                        envs=eval_envs,#[deepcopy(env.env_set[0]) for _ in range(2)],
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
def run_umdsac_algorithm(args):
    #import robel
    #import gym
    #import pybulletgym
    from torchoptim.optimizers import TAdam
    from torch.optim import Adam
    import numpy as np
    from pytorch_rl_collection.replay_buffers.replay_memory import MDReplayMemorySet
    from pytorch_rl_collection.sac_codes.umdsac.umdsac_agent import UMDSAC_AGENT
    from pytorch_rl_collection.envs_utils.env_set import MDEnvSet
    ######
    use_encoder = not args.no_encoding#False
    # - Initialize the environment
    env = MDEnvSet(env_name=args.env_name, multi_dim=args.multi_dim, n_regions=args.n_regions, seed=args.seed, use_encoder=use_encoder,
        add_noise=False, use_quasi_random=False
    )
    #print(env._max_episode_steps)
    np.random.seed(args.seed)
    env.seed(args.seed)

    states_dim = env.observation_space.shape[0]
    actions_dim = env.action_space.shape[0]
    domains_dim = env.subdomains.domain_dim
    if "DClaw" in args.env_name:
        value_range = env.env_set[0].robot.get_config('dclaw').qpos_range
        add_observation_noise = False#True#
    else:
        value_range = None
        add_observation_noise = False
    # - Automatically set the lambda step length:
    lambda_steplength = 1. / (args.total_num_steps - args.warmup)
    assert lambda_steplength > 0.
    # - Initialize the agent
    model_type = ""
    ### The agent carries all the functions/operations inherent to the algorithm
    ### such as update_actor(), store_transition() or even initialize_replay_buffer().
    ### It is therefore just a function holder.
    agent = UMDSAC_AGENT(states_dim, actions_dim, domains_dim, args, OptimCls=TAdam,
                       ReplayBufferCls=MDReplayMemorySet, lambda_steplength=lambda_steplength,
                       value_range=value_range,
                       add_noise=add_observation_noise,
                       model_type=model_type)

    # - Run the sMDSAC algorithm
    date = datetime.now()
    date = [str(date.year), str(date.month), str(date.day)]
    date[2] = date[2] if len(date[2]) > 1 else "0"+date[2]
    date[1] = date[1] if len(date[1]) > 1 else "0"+date[1]
    if not args.just_run_test_target:
        output_path_suffix = "_20240802"#"_" + "".join(date)
    else:
        output_path_suffix = "_20240710" if args.multi_dim else "_20240802"#
    ###
    algo = args.algo + ("_UT" if args.use_ut else "_MC") + ("_EX" if args.explicit_scal else "_IMP") + ("_resample" if args.resample else "") + ("_cont" if args.continuous else "_disc")
    env_key_list = list(env.params_idxs.keys())
    output_path = "./trained_agents/{}/{}_{}/{}/".format(
        algo + ("_no_solver" if not USE_SOLVER else "") + "_her" + (("_" + model_type) if model_type != "" else ""),# + "_sampleDegenPol",
        args.env_name + ("_ObsNoise" if add_observation_noise else ""),
        ("all" if len(env_key_list) > 2 else "_".join(env_key_list)) + output_path_suffix,
        args.seed
    )
    ####
    umdsac_algorithm(agent, env, args, np, use_encoder, output_path)
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
            actor_seed=args.seed, dim=domains_dim, osi_net=agent.systemID,
            ccs=True, continuous=args.continuous, use_osi=use_osi, save_results=True, use_encoder=use_encoder,
            seed=1234, n_runs=50, cuda=args.cuda, cuda_id=args.cuda_id)
