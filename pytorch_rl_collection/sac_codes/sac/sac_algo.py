import os
import pandas as pd
from itertools import count
from pytorch_rl_collection.utils import evaluate, make_Dirs
from datetime import datetime

import numpy as np

def run_evaluation(eval_env, agent, args, output_path, eval_csv_writing_mode, best_reward=None):
    agent.save_model(output_path + "checkpoint/")
    ######
    if args.save_best:
        if best_reward is None:
            import math
            best_reward = -math.inf
        #avg_reward = 0.
        rewards = []
        for _ in range(args.num_evals):
            episode_reward = evaluate(eval_env, agent, algo_name=args.algo.upper())
            #avg_reward += episode_reward
            rewards.append(episode_reward)
        #avg_reward /= args.num_evals
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save_model(output_path + "best/")
    else:
        #avg_reward = evaluate(eval_env, agent, algo_name=args.algo.upper())
        #avg_reward = 0.
        rewards = []
        for _ in range(args.num_evals):
            episode_reward = evaluate(eval_env, agent, algo_name=args.algo.upper())
            #avg_reward += episode_reward
            rewards.append(episode_reward)
        #avg_reward /= args.num_evals
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
    ######
    file_not_exists = not os.path.isfile(output_path + "checkpoint/eval_returns.csv")
    overwrite = (eval_csv_writing_mode == "w")
    pd.DataFrame({"Steps": [agent.steps], "Scores": [avg_reward]}).to_csv(output_path + "checkpoint/eval_returns.csv",
        mode=eval_csv_writing_mode, header=(file_not_exists or overwrite)
    )
    return avg_reward, std_reward, best_reward

def sac_algorithm(agent, env, args, output_path, eval_env):
    #output_path = "./trained_agents/{}/{}/{}/".format(args.algo, args.env_name, args.seed)
    make_Dirs(output_path + "checkpoint/")
    if args.save_best:
        make_Dirs(output_path + "best/")
    #####
    best_reward = None
    evaluated = False
    eval_milestone = args.total_num_steps // 200#5 * env._max_episode_steps
    #####
    train_csv_writing_mode = 'w'
    eval_csv_writing_mode = 'w'
    ## + Randomly initialize actor and the two critic networks
    agent.initialize_critics()
    agent.initialize_actor()
    ## + Initialize entropy temperature parameter \alpha
    agent.initialize_entropy_temperature()
    ## + Initialize target networks and affect to them the same weights as the two critics
    ## networks previously initialized
    agent.initialize_targets()
    ## + Initialize the replay buffer
    agent.initialize_replay_buffer()
    ## + With M = total_num_episodes
    ## for episode = 1 to M do:
    i_episode = 0
    #for i_episode in range(1, total_num_episodes+1):
    while agent.steps < args.total_num_steps:
        i_episode += 1
        ## + Receive initial observation (with reset of course)
        state, _ = env.reset()
        ## + With T = number of steps per episode
        ## (note that T can be different for each episode, so we use an infinite loop and break when done)
        ## for t = 1 to T do:
        episode_reward = 0.
        done = False
        for t in count(0):
            ## + Select action according to the actor or
            ## randomly during the warmup steps.
            action = agent.select_action(state)
            ## + Execute the action and observe reward r_t, new state and done signal
            new_state, reward, done, _, info = env.step(action)
            episode_reward += reward
            ## + Store transition (s_t, a_t, r_t, s_{t+1}) in replay buffer R
            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1. if t+1 == env._max_episode_steps else float(not done)
            agent.store_transition((state, action, reward, new_state, mask, info))
            ## + With K the number of updates
            ## for k = 1 to K do:
            for k in range(args.number_of_updates):
                ## + Sample a random minibatch of N transitions (s_i, a_i, r_i, s_{i+1}) from the replay buffer R
                ## (This is only possible if there are at least N transitions in the replay buffer)
                minibatch_states, minibatch_actions, minibatch_rewards, minibatch_new_states, minibatch_masks, minibatch_info = agent.sample_minibatch(batch_size=args.batch_size)
                ## + Set y_i = r_i + \gamma (min(Q_1'(s_{i+1}, \mu(s_{i+1})), Q_2'(s_{i+1}, \mu(s_{i+1}))) - \alpha * \log{\mu(s_{i+1})})
                ## (y_i is the temporal difference target)
                y = agent.get_td_target(states=minibatch_states, rewards=minibatch_rewards, next_states=minibatch_new_states, masks=minibatch_masks)
                ## + Update both critics
                ## (Using the mean squared TD error, \frac{1}{N} \sum\limits_{i=1}^{N} [y_i - Q(s_i, a_i)]^2)
                agent.update_critics(td_target=y, states=minibatch_states, actions=minibatch_actions)
                ## + Update the actor policy using the sampled policy gradient
                agent.update_actor(states=minibatch_states)
                ## + Update the entropy temperature parameter
                agent.update_temperature(states=minibatch_states)
                ## + Update the target networks
                agent.update_target_networks(minibatch_states is not None)
            ##
            state = new_state
            #####
            #if i_episode % 10 == 0:
            if agent.steps % eval_milestone == 0:
                ######
                avg_reward, std_reward, best_reward = run_evaluation(eval_env, agent, args, output_path,
                    eval_csv_writing_mode, best_reward=best_reward
                )
                eval_csv_writing_mode = 'a'
                evaluated = True
            #####
            if done or (t+1) == env._max_episode_steps:
                print("[SAC] Episode: {}; Steps {}; Episode Acc. Reward: {}".format(i_episode, agent.steps, episode_reward))
                file_not_exists = not os.path.isfile(output_path + "checkpoint/train_returns.csv")
                overwrite = (train_csv_writing_mode == "w")
                pd.DataFrame({"Episodes": [i_episode], "Steps": [agent.steps], "Scores": [episode_reward],
                            "Temperature": [agent.alpha]}).to_csv(output_path + "checkpoint/train_returns.csv",
                             mode=train_csv_writing_mode, header=(file_not_exists or overwrite)
                )
                train_csv_writing_mode = 'a'
                if evaluated:
                    print("--------------------------------")
                    print("[{}] Evaluation over {} Episodes Avg. Acc. Reward: {} ({})".format(args.algo.upper(), args.num_evals, avg_reward, std_reward))
                    print("--------------------------------")
                    evaluated = False
                break
        ####

###########################################
def run_sac_algorithm(args):
    import robel
    import custom_gym
    import gymnasium as gym
    import pybulletgym
    import torch
    from torchoptim.optimizers import AdaTerm
    from torchoptim.optimizers import TAdam, TAdaBelief, AdaSkew
    #import numpy as np
    from pytorch_rl_collection.replay_buffers.replay_memory import ReplayMemory
    from pytorch_rl_collection.sac_codes.sac.sac_agent import SAC_AGENT
    from pytorch_rl_collection.envs_utils.normalized_env import NormalizedEnv

    # - Initialize the environment
    if "Pendulum" in args.env_name:
        max_episode_steps = 200
    elif "DClaw" in args.env_name:
        max_episode_steps = 40
        observation_keys = (
           'claw_qpos',
           #'claw_qvel',
           #'object_qpos',
           'object_x',
           'object_y',
           #'object_qvel',
           'last_action',
           #'target_qpos',
           #'target_x',
           #'target_y',
           'target_error',
        )
        env = gym.make(args.env_name, sim_observation_noise=None, observation_keys=observation_keys)
        value_range = env.robot.get_config('dclaw').qpos_range
        add_observation_noise = False
        eval_env = gym.make(args.env_name, sim_observation_noise=None, observation_keys=observation_keys)
    elif "Hexapod" in args.env_name:
        max_episode_steps = 400
        env = gym.make(args.env_name)
        value_range = None
        add_observation_noise = False
        eval_env = gym.make(args.env_name)
    else:
        max_episode_steps = 1000
        env = gym.make(args.env_name)
        value_range = None
        add_observation_noise = False
        eval_env = gym.make(args.env_name)
    ##
    if getattr(env, "_max_episode_steps", None) is None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=max_episode_steps)
    #print(env._max_episode_steps)
    env = NormalizedEnv(env)
    eval_env = NormalizedEnv(eval_env)
    #print(env._max_episode_steps)
    np.random.seed(args.seed)
    _ = env.reset(seed=args.seed)
    _ = eval_env.reset(seed=args.seed)

    states_dim = env.observation_space.shape[0]
    actions_dim = env.action_space.shape[0]
    # - Initialize the agent
    ### The agent carries all the functions/operations inherent to the algorithm
    ### such as update_actor(), store_transition() or even initialize_replay_buffer().
    ### It is therefore just a function holder.
    agent = SAC_AGENT(states_dim, actions_dim, args, OptimCls=TAdam,#TAdaBelief,#AdaTerm,#torch.optim.Adam,#
                       ReplayBufferCls=ReplayMemory, value_range=value_range,
                       add_noise=add_observation_noise, OptimArgs={"beta_dof": 1., "k_dof": 1.})

    # - Make output path
    date = datetime.now()
    date = [str(date.day), str(date.month), str(date.year)]
    date[0] = date[0] if len(date[0]) > 1 else "0"+date[0]
    date[1] = date[1] if len(date[1]) > 1 else "0"+date[1]
    date = reversed(date)
    opt = "tadam"
    output_path = "./trained_agents/{}/{}/{}/".format(
        args.algo + ("_vargrad" if args.vargrad else "") + "_{}".format(opt.lower()),
        args.env_name + ("_ObsNoise" if add_observation_noise else "") + "_" + "".join(date), args.seed
    )
    make_Dirs(output_path)
    # - Run the SAC algorithm
    sac_algorithm(agent, env, args, output_path, eval_env)
    # - Close the environment
    env.close()
    eval_env.close()
    # - Save the agent actor and critic
    agent.save_model(output_path)
