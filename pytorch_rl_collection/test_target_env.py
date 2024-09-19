import gymnasium as gym
import robel
import pybulletgym
import torch
import numpy as np
import pandas as pd
import copy
from itertools import count
from pytorch_rl_collection.envs_utils.normalized_env import NormalizedEnv
from pytorch_rl_collection.utils import to_numpy, SubDomains, make_Dirs

from pytorch_rl_collection.utils import rsetattr, rgetattr, fetch_uniform_unscented_transfo, estimate_cvar, interquartile_mean
from pytorch_rl_collection.envs_utils.env_set import MDEnvSet

import yaml
import collections

from tqdm import tqdm
from gymnasium.wrappers import TimeLimit

from torch.distributions import Normal, Independent, Categorical, MixtureSameFamily

import os

################################################################################
HIDDEN_DIM_DICT = {
   "Hopper-v3": [256]*2,
   "HalfCheetah-v3": [256]*2,
   "Ant-v3": [256]*2,
   "Walker2d-v3": [256]*2,
   "DClawTurnFixed-v0": [256]*2,
   "DClawTurnRandom-v0": [256]*2,
   "Pendulum-v1": [64]*2,
}

################################################################################
### Modify simulator parameters
def make_target_env(env_name, seed, multi_dim, change_integrator=False, use_encoder=False):
    #env_name = str(env).split("<")[-1].split(">")[0]
    env = gym.make(env_name)#.replace("Fixed-v0", "FixedNoisy-v0"))
    env = NormalizedEnv(env)
    ########### Read config file
    if use_encoder:
        config_filename = env_name.split("-")[0] + ".yaml"
        if "DClaw" in config_filename:
            config_filename = "DClaw.yaml"
    else:
        config_filename = env_name.split("-")[0] + "_for_test" + ".yaml"
        if "DClaw" in config_filename:
            config_filename = "DClaw_for_test.yaml"
    ###
    with open("./pytorch_rl_collection/configs/" + config_filename, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    ######
    if multi_dim:
        config = config["multi_dim"]
    else:
        config = config["2d"]
    ######
    params_idxs = config["params_idxs"]
    params_env_idxs = config["params_env_idxs"]
    params_relative_ranges = config["params_relative_ranges"]
    ############
    if use_encoder:
        all_params = []
        if getattr(env, "sim", None) is not None:
            relative_coefs = {
              "body_mass": 1.,
              "dof_armature": 1.,
              "dof_damping": 1.,
              "actuator_gear": 1.,
              "opt.gravity": 1.,
              "opt.timestep": 1.#[0.75, 1.25]
            }
            if "DClaw" in config_filename:
                relative_coefs["body_pos"] = 1.
            #####
            orig_integrator = env.sim.model.opt.integrator
            if change_integrator and orig_integrator == 1:
                env.sim.model.opt.integrator = 0
            elif change_integrator and orig_integrator == 0:
                env.sim.model.opt.integrator = 1
            #####
            keys = relative_coefs.keys()
            #["body_mass", "dof_armature", "dof_damping", "opt.gravity"]#, "opt.density", "opt.viscosity"]
            for key in keys:
                idxs = params_env_idxs[key]
                take_tuple = False
                for elt in idxs:
                    if isinstance(elt, list):
                        take_tuple = True
                        break
                if take_tuple:
                    idxs = tuple(idxs)
                #####################
                params = copy.deepcopy(rgetattr(env.sim.model, key, None))
                assert params is not None
                #rsetattr(env.sim.model, key, relative_coefs[key] * params)
                if len(idxs) > 0:
                    #rgetattr(env.sim.model, key, None)[:] = relative_coefs[key] * params
                    rgetattr(env.sim.model, key, None)[idxs] = relative_coefs[key] * params[idxs]
                    all_params += [relative_coefs[key] for _ in range(params[idxs].shape[0])]
                else:
                    rsetattr(env.sim.model, key, relative_coefs[key] * params)
                    all_params += [relative_coefs[key]]
        elif env_name == "Pendulum-v1":
            relative_coefs = {
              "m": 1.,
              "l": 1.,
              "g": 1.,
              "dt": 1. #[]
            }
            #####
            keys = relative_coefs.keys()
            for key in keys:
                params = copy.deepcopy(rgetattr(env.env.env.env, key, None))
                assert params is not None
                rsetattr(env.env.env.env, key, relative_coefs[key] * params)
                all_params += [relative_coefs[key]]
        else:
            raise NotImplementedError
        ########
        if "DClaw" in env_name:
            env.reset()
        else:
            env.reset(seed=seed)
        ########
        all_params = np.array(all_params)
        print(all_params, all_params.shape)
        #####
        from pytorch_rl_collection.model_networks.model_networks import Encoder, Decoder
        enc_net = Encoder(nb_envparams=all_params.shape[0], nb_latent=2, hidden_layers=[300, 200], init_w=3e-3, verbose=False)
        enc_net.load_state_dict(
            torch.load('./trained_agents/params_encoding_res/{}/encoder.pkl'.format(env_name))
        )
        with torch.no_grad():
            encoded_params = enc_net(torch.FloatTensor(all_params).unsqueeze(0)).squeeze().numpy()
        print(encoded_params)
        ########
        p_lb, p_ub = np.zeros_like(all_params) + 0.5, np.zeros_like(all_params) + 2.
        p_lb[-1], p_ub[-1] = 0.75, 1.25
        dec_net = Decoder(nb_envparams=all_params.shape[0], nb_latent=2,
                lb=torch.FloatTensor(p_lb), ub=torch.FloatTensor(p_ub),
                hidden_layers=[200, 300], init_w=3e-3, verbose=False
        )
        dec_net.load_state_dict(
            torch.load('./trained_agents/params_encoding_res/{}/decoder.pkl'.format(env_name))
        )
        with torch.no_grad():
            phat = dec_net(torch.FloatTensor(encoded_params).unsqueeze(0))
            print(phat.squeeze().numpy())
            print(enc_net(phat).squeeze().numpy())
        #print(klkl)
        ########
        return env, {"x": encoded_params[0], "y": encoded_params[1]}, config["latent_ranges"]["x"], config["latent_ranges"]["y"], ""#relative_coefs
    else:
        latent_ranges = []
        for key, elt in params_idxs.items():
            latent_ranges.append(np.array(params_relative_ranges[key]).reshape(1, -1).repeat(len(elt), axis=0))
        #####
        latent_ranges = np.concatenate(latent_ranges, axis=0)
        ######
        lb, ub = latent_ranges[:,0], latent_ranges[:,1]
        #"""
        rng = np.random.RandomState(1234)
        for i in range(1):
            #params = np.random.uniform(low=lb, high=ub).reshape(-1)#np.ones_like(ub).reshape(-1) + 0.01#
            params = lb + (ub - lb) * rng.uniform(size=(lb.shape[0],)); filename_suffix = "Param{}".format(i+1)
            #_ = rng.uniform(size=(lb.shape[0],))
        #"""
        #params = np.ones_like(ub).reshape(-1); filename_suffix = "Nominal"
        #params = np.array([1.0, 0.08]); filename_suffix = "Varpi5Mean"
        #params = np.array([0.55, 1.81]); filename_suffix = "Varpi6Mean"
        print(params, params.shape)
        ######
        keys = params_idxs.keys()
        ###############
        if env_name == "Pendulum-v1":
            for key in keys:
                nominal_params = copy.deepcopy(rgetattr(env.env.env.env, key, None))
                assert nominal_params is not None
                rsetattr(env.env.env.env.env, key, params[params_idxs[key][0]] * nominal_params)
        elif getattr(env, "model", None) is not None:
            for key in keys:
                idxs = params_env_idxs[key]
                nominal_params = copy.deepcopy(rgetattr(env.sim.model, key, None))
                assert nominal_params is not None
                take_tuple = False
                for elt in idxs:
                    if isinstance(elt, list):
                        take_tuple = True
                        break
                if take_tuple:
                    idxs = tuple(idxs)
                ####
                nominal_params = nominal_params[idxs]
                #print(idxs)
                #print(key, params_idxs[key])
                if len(idxs) > 0:
                    rgetattr(env.sim.model, key, None)[idxs] = params[params_idxs[key]] * nominal_params
                    if "actuator_gainprm" in key:
                        idxs = (idxs[0], 1)
                        rgetattr(env.sim.model, key.replace("actuator_gainprm", "actuator_biasprm"), None)[idxs] = - params[params_idxs[key]] * nominal_params
                else:
                    rsetattr(env.sim.model, key, params[params_idxs[key]] * nominal_params)
        return env, {"x": params[0], "y": params[1]}, lb, ub, "_osi_Params" + filename_suffix

################################################################################
### Input to the actor
def make_actor_input(state, varpi, algo_name, continuous=False, device=torch.device("cpu")):
    if not isinstance(state, torch.Tensor):
        state = torch.FloatTensor(state).to(device)
    if len(state.shape) == 1:
        state = state.unsqueeze(0)
    ######
    if "md" in algo_name or "sirsa" in algo_name:
        if not isinstance(varpi, torch.Tensor):
            varpi = torch.FloatTensor(varpi).to(device)
        #print(varpi)
        if len(varpi.shape) == 1:
            varpi = varpi.unsqueeze(0)
        assert state.shape[0] == varpi.shape[0], "{} vs {}".format(state.shape, varpi.shape)
        ######
        if "ddpg" in algo_name:
            return (state, varpi)
        elif "sac" in algo_name or "sirsa" in algo_name:
            if not continuous:
                varpi = varpi / varpi.sum()
                assert np.round(np.sum(varpi.numpy()), 1) == 1., "{}; {}".format(varpi, varpi.sum())
            return (torch.cat((state, varpi), dim=-1).to(device),)
    else:
        return (state.to(device),)

################################################################################
### Input to the actor
def make_osi_input(state, action, new_state, varpi, device=torch.device("cpu")):
    if not isinstance(state, torch.Tensor):
        state = torch.FloatTensor(state).to(device)
    if not isinstance(action, torch.Tensor):
        action = torch.FloatTensor(action).to(device)
    if not isinstance(new_state, torch.Tensor):
        new_state = torch.FloatTensor(new_state).to(device)
    if not isinstance(varpi, torch.Tensor):
        varpi = torch.FloatTensor(varpi).to(device)
    ######
    if len(state.shape) == 1:
        state = state.unsqueeze(0)
    if len(action.shape) == 1:
        action = action.unsqueeze(0)
    if len(new_state.shape) == 1:
        new_state = new_state.unsqueeze(0)
    if len(varpi.shape) == 1:
        varpi = varpi.unsqueeze(0)
    assert state.shape[0] == varpi.shape[0], "{} vs {}".format(state.shape, varpi.shape)
    ######
    return (torch.cat((state, action, new_state, varpi), dim=-1).to(device),)

################################################################################
### Varpi
def make_varpis(n_regions, algo_name, continuous):
    if "md" in algo_name or "sirsa" in algo_name:
        if continuous:
            #"""
            means = np.linspace(0.5, 2., n_regions).reshape(-1, 1).repeat(2, axis=1)
            stds = np.zeros_like(means)
            mean = 0.5 * 2.5
            means = np.concatenate((means, np.array([[mean, mean]])), axis=0)
            std = 1.5 * np.sqrt(1./12.)
            stds = np.concatenate((stds, np.array([[std, std]])), axis=0)
            return [elt for elt in torch.cat((torch.FloatTensor(means), torch.FloatTensor(stds)), dim=-1)] # continuous case
            """
            means1 = np.linspace(0.9, 1.1, n_regions).reshape(-1, 1)
            means2 = np.linspace(0.8955, 1.0149, n_regions).reshape(-1, 1)
            means = np.concatenate((means1, means2), axis=1)
            stds = np.zeros_like(means)
            mean = 0.5 * np.array([[2., 0.8955 + 1.0149]])
            #print(means.shape, mean.shape)
            means = np.concatenate((means, mean), axis=0)
            std = np.array([[0.2, 1.0149 - 0.8955]]) * np.sqrt(1./12.)
            stds = np.concatenate((stds, std), axis=0)
            return [elt for elt in torch.cat( (torch.FloatTensor(means), torch.FloatTensor(stds)), dim=-1 )]
            #"""
        else:
            return [elt for elt in torch.cat((torch.eye(n_regions), torch.ones(1, n_regions) / n_regions), dim=0)] # discrete case
    else:
        return [None]

################################################################################
### Run
def run_episodes(algo_name, n_runs, env, actor, varpi, osi_net, n_regions, continuous, subdomains=None, data=None):
    episode_scores = []
    #####
    str_varpi = varpi
    if str_varpi == "osi":
        varpi = torch.ones(1, n_regions)# / n_regions
        states_dim = env.observation_space.shape[0]
        actions_dim = env.action_space.shape[0]
        ###
        #subdomains
    #####
    for i in range(n_runs):
        state = env.reset()
        if str_varpi == "osi":
            l = [np.zeros(states_dim), np.zeros(actions_dim)] + [np.zeros(states_dim)]
            H = collections.deque(l, len(l))
            H.append(np.zeros(actions_dim))
            H.append(state)
        episode_score = 0.
        done = False
        #####
        for t in count(0):
            state = torch.FloatTensor(state).unsqueeze(0)
            #######
            with torch.no_grad():
                inp = make_actor_input(state, varpi, algo_name, continuous)
                if "ddpg" in algo_name:
                    action = to_numpy(actor(*inp)).squeeze(axis=0)
                elif "sac" in algo_name:
                    action = to_numpy(actor(*inp)[-1]).squeeze(axis=0)#
            ########
            new_state, reward, done, info = env.step(action)
            episode_score += reward
            ##
            state = new_state
            ########
            if str_varpi == "osi":
                H.append(action)
                H.append(new_state)
                with torch.no_grad():
                    est_params, _ = osi_net.estimate_alpha(torch.FloatTensor(np.concatenate(list(H))).unsqueeze(0))
                    ####
                    est_params = est_params.squeeze(0).numpy()
                v, v_idx, _ = subdomains.get_point_domain_id(point=est_params)
                varpi[:, v_idx] += 1
            ########
            if done or (t+1) == env._max_episode_steps:
                break
        #####
        episode_scores.append(episode_score)
    #####
    print("########")
    if varpi is not None:
        varpi = varpi.numpy()
        if not continuous:
            varpi = varpi / np.sum(varpi)
        varpi = np.round(varpi, 2)
        if str_varpi != "osi":
            str_varpi = np.round(str_varpi.numpy(), 2)
    #####
    episode_return_mean = np.mean(episode_scores)
    episode_return_std = np.std(episode_scores)
    #####
    print(u"[TEST:{}; {}] Average Reward per Runs: {} \u00B1 {}".format(
          algo_name, varpi,
          episode_return_mean, episode_return_std
    ))
    #####
    if data is not None:
        #data["Varpi"] = [str_varpi]
        data["Env. Returns"] = episode_scores
    #####
    return data

################################################################################
### Run
def run_episodes_decay_uncertainty(algo_name, n_runs, n_episodes_per_run, env, actor, final_varpi, n_regions, continuous, lb=np.array([0.5, 0.5]), ub=np.array([2., 2.]), osi_net=None, data=None):
    #####
    env_name = str(env).split('<')[-1].split('>')[0]
    if "DClaw" in env_name:
        if "Fixed" in env_name:
            n_runs = 1
        episode_finalpos = [[] for _ in range(n_runs)]
    #####
    episode_scores = [[] for _ in range(n_runs)]
    #####
    final_varpi = torch.FloatTensor(final_varpi)
    #####
    ## Initialize uncertainty decay
    if osi_net is None:
        udecay = 1. / (n_episodes_per_run-1.)
    else:
        history_length = 10
        states_dim, actions_dim = env.observation_space.shape[0], env.action_space.shape[0]
        #####
        mu_list, std_list = [], []
    #####
    if "drsac" in algo_name or algo_name == "sac":
        orig_n = n_episodes_per_run
        n_episodes_per_run = 1
    #####
    for i in tqdm(range(n_runs)):
        varpi_list = []
        ## Initialize varpi
        varpi = np.concatenate((0.5 * (ub + lb), np.sqrt(1./12.) * (ub - lb)))
        initial_varpi = torch.FloatTensor(varpi)
        varpi = torch.FloatTensor(varpi)
        ####
        for j in range(n_episodes_per_run):
            if "DClaw" in env_name:
                env.robot.random_state = np.random.RandomState(123*(i+1))
            state, _ = env.reset(seed=123*(i+1))
            varpi_list.append(varpi.squeeze().numpy())#(np.round(varpi.numpy(), 2))
            #### Check support
            if osi_net is None:
                mu, sig = np.split(varpi.numpy(), 2, axis=-1)
                p, _ = np.split(final_varpi.numpy(), 2, axis=-1)
                assert (p >= mu - np.sqrt(3.) * sig).all() and (p <= mu + np.sqrt(3.) * sig).all(), "{} <= {} <= {}?".format(mu - np.sqrt(3.) * sig, p, mu + np.sqrt(3.) * sig)
            else:
                l = history_length*[np.zeros(states_dim), np.zeros(actions_dim)] + [np.zeros(states_dim)]
                H = collections.deque(l, len(l)); H.append(np.zeros(actions_dim)); H.append(state)
            ####
            episode_score = 0.
            done = False
            #####
            for t in count(0):
                state = torch.FloatTensor(state).unsqueeze(0)
                #######
                with torch.no_grad():
                    inp = make_actor_input(state, varpi, algo_name, continuous)
                    if "ddpg" in algo_name:
                        action = to_numpy(actor(*inp)).squeeze(axis=0)
                    elif "sac" in algo_name:
                        action = to_numpy(actor(*inp)[-1]).squeeze(axis=0)#
                ########
                new_state, reward, done, _, info = env.step(action)
                episode_score += reward
                ########
                if osi_net is not None:
                    H.append(action)
                    H.append(new_state)
                    if not (list(H)[0] == 0.).all():
                        Hist = torch.FloatTensor(np.concatenate(list(H), axis=-1)).unsqueeze(0)
                        with torch.no_grad():
                            _, p_dist, _ = osi_net(Hist)
                            mu_list.append(p_dist.mean.detach().squeeze())
                            std_list.append(p_dist.stddev.detach().squeeze())
                ########
                state = new_state
                ########
                if done or (t+1) == env._max_episode_steps:
                    break
            #####
            episode_scores[i].append(episode_score)
            if "DClaw" in env_name:
                episode_finalpos[i].append(abs(state[-1]))
            #####
            ## Update varpi
            if osi_net is None:
                beta = min(max(udecay * (j+1), 0.), 1.)
                varpi = initial_varpi.mul(1. - beta).add(final_varpi, alpha=beta)
            else:
                N = len(mu_list)
                mix = Categorical(torch.ones(N))
                comp = Independent(
                    Normal(
                        torch.stack(mu_list),
                        torch.stack(std_list)
                    ), 1
                )
                gmm = MixtureSameFamily(mix, comp)
                varpi = torch.cat(
                    (gmm.mean, gmm.variance.sqrt()), dim=-1
                )
    #####
    print("########")
    varpi = varpi.numpy()
    if not continuous:
        varpi = varpi / np.sum(varpi)
    varpi = np.round(varpi, 2)
    if osi_net is not None:
        print("True Varpi: ", np.round(final_varpi.numpy(), 2))
    print("Varpi: ", varpi)
    #print(np.array(varpi_list))
    if "mdsac" in algo_name or "sirsa" in algo_name:
        path = "./results/test_target_results/osi_varpi_evol/"
        make_Dirs(path)
        #varpi_list = np.stack(varpi_list)
        #print(varpi_list, data["Actor seed"][0])
        np.savetxt(path + "{}_{}_seed{}.txt".format(algo_name, env_name, data["Actor seed"][0]), varpi_list)
    #####
    if "drsac" in algo_name or algo_name == "sac":
        #print(orig_n, episode_scores, [score * orig_n for score in episode_scores])
        episode_scores = [score * orig_n for score in episode_scores]
        episode_return_mean = np.mean(episode_scores, axis=0)
        episode_return_std = np.std(episode_scores, axis=0)
        if "DClaw" in env_name:
            episode_finalpos = [score * orig_n for score in episode_finalpos]
            episode_finalpos_mean = np.mean(episode_finalpos, axis=0)
            episode_finalpos_std = np.std(episode_finalpos, axis=0)
    else:
        episode_return_mean = np.mean(episode_scores, axis=0)
        episode_return_std = np.std(episode_scores, axis=0)
        if "DClaw" in env_name:
            episode_finalpos_mean = np.mean(episode_finalpos, axis=0)
            episode_finalpos_std = np.std(episode_finalpos, axis=0)
    #####
    result_string = u"[TEST:{}; {}] Average Reward per Runs: {} \u00B1 {}".format(
          algo_name, varpi,
          episode_return_mean, episode_return_std
    ) + ("; Avg. final pos: {} \u00B1 {}".format(episode_finalpos_mean, episode_finalpos_std) if "DClaw" in env_name else "")
    print(result_string)
    #####
    if data is not None:
        #data["Varpi"] = [str_varpi]
        data["Env. Returns"] = episode_scores
        if "DClaw" in env_name:
            data["Env. Final Pos."] = episode_finalpos
    #####
    return data

################################################################################
### Run with environment set
def run_ccs_episodes(algo_name, n_runs, env, actor, varpi, continuous, data=None, cuda=True, cuda_id=2):
    episode_scores = []
    episode_ccs_scores = []
    orig_n_runs = n_runs
    ####
    env_name = str(env.env_set[0]).split('<')[-1].split('>')[0]
    if "DClaw" in env_name:
        episode_final_pos = []
        episode_ccs_final_pos = []
        if "Fixed" in env_name:
            n_runs = 1
    #####
    if cuda:
        device = torch.device("cuda:{}".format(cuda_id))
        actor = actor.to(device)
    else:
        device = torch.device("cpu")
    #####
    for i in range(n_runs):
        state = env.reset()
        inp_varpi = varpi.repeat(env.n_regions, axis=0)
        #print(state.shape, inp_varpi.shape)
        episode_score = np.zeros(env.n_regions)
        episode_finalpos = np.zeros(env.n_regions) + np.inf
        done = False
        #####
        for t in count(0):
            #######
            pre_not_dones = ~env.get_dones()
            with torch.no_grad():
                inp = make_actor_input(state, inp_varpi, algo_name, continuous, device)
                if "ddpg" in algo_name:
                    action = to_numpy(actor(*inp))#.squeeze(axis=0)
                elif "sac" in algo_name or "sirsa" in algo_name:
                    action = to_numpy(actor(*inp)[-1])#.squeeze(axis=0)#
            ########
            new_state, reward, done, _, _ = env.step(action)
            ########
            episode_score[pre_not_dones] += reward
            if "DClaw" in env_name:
                episode_finalpos[pre_not_dones] = abs(new_state[:, -1])
            ##
            state = new_state[~done]
            inp_varpi = inp_varpi[~done]
            if (done == True).all() or (t+1) == env._max_episode_steps:
                #if "DClaw" in env_name: print(episode_finalpos)
                break
        #####
        if continuous:
            episode_ccs_scores.append(episode_score.mean())#(estimate_cvar(episode_score, 0.005))#
            if "DClaw" in env_name:
                episode_ccs_final_pos.append(episode_finalpos.mean())#(estimate_cvar(episode_finalpos, 0.005))#
        else:
            episode_ccs_scores.append(varpi.dot(episode_score).squeeze(0))
        episode_scores.append(episode_score)
        if "DClaw" in env_name:
            episode_final_pos.append(episode_finalpos)
    #####
    if varpi is not None:
        varpi = np.round(varpi, 2)
    #####
    #episode_scores = np.array(episode_scores)
    episode_scores_mean = np.mean(episode_scores, axis=0)
    episode_scores_std = np.std(episode_scores, axis=0)
    ###
    episode_ccs_scores_mean = np.mean(episode_ccs_scores)
    episode_ccs_scores_std = np.std(episode_ccs_scores)
    episode_ccs_scores_iqm = interquartile_mean(episode_ccs_scores)
    #####
    if "DClaw" in env_name:
        episode_finalpos_mean = np.mean(episode_final_pos, axis=0)
        episode_finalpos_std = np.std(episode_final_pos, axis=0)
        ###
        episode_ccs_finalpos_mean = np.mean(episode_ccs_final_pos)
        episode_ccs_finalpos_std = np.std(episode_ccs_final_pos)
        episode_ccs_finalpos_iqm = interquartile_mean(episode_ccs_final_pos)
    #####
    print(u"[TEST:{}; {}] Average Reward per Runs: {} \u00B1 {}; {} \u00B1 {}; IQM: {}; {}".format(
          algo_name, varpi,
          episode_ccs_scores_mean, episode_ccs_scores_std,
          episode_ccs_finalpos_mean, episode_ccs_finalpos_std,
          episode_ccs_scores_iqm, episode_ccs_finalpos_iqm,
    ) if "DClaw" in env_name else u"[TEST:{}; {}] Average Reward per Runs: {} \u00B1 {}; IQM: {}".format(
          algo_name, varpi,
          episode_ccs_scores_mean, episode_ccs_scores_std,
          episode_ccs_scores_iqm,
    ))
    #####
    if data is not None:
        #print(orig_n_runs, len(episode_scores), len(episode_ccs_scores), len(data))
        data["Varpi"] = [varpi]*n_runs
        data["Env. Returns"] = episode_scores
        data["CCS Return"] = episode_ccs_scores
        if "DClaw" in env_name:
            data["Env. FinalPos"] = episode_final_pos
            data["CCS FinalPos"] = episode_ccs_final_pos
    #####
    return data

################################################################################
### Run with environment set
def run_ccs_episodes_with_osi(algo_name, n_runs, env, actor, osi_net, varpi, continuous, data=None, cuda=True, cuda_id=2):
    episode_scores = []
    episode_ccs_scores = []
    ####
    env_name = str(env.env_set[0]).split('<')[-1].split('>')[0]
    if "DClaw" in env_name:
        episode_final_pos = []
        episode_ccs_final_pos = []
        if "Fixed" in env_name:
            n_runs = 1
    #####
    if cuda:
        device = torch.device("cuda:{}".format(cuda_id))
        actor = actor.to(device)
    else:
        device = torch.device("cpu")
    #####
    lb = torch.FloatTensor(env.latent_ranges[:, 0]).to(device)
    ub = torch.FloatTensor(env.latent_ranges[:, 1]).to(device)
    #####
    for i in range(n_runs):
        state = env.reset()
        inp_varpi = torch.FloatTensor(varpi.repeat(env.n_regions, axis=0)).to(device)
        #print(state.shape, inp_varpi.shape)
        episode_score = np.zeros(env.n_regions)
        episode_finalpos = np.zeros(env.n_regions) + np.inf
        done = False
        #####
        for t in count(0):
            #######
            pre_not_dones = ~env.get_dones()
            with torch.no_grad():
                inp = make_actor_input(state, inp_varpi, algo_name, continuous, device)
                if "ddpg" in algo_name:
                    action = to_numpy(actor(*inp))#.squeeze(axis=0)
                elif "sac" in algo_name or "sirsa" in algo_name:
                    action = to_numpy(actor(*inp)[-1])#.squeeze(axis=0)#
            ########
            new_state, reward, done, _, _ = env.step(action)
            ########
            episode_score[pre_not_dones] += reward
            if "DClaw" in env_name:
                episode_finalpos[pre_not_dones] = abs(new_state[:, -1])
            ##
            with torch.no_grad():
                inp = make_osi_input(state, action, new_state, inp_varpi, device)
                """if "sirsa" in algo_name:
                    predicted_sigmas = osi_net(*inp)
                    mu, sig = torch.split(inp_varpi, inp_varpi.shape[-1]//2, dim=-1)
                    inp_varpi = reconfigure_osi_varpis(mu, sig, predicted_sigmas, ub, lb)
                else:"""
                inp_varpi = osi_net.get_varpi(*inp, bounds=(lb, ub))
            ####
            state = new_state[~done]
            inp_varpi = inp_varpi[~done]
            if (done == True).all() or (t+1) == env._max_episode_steps:
                #if "DClaw" in env_name: print(episode_finalpos)
                break
        #####
        if continuous:
            episode_ccs_scores.append(episode_score.mean())#(estimate_cvar(episode_score, 0.005))#
            if "DClaw" in env_name:
                episode_ccs_final_pos.append(episode_finalpos.mean())#(estimate_cvar(episode_finalpos, 0.005))#
        else:
            episode_ccs_scores.append(varpi.dot(episode_score).squeeze(0))
        episode_scores.append(episode_score)
        if "DClaw" in env_name:
            episode_final_pos.append(episode_finalpos)
    #####
    if varpi is not None:
        varpi = np.round(varpi, 2)
    #####
    #episode_scores = np.array(episode_scores)
    episode_scores_mean = np.mean(episode_scores, axis=0)
    episode_scores_std = np.std(episode_scores, axis=0)
    ###
    episode_ccs_scores_mean = np.mean(episode_ccs_scores)
    episode_ccs_scores_std = np.std(episode_ccs_scores)
    episode_ccs_scores_iqm = interquartile_mean(episode_ccs_scores)
    #####
    if "DClaw" in env_name:
        episode_finalpos_mean = np.mean(episode_final_pos, axis=0)
        episode_finalpos_std = np.std(episode_final_pos, axis=0)
        ###
        episode_ccs_finalpos_mean = np.mean(episode_ccs_final_pos)
        episode_ccs_finalpos_std = np.std(episode_ccs_final_pos)
        episode_ccs_finalpos_iqm = interquartile_mean(episode_ccs_final_pos)
    #####
    print(u"[TEST:{}; {}] Average Reward per Runs: {} \u00B1 {}; {} \u00B1 {}; IQM: {}; {}".format(
          algo_name, varpi,
          episode_ccs_scores_mean, episode_ccs_scores_std,
          episode_ccs_finalpos_mean, episode_ccs_finalpos_std,
          episode_ccs_scores_iqm, episode_ccs_finalpos_iqm,
    ) if "DClaw" in env_name else u"[TEST:{}; {}] Average Reward per Runs: {} \u00B1 {}; IQM: {}".format(
          algo_name, varpi,
          episode_ccs_scores_mean, episode_ccs_scores_std,
          episode_ccs_scores_iqm,
    ))
    #####
    if data is not None:
        data["Varpi"] = [varpi]*n_runs
        data["Env. Returns"] = episode_scores
        data["CCS Return"] = episode_ccs_scores
        if "DClaw" in env_name:
            data["Env. FinalPos"] = episode_final_pos
            data["CCS FinalPos"] = episode_ccs_final_pos
    #####
    return data

################################################################################
def sample_ccs_varpi(rng, lb, ub):
    mu = lb + (ub - lb) * rng.uniform(size=(1, lb.shape[0]))
    std = rng.uniform(size=(1, lb.shape[0])) * np.min(((mu - lb), (ub - mu)), axis=0) / np.sqrt(3.)
    return np.concatenate((mu, std), axis=-1)#np.array([[mu1, mu2, std1, std2]])

################################################################################
def reconfigure_osi_varpis(mean, sigma, preds, ub, lb):
    new_varpi_elements = torch.stack([mean + np.sqrt(3.) * preds[i].clip(min=-sigma, max=sigma)
        for i in range(preds.shape[0])])
    mean, std = new_varpi_elements.mean(0), new_varpi_elements.std(0)
    std = torch.min(std, torch.min(ub - mean, mean - lb)/np.sqrt(3.))
    new_varpi = torch.cat((mean, std), dim=-1).detach()
    return new_varpi

################################################################################
def run_test_target(actor, output_path, env_name, algo_name, actor_seed, dim, osi_net=None,
        ccs=True, continuous=True, use_osi=False, save_results=True, use_encoder=False,
        seed=1234, n_runs=50, cuda=True, cuda_id=0):
    if use_osi:
        ccs = False
    if save_results:
        if "DClaw" in env_name:
            data = pd.DataFrame(columns=["Env. Params", "Algo", "Varpi",
                                         "Env. Returns", "CCS Return",
                                         "Env. FinalPos", "CCS FinalPos",
                                         "Actor seed"
            ])
            n_runs = 1
        else:
            data = pd.DataFrame(columns=["Env. Params", "Algo", "Varpi",
                                         "Env. Returns", "CCS Return",
                                         "Actor seed"
            ])
    else:
        data = None
    ######
    multi_dim = (dim > 2)
    ######
    n_regions = 2 * dim + 1#50
    sigma_points = fetch_uniform_unscented_transfo(dim=dim)["sigma_points"]
    if ccs:
        rnd_gen = np.random.RandomState(seed*3)
        #######
        env = MDEnvSet(env_name=env_name, multi_dim=multi_dim, n_regions=n_regions, seed=seed, use_encoder=use_encoder)
        env_dims = (env.observation_space.shape[0], env.action_space.shape[0], env.subdomains.domain_dim)
        env.seed(seed)
        _ = env.reset()
        filename_suffix = ""
        #######
        if continuous:
            lb = env.latent_ranges[:, 0]
            ub = env.latent_ranges[:, 1]
            print(lb.shape, ub.shape)
            varpi1 = sample_ccs_varpi(rnd_gen, lb, ub)
            varpi2 = sample_ccs_varpi(rnd_gen, lb, ub)
            varpi3 = sample_ccs_varpi(rnd_gen, lb, ub)
            varpi4 = sample_ccs_varpi(rnd_gen, lb, ub)
            varpi5 = sample_ccs_varpi(rnd_gen, lb, ub)
            varpi6 = sample_ccs_varpi(rnd_gen, lb, ub)
            varpi7 = sample_ccs_varpi(rnd_gen, lb, ub)
            varpi8 = sample_ccs_varpi(rnd_gen, lb, ub)
            varpi9 = sample_ccs_varpi(rnd_gen, lb, ub)
            varpi10 = sample_ccs_varpi(rnd_gen, lb, ub)
            #varpi11 = sample_ccs_varpi(rnd_gen, lb, ub)
            if not use_encoder:
                varpi_list = []##[np.concatenate((np.ones(lb.shape).reshape(1, -1), np.zeros(lb.shape).reshape(1, -1)), axis=-1)] # Nominal environment
            else:
                if "Pendulum" in env_name:
                    varpi_list = [np.array([[0.61613655, 0.38670707, 0., 0.]])] # Nominal environment encoding
                elif "Hopper" in env_name:
                    varpi_list = [np.array([[0.07886374, 0.42413795, 0., 0.]])] # Nominal environment encoding
                elif "Ant" in env_name:
                    varpi_list = [np.array([[-7.280368, -1.3467762, 0., 0.]])] # Nominal environment encoding
                else:
                    raise NotImplementedError
            #"""
            varpi_list += [
                 np.concatenate((0.5 * (ub + lb).reshape(1, -1), np.sqrt(1./12.) * (ub - lb).reshape(1, -1)), axis=-1),
                 varpi1,
                 varpi2,
                 varpi5,
                 varpi10,
            ]
            #"""
            varpi_list += [
                varpi3,
                varpi4,
                varpi6,
                varpi7,
                varpi8,
                varpi9
            ]
            #"""
            try:
                rndparams = sigma_points[1]#rnd_gen.uniform(size=(50, lb.shape[0]))
            except IndexError as e:
                rndparams = sigma_points[0]
            _ = env.sample_env_params(use_varpi=False)
        else:
            nominal_varpi, _, _ = env.subdomains.get_point_domain_id(point=np.ones_like(env.ub_params))
            varpi_list = [np.array([nominal_varpi]), np.array([np.eye(n_regions)[-1]]),
                 np.ones((1, n_regions)) / n_regions,
                 rnd_gen.dirichlet(np.zeros(n_regions) + 0.5, size=1),
                 rnd_gen.dirichlet(np.zeros(n_regions) + 0.5, size=1)
            ]
            rndparams = env.sample_env_params(use_varpi=False)
        print("---------------------")
        print("---------------------")
        N_v = len(varpi_list)
        for nth_v, varpi in enumerate(varpi_list):
            print("########")
            print(nth_v+1, " / ", N_v)
            if continuous:
                pmu, pstd = np.split(varpi, 2, axis=-1)
                envparams = pmu + (2. * rndparams - 1.) * np.sqrt(3.) * pstd
                env.env_params[:] = envparams
                env.change_all_env_params(env.env_params)
            #####
            if data is not None:
                df = pd.DataFrame(columns=data.columns)
                if continuous:
                    df["Env. Params"] = [envparams]*n_runs
                else:
                    df["Env. Params"] = [rndparams]*n_runs
                df["Algo"] = [algo_name]*n_runs
                df["Varpi"] = [varpi]*n_runs
                df["Actor seed"] = [actor_seed]*n_runs
            else:
                df = None
            ########
            env.seed(seed)
            df = run_ccs_episodes(algo_name, n_runs, env, actor, varpi, continuous, data=df, cuda=cuda, cuda_id=cuda_id)
            ########
            if data is not None:
                data = pd.concat([data, df], ignore_index=True, sort=True)
        ######
        env.close()
    elif use_osi:
        assert osi_net is not None
        rnd_gen = np.random.RandomState(seed*3)
        #######
        env = MDEnvSet(env_name=env_name, multi_dim=multi_dim, n_regions=n_regions, seed=seed, use_encoder=use_encoder)
        env_dims = (env.observation_space.shape[0], env.action_space.shape[0], env.subdomains.domain_dim)
        env.seed(seed)
        _ = env.reset()
        filename_suffix = ""
        #######
        if continuous:
            lb = env.latent_ranges[:, 0]
            ub = env.latent_ranges[:, 1]
            print(lb.shape, ub.shape)
            varpi1 = sample_ccs_varpi(rnd_gen, lb, ub)
            varpi2 = sample_ccs_varpi(rnd_gen, lb, ub)
            varpi3 = sample_ccs_varpi(rnd_gen, lb, ub)
            varpi4 = sample_ccs_varpi(rnd_gen, lb, ub)
            varpi5 = sample_ccs_varpi(rnd_gen, lb, ub)
            varpi6 = sample_ccs_varpi(rnd_gen, lb, ub)
            varpi7 = sample_ccs_varpi(rnd_gen, lb, ub)
            varpi8 = sample_ccs_varpi(rnd_gen, lb, ub)
            varpi9 = sample_ccs_varpi(rnd_gen, lb, ub)
            varpi10 = sample_ccs_varpi(rnd_gen, lb, ub)
            #varpi11 = sample_ccs_varpi(rnd_gen, lb, ub)
            if not use_encoder:
                varpi_list = []##[np.concatenate((np.ones(lb.shape).reshape(1, -1), np.zeros(lb.shape).reshape(1, -1)), axis=-1)] # Nominal environment
            else:
                if "Pendulum" in env_name:
                    varpi_list = [np.array([[0.61613655, 0.38670707, 0., 0.]])] # Nominal environment encoding
                elif "Hopper" in env_name:
                    varpi_list = [np.array([[0.07886374, 0.42413795, 0., 0.]])] # Nominal environment encoding
                elif "Ant" in env_name:
                    varpi_list = [np.array([[-7.280368, -1.3467762, 0., 0.]])] # Nominal environment encoding
                else:
                    raise NotImplementedError
            #"""
            varpi_list += [
                 np.concatenate((0.5 * (ub + lb).reshape(1, -1), np.sqrt(1./12.) * (ub - lb).reshape(1, -1)), axis=-1),
                 varpi1,
                 varpi2,
                 varpi5,
                 varpi10,
            ]
            #"""
            varpi_list += [
                varpi3,
                varpi4,
                varpi6,
                varpi7,
                varpi8,
                varpi9
            ]
            #"""
            try:
                rndparams = sigma_points[1]#rnd_gen.uniform(size=(50, lb.shape[0]))
            except IndexError as e:
                rndparams = sigma_points[0]
            assert (rndparams > 0.).all()
            assert (rndparams < 1.).all()
            _ = env.sample_env_params(use_varpi=False)
        else:
            nominal_varpi, _, _ = env.subdomains.get_point_domain_id(point=np.ones_like(env.ub_params))
            varpi_list = [np.array([nominal_varpi]), np.array([np.eye(n_regions)[-1]]),
                 np.ones((1, n_regions)) / n_regions,
                 rnd_gen.dirichlet(np.zeros(n_regions) + 0.5, size=1),
                 rnd_gen.dirichlet(np.zeros(n_regions) + 0.5, size=1)
            ]
            rndparams = env.sample_env_params(use_varpi=False)
        print("---------------------")
        print("---------------------")
        N_v = len(varpi_list)
        for nth_v, varpi in enumerate(varpi_list):
            print("########")
            print(nth_v+1, " / ", N_v)
            if continuous:
                pmu, pstd = np.split(varpi, 2, axis=-1)
                envparams = pmu + (2. * rndparams - 1.) * np.sqrt(3.) * pstd
                env.env_params[:] = envparams
                env.change_all_env_params(env.env_params)
            #####
            if data is not None:
                df = pd.DataFrame(columns=data.columns)
                if continuous:
                    df["Env. Params"] = [envparams]*n_runs
                else:
                    df["Env. Params"] = [rndparams]*n_runs
                df["Algo"] = [algo_name]*n_runs
                df["Varpi"] = [varpi]*n_runs
                df["Actor seed"] = [actor_seed]*n_runs
            else:
                df = None
            ########
            env.seed(seed)
            df = run_ccs_episodes_with_osi(algo_name, n_runs, env, actor, osi_net, varpi, continuous, data=df, cuda=cuda, cuda_id=cuda_id)
            ########
            if data is not None:
                data = pd.concat([data, df], ignore_index=True, sort=True)
        ######
        env.close()
    else:
        if save_results:
            """data = pd.DataFrame(columns=["Env. Params", "Algo",
                                         "Env. Returns Mean", "Env. Returns Std",
                                         "Actor seed"
            ] + (["Env. Final Pos. Mean", "Env. Final Pos. Std"] if "DClaw" in env_name else []))"""
            data = pd.DataFrame(columns=["Env. Params", "Algo",
                                         "Env. Returns", "Actor seed"
            ] + (["Env. Final Pos."] if "DClaw" in env_name else []))
        #######
        env, rndparams, lb, ub, filename_suffix = make_target_env(env_name, seed, multi_dim, change_integrator=False, use_encoder=False)
        print("---------------------")
        print("environment parameters:")
        print(rndparams, lb, ub)
        print("---------------------")
        final_varpi = np.array([[v for (k, v) in rndparams.items()]])
        final_varpi = np.concatenate((final_varpi, np.zeros_like(final_varpi)), axis=-1)
        print("Final varpi:")
        print(final_varpi)
        env_dims = (env.observation_space.shape[0], env.action_space.shape[0], lb.shape[0])
        ########
        if data is not None:
            df = pd.DataFrame(columns=data.columns)
            df["Env. Params"] = [np.array([rndparams['x'], rndparams['y']])]*n_runs
            df["Algo"] = [algo_name]*n_runs
            df["Actor seed"] = [actor_seed]*n_runs
        else:
            df = None
        ########
        if "DClaw" in env_name:
            env.reset()
        else:
            env.reset(seed=seed)
        df = run_episodes_decay_uncertainty(algo_name, n_runs, 10, env, actor, final_varpi, n_regions, continuous, lb, ub, osi_net, df)
        ########
        if data is not None:
            data = pd.concat([data, df], ignore_index=True, sort=True)
        ######
        env.close()
    #########
    print("##########################")
    print("++ Result data")
    print(data)
    if save_results:
        filename = "{}_test_target".format(env_name.split("-")[0]) + ("_ccs" if ccs else ("_osi" if use_osi else ""))
        filename += ("_ut_on_ut_cont_no_her" if continuous else "") + "_rndseed{}{}_{}.csv".format(seed, filename_suffix, algo_name)
        mode = "w"
        print("++ Save results to ", output_path + filename)
        data.to_csv(output_path + filename, mode=mode, header=(mode=="w" or not os.path.isfile(output_path + filename)))
        """if ccs:
            df = data.loc[data["Algo"].isin([algo_name])]
            df.to_csv(output_path + filename.replace("_all.csv", "_{}.csv".format(algo_name)), mode=mode, header=(mode=="w" or not os.path.isfile(output_path + filename)))"""
    print("##########################")
