import copy
import random
import numpy as np

import gymnasium as gym
import pybulletgym
import robel
#import custom_gym
from pytorch_rl_collection.utils import DotDict
from pytorch_rl_collection.envs_utils.normalized_env import NormalizedEnv

from pytorch_rl_collection.utils import ut_uniform, SubDomains, rsetattr, rgetattr, MultiDimSubDomains#, QuasiRandomGenerator

import yaml
#from collections import Sequence

from torch import FloatTensor, no_grad
from torch import load as torch_load
from pytorch_rl_collection.model_networks.model_networks import Decoder

###########################################################################################################################
###########################################################################################################################
def rotate(v, ang):
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return R @ v


class MDEnvSet:
    def __init__(self, env_name, multi_dim, n_regions=10, seed=None, use_encoder=True, add_noise=False, use_quasi_random=False, linear=False):
        ###### Read config file
        config_filename = env_name.split("-")[0] + ".yaml"
        if "DClaw" in config_filename:
            config_filename = "DClaw.yaml"
            self.observation_noise = 5e-3 if add_noise else None
            self.observation_keys = (
               'claw_qpos',
               'object_x',
               'object_y',
               'last_action',
               'target_error',
            )
            print("++ Observation noise: ", self.observation_noise)
        else:
            self.observation_noise = None
        with open("./pytorch_rl_collection/configs/" + config_filename, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        ######
        assert isinstance(multi_dim, bool), "Expected 'multi_dim' as Boolean, got {} with type {}".format(multi_dim, type(multi_dim))
        if multi_dim:
            config = config["multi_dim"]
        else:
            config = config["2d"]
            linear = False
        ######
        self.params_idxs = config["params_idxs"]
        self.params_env_idxs = config["params_env_idxs"]
        self.params_relative_ranges = config["params_relative_ranges"]
        ######
        print("++ Randomized params: ", self.params_idxs.keys())
        ######
        self.env_set = []
        self.action_space = None
        self.observation_space = None
        self._max_episode_steps = None
        ######
        self._ep_dones = None
        #######
        self._linear = linear
        self._env_name = env_name
        if "Pendulum" in env_name:
            self._max_episode_steps = 200
        elif "DClaw" in env_name:
            self._max_episode_steps = 40
        else:
            self._max_episode_steps = 1000
        param_ranges = self.initialize_envs(env_name, n_regions, self._linear)
        #######
        try:
            if "mujoco" in str(type(self.env_set[0].env.env.env.env)):
                if getattr(self.env_set[0].unwrapped, "sim", None) is None:
                    self.reset = self._mujoco_reset
                else:
                    self.reset = self._mujocopy_reset
            else:
                self.reset = self._mujocopy_reset
        except AttributeError:
            self.reset = self._mujocopy_reset
        #######
        if seed is not None:
            self.seed(seed)
        #######
        self.use_encoder = use_encoder
        if self.use_encoder:
            self.latent_ranges = np.array([config["latent_ranges"]["x"], config["latent_ranges"]["y"]])
            self.subdomains = SubDomains(n_regions, self.latent_ranges, seed=seed, verbose=False, use_quasi_random=use_quasi_random)
            self._latent_decoder = Decoder(nb_envparams=param_ranges.shape[0], nb_latent=2,
                    lb=FloatTensor(param_ranges[:, 0]), ub=FloatTensor(param_ranges[:, 1]),
                    hidden_layers=[200, 300], init_w=3e-3, verbose=False
            )
            #print(self._latent_decoder._ub)
            self._latent_decoder.load_state_dict(
                torch_load('./trained_agents/params_encoding_res/{}/decoder.pkl'.format(env_name))
            )
        else:
            if not self._linear:
                env_key_list = list(self.params_idxs.keys())
                self.latent_ranges = []
                safety_check_params_idx = None
                for key, elt in self.params_idxs.items():
                    if safety_check_params_idx is None:
                        safety_check_params_idx = elt
                    else:
                        assert safety_check_params_idx != elt, "Params index repeatition detected {}".format(elt)
                    if "body_pos" in key:
                        self.latent_ranges.append(np.array(self.params_relative_ranges[key + "_angle"]).reshape(1, -1))
                        self.latent_ranges.append(np.array(self.params_relative_ranges[key + "_radius"]).reshape(1, -1))
                    else:
                        self.latent_ranges.append(np.array(self.params_relative_ranges[key]).reshape(1, -1).repeat(len(elt), axis=0))
                #####
                self.latent_ranges = np.concatenate(self.latent_ranges, axis=0)
            else:
                self.latent_ranges = param_ranges
            #######
            if self.latent_ranges.shape[0] == 2:
                self.subdomains = SubDomains(n_regions, self.latent_ranges, seed=seed, verbose=False, use_quasi_random=use_quasi_random)
            else:
                self.subdomains = MultiDimSubDomains(n_regions, self.latent_ranges, seed=seed, verbose=False, use_quasi_random=use_quasi_random)
            print(self.subdomains)
        #######
        self.set_varpi_ranges(self.subdomains.ranges)
        #######
        self.rand_seed = seed

    def initialize_envs(self, env_name, n_regions, linear=False):
        assert n_regions > 0
        self.env_set = []
        for i in range(n_regions):
            if self.observation_noise is None:
                env = gym.make(env_name)#, observation_keys=self.observation_keys)
                if getattr(env, "_max_episode_steps", None) is None:
                    env = gym.wrappers.TimeLimit(env, max_episode_steps=self._max_episode_steps)
                else:
                    self._max_episode_steps = env._max_episode_steps
                self.env_set.append(NormalizedEnv(env))
            else:
                self.env_set.append(NormalizedEnv(gym.make(env_name, sim_observation_noise=self.observation_noise, observation_keys=self.observation_keys)))
        ######
        if self.action_space is None:
            self.action_space = self.env_set[0].action_space
            self.observation_space = self.env_set[0].observation_space
            self._max_episode_steps = self.env_set[0]._max_episode_steps
            #####
            self.nominal_params = {}
            self.lb_params, self.ub_params = [], []
            if env_name == "Pendulum-v1":
                #self._env_masses = 1.0 * self.env_set[0].m
                for key in self.params_relative_ranges.keys():
                    lb, ub = self.params_relative_ranges[key]
                    params = rgetattr(self.env_set[0], key, None)
                    self.lb_params += [lb]#[lb * params]
                    self.ub_params += [ub]#[ub * params]
                    self.nominal_params[key] = float(params)
            else:
                #if getattr(self.env_set[0].unwrapped, "model", None) is not None:
                #    self._env_masses = copy.deepcopy(self.env_set[0].unwrapped.model.body_mass.copy())
                for key in self.params_relative_ranges.keys():
                    lb, ub = self.params_relative_ranges[key]
                    ##
                    if "body_pos" in key:
                        self.nominal_params[key] = 0.5232783221319756 if "angle" in key else 0.06003332407921454
                        self.lb_params += [lb] if not linear else [lb * self.nominal_params[key]]
                        self.ub_params += [ub] if not linear else [ub * self.nominal_params[key]]
                    else:
                        try:
                            params = rgetattr(self.env_set[0].unwrapped.sim.model, key, None)
                        except AttributeError:
                            params = rgetattr(self.env_set[0].unwrapped.model, key, None)
                        if isinstance(params, float):
                            #self.nominal_params += [params]
                            self.nominal_params[key] = float(params)
                            self.lb_params += [lb] if not linear else [lb * self.nominal_params[key]]
                            self.ub_params += [ub] if not linear else [ub * self.nominal_params[key]]
                        else:
                            idxs = self.params_env_idxs[key]
                            take_tuple = False
                            for elt in idxs:
                                if isinstance(elt, list):
                                    take_tuple = True
                                    break
                            if take_tuple:
                                idxs = tuple(idxs)
                            params = params[idxs]
                            #self.nominal_params += list(params)
                            self.nominal_params[key] = np.copy(params)
                            self.lb_params += list(lb * np.ones_like(params)) if not linear else list(lb * self.nominal_params[key])
                            self.ub_params += list(ub * np.ones_like(params)) if not linear else list(ub * self.nominal_params[key])
            #####
            param_ranges = np.array([self.lb_params, self.ub_params]).T
            self.n_regions = n_regions
            #assert len(param_ranges.shape) == 2
            #assert param_ranges.shape[-1] == 2
            ######
            self.lb_params, self.ub_params = np.array(self.lb_params), np.array(self.ub_params)
        #####
        return param_ranges

    def _get_idx(self, orig_list, val):
        val_ns, val_r, val_d = val
        for i, (ns, r, d) in enumerate(orig_list):
            if (ns == val_ns).all() and (r == val_r) and (d == val_d):
                return i
        return None

    def _key_function(self, iter):
        ## next_state, reward, done, info
        return iter[1] # compare list elements on the reward, i.e. on the index 1

    def push(self, new_env):
        if self.rand_seed is not None:
            new_env.seed(self.rand_seed)
        self.env_set.append(new_env)
        self.n_regions += 1
        if self.action_space is None:
            self.action_space = new_env.action_space
            self.observation_space = new_env.observation_space
            self._max_episode_steps = new_env._max_episode_steps
            #####
            if self._env_name == "Pendulum-v1":
                self._env_masses = 1.0 * new_env.m
            else:
                if getattr(new_env.unwrapped, "model", None) is not None:
                    self._env_masses = copy.deepcopy(new_env.unwrapped.model.body_mass.copy())

    def seed(self, seed, same=True):
        random.seed(seed)
        if len(self.env_set) > 0:
            for i, env in enumerate(self.env_set):
                try:
                    env.reset(seed=(int(not same) * i+1) * seed)
                except TypeError as err:
                    env.seed(seed=(int(not same) * i+1) * seed)
                if self.observation_noise is not None:
                    env.robot.random_state, _ = gym.utils.seeding.np_random((int(not same) * i+1) * seed)

    def change_env_params(self, env, params):
        if params is None:
            if self.use_encoder:
                params = np.ones_like(self.ub_params)
            elif not self._linear:
                params = np.ones(self.subdomains.domain_dim)
            else:
                params = []
                for v in self.nominal_params.values():
                    if not isinstance(v, np.ndarray):
                        params += [v]
                    else:
                        params += list(v)
                params = np.array(params)
        else:
            if self.use_encoder:
                with no_grad():
                    params = self._latent_decoder(FloatTensor(params).unsqueeze(0)).squeeze().numpy()
        ###############
        keys = self.params_idxs.keys()
        ###############
        if self._env_name == "Pendulum-v1":
            #env.env.env.env.env.m = relative_mass * self._env_masses[idx]
            for key in keys:
                if not self._linear:
                    rsetattr(env.env.env.env.env, key, params[self.params_idxs[key][0]] * self.nominal_params[key])
                else:
                    rsetattr(env.env.env.env.env, key, params[self.params_idxs[key][0]])
        elif getattr(env.unwrapped, "model", None) is not None:
            #for idx in range(len(self._env_masses)):
            #    env.unwrapped.model.body_mass[idx] = relative_mass * self._env_masses[idx]
            for key in keys:
                idxs = self.params_env_idxs[key]
                take_tuple = False
                for elt in idxs:
                    if isinstance(elt, list):
                        take_tuple = True
                        break
                if take_tuple:
                    idxs = tuple(idxs)
                if len(idxs) > 0:
                    ##
                    if "body_pos" in key:
                        if not self._linear:
                            orig_ang, orig_radius = self.nominal_params[key + "_angle"], self.nominal_params[key + "_radius"]
                            coef_orig_ang, coef_orig_radius = params[self.params_idxs[key]]
                            v_ffbase = rotate(v=np.array([0., -orig_radius*coef_orig_radius]), ang=orig_ang*coef_orig_ang)
                        else:
                            new_ang, new_radius = params[self.params_idxs[key]]
                            v_ffbase = rotate(v=np.array([0., -new_radius]), ang=new_ang)
                        v_mfbase = rotate(v=v_ffbase, ang=2.*np.pi/3.)
                        v_tfbase = rotate(v=v_ffbase, ang=4.*np.pi/3.)
                        ###
                        new_vals = np.array([np.round(v_ffbase, 3), np.round(v_mfbase, 3), np.round(v_tfbase, 3)]*2)
                        rgetattr(env.unwrapped.sim.model, key, None)[idxs] = new_vals
                        ###
                    else:
                        if not self._linear:
                            try:
                                rgetattr(env.unwrapped.sim.model, key, None)[idxs] = params[self.params_idxs[key]] * self.nominal_params[key]
                                if "actuator_gainprm" in key:
                                    idxs = (idxs[0], 1)
                                    rgetattr(env.unwrapped.sim.model, key.replace("actuator_gainprm", "actuator_biasprm"), None)[idxs] = - params[self.params_idxs[key]] * self.nominal_params[key]
                            except AttributeError:
                                rgetattr(env.unwrapped.model, key, None)[idxs] = params[self.params_idxs[key]] * self.nominal_params[key]
                        else:
                            try:
                                rgetattr(env.unwrapped.sim.model, key, None)[idxs] = params[self.params_idxs[key]]
                                if "actuator_gainprm" in key:
                                    idxs = (idxs[0], 1)
                                    rgetattr(env.unwrapped.sim.model, key.replace("actuator_gainprm", "actuator_biasprm"), None)[idxs] = - params[self.params_idxs[key]]
                            except AttributeError:
                                rgetattr(env.unwrapped.model, key, None)[idxs] = params[self.params_idxs[key]]
                else:
                    if not self._linear:
                        try:
                            rsetattr(env.unwrapped.sim.model, key, params[self.params_idxs[key]] * self.nominal_params[key])
                        except AttributeError:
                            rsetattr(env.unwrapped.model, key, params[self.params_idxs[key]] * self.nominal_params[key])
                    else:
                        try:
                            rsetattr(env.unwrapped.sim.model, key, params[self.params_idxs[key]])
                        except AttributeError:
                            rsetattr(env.unwrapped.model, key, params[self.params_idxs[key]])

    def getState(self, env):
        return np.copy(env.env.env.state)

    def setState(self, qpos, qvel, env):
        env.env.env.state[:] = np.array([qpos, qvel])

    def bullet_getState(self, env):
        pos_obs = list(np.array([j.get_position() for j in env.ordered_joints], dtype=np.float32).flatten())
        vel_obs = list(np.array([j.get_velocity() for j in env.ordered_joints], dtype=np.float32).flatten())
        return DotDict({"qpos": np.array(pos_obs), "qvel": np.array(vel_obs)})

    def bullet_setState(self, qpos, qvel, env):
        for n, j in enumerate(env.ordered_joints):
            j.reset_current_position(qpos[n], qvel[n])

    def get_state(self):
        if "Pendulum" in self._env_name:
            st = self.getState(self.env_set[0])
        elif getattr(self.env_set[0].unwrapped, "sim", None) is None:
            st = self.bullet_getState(self.env_set[0])
        else:
            st = self.env_set[0].unwrapped.sim.get_state()
        return st

    def set_state(self, st):
        for env in self.env_set:
            if "Pendulum" in self._env_name:
                self.setState(st[0], st[1], env)
                s = self.getState(env)
                assert (np.round(s[0], 8) == np.round(st[0], 8)).all(), "{} != {}".format(s[0], st[0])
                assert (np.round(s[1], 8) == np.round(st[1], 8)).all(), "{} != {}".format(s[1], st[1])
            elif getattr(self.env_set[0].unwrapped, "sim", None) is None:
                self.bullet_setState(st.qpos, st.qvel, env)
                s = self.bullet_getState(env)
                assert (np.round(s.qpos, 8) == np.round(st.qpos, 8)).all(), "{} != {}".format(s.qpos, st.qpos)
                assert (np.round(s.qvel, 8) == np.round(st.qvel, 8)).all(), "{} != {}".format(s.qvel, st.qvel)
            else:
                try:
                    env.unwrapped.set_state(st.qpos, st.qvel)
                    s = env.unwrapped.sim.get_state()
                except TypeError as err:
                    ## DClaw
                    env.unwrapped.sim.set_state(st)
                    s = env.unwrapped.sim.get_state()
                assert (np.round(s.qpos, 8) == np.round(st.qpos, 8)).all(), "{} != {}".format(s.qpos, st.qpos)
                assert (np.round(s.qvel, 8) == np.round(st.qvel, 8)).all(), "{} != {}".format(s.qvel, st.qvel)

    def set_varpi_ranges(self, ranges):
        self.min_rel_param, self.max_rel_param = ranges.T
        assert (self.min_rel_param <= self.max_rel_param).all(), "The ranges must be a proper interval, i.e. [m, M] s.t. m <= M. Got m={} and M={} instead.".format(self.min_rel_param, self.max_rel_param)

    def sample_varpis(self, type="continuous"):
        ####
        if type == "continuous":
            #self.varpis = self.subdomains.sample_domains_varpis()
            varpi_params = self.subdomains._varpi_rnd_generator.uniform(0., 1. + 1e-6, size=(self.n_regions, 2*self.min_rel_param.shape[0]))
            varpi_means, varpi_sigmas = np.split(varpi_params, 2, axis=-1)
            ####
            varpi_means = self.min_rel_param + (self.max_rel_param - self.min_rel_param) * varpi_means
            varpi_sigmas = varpi_sigmas * np.min(((self.max_rel_param - varpi_means), (varpi_means - self.min_rel_param)), axis=0) / np.sqrt(3.)
            ####
            self.varpis = np.concatenate((varpi_means, varpi_sigmas), axis=-1)
        else:
            assert type == "discrete"
            alpha = np.zeros(self.n_regions) + 0.5
            self.varpis = self.subdomains._varpi_rnd_generator.dirichlet(alpha, size=self.n_regions)
        return self.varpis

    def sample_env_params(self, use_varpi=False):
        ########
        if use_varpi:
            self.env_params = self.subdomains.sample_using_varpis(self.varpis)[~self._ep_dones]
        else:
            self.env_params = self.subdomains.sample_from_domains()[~self._ep_dones]
        ###
        assert self.env_params.shape[0] == len(self.env_set), "{} vs {}".format(self.env_params.shape, len(self.env_set))
        for env_idx, env in enumerate(self.env_set):
            if self._ep_dones[env_idx]:
                continue
            else:
                self.change_env_params(env, self.env_params[env_idx])
        ###
        return self.env_params

    def change_all_env_params(self, env_params):
        for env_idx, env in enumerate(self.env_set):
            self.change_env_params(env, env_params[env_idx])

    def _mujocopy_reset(self, same=True):
        if same:
            idx = 0
            state, _ = self.env_set[idx].reset()
            if "Pendulum" in self._env_name:
                st = self.getState(self.env_set[idx])
            elif getattr(self.env_set[idx].unwrapped, "sim", None) is None:
                st = self.bullet_getState(self.env_set[idx])
            else:
                st = self.env_set[idx].unwrapped.sim.get_state()
            #####
            for i, env in enumerate(self.env_set):
                if i != idx:
                    _ = env.reset()
                    if "Pendulum" in self._env_name:
                        self.setState(st[0], st[1], env)
                        s = self.getState(env)
                        assert (np.round(s[0], 8) == np.round(st[0], 8)).all(), "{} != {}".format(s[0], st[0])
                        assert (np.round(s[1], 8) == np.round(st[1], 8)).all(), "{} != {}".format(s[1], st[1])
                    elif getattr(self.env_set[idx].unwrapped, "sim", None) is None:
                        self.bullet_setState(st.qpos, st.qvel, env)
                        s = self.bullet_getState(env)
                        assert (np.round(s.qpos, 8) == np.round(st.qpos, 8)).all(), "{} != {}".format(s.qpos, st.qpos)
                        assert (np.round(s.qvel, 8) == np.round(st.qvel, 8)).all(), "{} != {}".format(s.qvel, st.qvel)
                    else:
                        try:
                            env.unwrapped.set_state(st.qpos, st.qvel)
                            s = env.unwrapped.sim.get_state()
                        except TypeError as err:
                            ## DClaw
                            env.unwrapped.sim.set_state(st)
                            s = env.unwrapped.sim.get_state()
                        assert (np.round(s.qpos, 8) == np.round(st.qpos, 8)).all(), "{} != {}".format(s.qpos, st.qpos)
                        assert (np.round(s.qvel, 8) == np.round(st.qvel, 8)).all(), "{} != {}".format(s.qvel, st.qvel)
            ####
            self._ep_dones = np.array([False for _ in range(self.n_regions)])
            ####
            return state.reshape(1, -1).repeat(self.n_regions, axis=0)
        else:
            states = []
            #####
            for i, env in enumerate(self.env_set):
                state, _ = env.reset()
                states.append(state)
            ####
            states = np.stack(states)
            assert states.shape[0] == self.n_regions, "{} vs {}".format(states.shape[0], self.n_regions)
            ####
            self._ep_dones = np.array([False for _ in range(self.n_regions)])
            ####
            return states

    def _mujoco_reset(self, same=True):
        if same:
            idx = 0
            state, _ = self.env_set[idx].reset()
            st = (self.env_set[idx].data.qpos, self.env_set[idx].data.qvel)
            #####
            for i, env in enumerate(self.env_set):
                if i != idx:
                    _ = env.reset()
                    env.unwrapped.set_state(st[0], st[1])
                    s = (env.data.qpos, env.data.qvel)
                    assert (np.round(s[0], 8) == np.round(st[0], 8)).all(), "{} != {}".format(s[0], st[0])
                    assert (np.round(s[1], 8) == np.round(st[1], 8)).all(), "{} != {}".format(s[1], st[1])
            ####
            self._ep_dones = np.array([False for _ in range(self.n_regions)])
            ####
            return state.reshape(1, -1).repeat(self.n_regions, axis=0)
        else:
            states = []
            #####
            for i, env in enumerate(self.env_set):
                state, _ = env.reset()
                states.append(state)
            ####
            states = np.stack(states)
            assert states.shape[0] == self.n_regions, "{} vs {}".format(states.shape[0], self.n_regions)
            ####
            self._ep_dones = np.array([False for _ in range(self.n_regions)])
            ####
            return states

    def step(self, actions):
        if self.env_params.shape[0] == self._ep_dones.shape[0]:
            env_params = self.env_params[~self._ep_dones]
        else:
            env_params = self.env_params
        varpis = getattr(self, "varpis", None)#self.varpis[~self._ep_dones]
        if varpis is not None:
            varpis = varpis[~self._ep_dones]
        if len(actions.shape) == 1:
            actions = actions.reshape(1, -1).repeat(self.n_regions, axis=0)[~self._ep_dones]
        elif actions.shape[0] == 1:
            actions = actions.repeat(self.n_regions, axis=0)[~self._ep_dones]
        assert env_params.shape[0] == actions.shape[0], "{} vs {}; {}".format(env_params.shape, actions.shape, self._ep_dones.shape)
        ######
        if actions.shape[-1] == 1:
            actions = actions.squeeze(axis=-1)
        ######
        transitions, infos = [], []
        env_idx = 0
        for i, env in enumerate(self.env_set):
            if self._ep_dones[i]:
                continue
            else:
                #self.change_env_params(env, env_params[env_idx])
                ######
                ns, r, d, _, info = env.step(actions[env_idx])
                transitions.append((ns, r, d))
                self._ep_dones[i] = d
                infos.append(info)
                env_idx += 1
        ####
        next_states, rewards, dones = map(np.stack, zip(*transitions))
        return next_states, rewards, dones, False, infos#, env_params, varpis

    def sample_env(self):
        idx = random.sample(range(self.n_regions), 1)[0]
        return self.env_set[idx], idx

    def get_dones(self):
        return np.copy(self._ep_dones)

    def get_stillActiveIdxs(self):
        idxs = np.array([i for i in range(self.n_regions)])
        return idxs[~self._ep_dones]

    def close(self):
        for env in self.env_set:
            env.close()
        del self.env_set[:]

    def __len__(self):
        return len(self.env_set)


###########################################################################################################################
###########################################################################################################################

class SIRSAMDEnvSet:
    def __init__(self, env_name, multi_dim, n_regions=100, seed=None, use_encoder=False, add_noise=False, use_quasi_random=False):
        ###### Read config file
        config_filename = env_name.split("-")[0] + ".yaml"
        if "DClaw" in config_filename:
            config_filename = "DClaw.yaml"
            self.observation_noise = 5e-3 if add_noise else None
            self.observation_keys = (
               'claw_qpos',
               'object_x',
               'object_y',
               'last_action',
               'target_error',
            )
            print("++ [SIRSAMDEnvSet] Observation noise: ", self.observation_noise)
        else:
            self.observation_noise = None
        with open("./pytorch_rl_collection/configs/" + config_filename, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        ######
        assert isinstance(multi_dim, bool), "Expected 'multi_dim' as Boolean, got {} with type {}".format(multi_dim, type(multi_dim))
        if multi_dim:
            config = config["multi_dim"]
        else:
            config = config["2d"]
        ######
        self.params_idxs = config["params_idxs"]
        self.params_env_idxs = config["params_env_idxs"]
        self.params_relative_ranges = config["params_relative_ranges"]
        ######
        print("++ [SIRSAMDEnvSet] Randomized params: ", self.params_idxs.keys())
        ######
        self.env_set = []
        self.action_space = None
        self.observation_space = None
        self._max_episode_steps = None
        ######
        self._ep_dones = None
        #######
        self._env_name = env_name
        if "Pendulum" in env_name:
            self._max_episode_steps = 200
        elif "DClaw" in env_name:
            self._max_episode_steps = 40
        else:
            self._max_episode_steps = 1000
        param_ranges = self.initialize_envs(env_name)
        #######
        try:
            if "mujoco" in str(type(self.env_set[0].env.env.env.env)):
                if getattr(self.env_set[0].unwrapped, "sim", None) is None:
                    self.reset = self._mujoco_reset
                else:
                    self.reset = self._mujocopy_reset
            else:
                self.reset = self._mujocopy_reset
        except AttributeError:
            self.reset = self._mujocopy_reset
        #######
        if seed is not None:
            self.seed(seed)
        #######
        self.use_encoder = use_encoder
        if self.use_encoder:
            self.latent_ranges = np.array([config["latent_ranges"]["x"], config["latent_ranges"]["y"]])
            self.subdomains = SubDomains(n_regions, self.latent_ranges, seed=seed, verbose=False, use_quasi_random=use_quasi_random)
            self._latent_decoder = Decoder(nb_envparams=param_ranges.shape[0], nb_latent=2,
                    lb=FloatTensor(param_ranges[:, 0]), ub=FloatTensor(param_ranges[:, 1]),
                    hidden_layers=[200, 300], init_w=3e-3, verbose=False
            )
            self._latent_decoder.load_state_dict(
                torch_load('./trained_agents/params_encoding_res/{}/decoder.pkl'.format(env_name))
            )
        else:
            env_key_list = list(self.params_idxs.keys())
            self.latent_ranges = []
            safety_check_params_idx = None
            for key, elt in self.params_idxs.items():
                if safety_check_params_idx is None:
                    safety_check_params_idx = elt
                else:
                    assert safety_check_params_idx != elt, "Params index repeatition detected {}".format(elt)
                if "body_pos" in key:
                    self.latent_ranges.append(np.array(self.params_relative_ranges[key + "_angle"]).reshape(1, -1))
                    self.latent_ranges.append(np.array(self.params_relative_ranges[key + "_radius"]).reshape(1, -1))
                else:
                    self.latent_ranges.append(np.array(self.params_relative_ranges[key]).reshape(1, -1).repeat(len(elt), axis=0))
            #####
            self.latent_ranges = np.concatenate(self.latent_ranges, axis=0)
            #####
            if self.latent_ranges.shape[0] == 2:
                self.subdomains = SubDomains(n_regions, self.latent_ranges, seed=seed, verbose=False, use_quasi_random=use_quasi_random)
            else:
                self.subdomains = MultiDimSubDomains(n_regions, self.latent_ranges, seed=seed, verbose=False, use_quasi_random=use_quasi_random)
            print("[SIRSAMDEnvSet]", self.subdomains)
        #######
        self.set_varpi_ranges(self.subdomains.ranges)
        #######
        self.rand_seed = seed
        self.n_regions = n_regions
        self.env_params = None

    def initialize_envs(self, env_name):
        self.env_set = []
        if self.observation_noise is None:
            env = gym.make(env_name)#, observation_keys=self.observation_keys)
            if getattr(env, "_max_episode_steps", None) is None:
                env = gym.wrappers.TimeLimit(env, max_episode_steps=self._max_episode_steps)
            else:
                self._max_episode_steps = env._max_episode_steps
            self.env_set.append(NormalizedEnv(env))
        else:
            self.env_set.append(NormalizedEnv(gym.make(env_name, sim_observation_noise=self.observation_noise, observation_keys=self.observation_keys)))
        ######
        if self.action_space is None:
            self.action_space = self.env_set[0].action_space
            self.observation_space = self.env_set[0].observation_space
            self._max_episode_steps = self.env_set[0]._max_episode_steps
            #####
            self.nominal_params = {}#[]
            self.lb_params, self.ub_params = [], []
            if env_name == "Pendulum-v1":
                for key in self.params_relative_ranges.keys():
                    lb, ub = self.params_relative_ranges[key]
                    params = rgetattr(self.env_set[0], key, None)
                    self.lb_params += [lb]
                    self.ub_params += [ub]
                    self.nominal_params[key] = float(params)
            else:
                for key in self.params_relative_ranges.keys():
                    lb, ub = self.params_relative_ranges[key]
                    ##
                    if "body_pos" in key:
                        self.lb_params += [lb]
                        self.ub_params += [ub]
                        self.nominal_params[key] = 0.5232783221319756 if "angle" in key else 0.06003332407921454
                    else:
                        try:
                            params = rgetattr(self.env_set[0].unwrapped.sim.model, key, None)
                        except AttributeError:
                            params = rgetattr(self.env_set[0].unwrapped.model, key, None)
                        if isinstance(params, float):
                            self.lb_params += [lb]
                            self.ub_params += [ub]
                            self.nominal_params[key] = float(params)
                        else:
                            idxs = self.params_env_idxs[key]
                            take_tuple = False
                            for elt in idxs:
                                if isinstance(elt, list):
                                    take_tuple = True
                                    break
                            if take_tuple:
                                idxs = tuple(idxs)
                            params = params[idxs]
                            self.lb_params += list(lb * np.ones_like(params))
                            self.ub_params += list(ub * np.ones_like(params))
                            self.nominal_params[key] = np.copy(params)
            #####
            param_ranges = np.array([self.lb_params, self.ub_params]).T
            ######
            self.lb_params, self.ub_params = np.array(self.lb_params), np.array(self.ub_params)
        #####
        return param_ranges

    def _get_idx(self, orig_list, val):
        val_ns, val_r, val_d = val
        for i, (ns, r, d) in enumerate(orig_list):
            if (ns == val_ns).all() and (r == val_r) and (d == val_d):
                return i
        return None

    def _key_function(self, iter):
        ## next_state, reward, done, info
        return iter[1] # compare list elements on the reward, i.e. on the index 1

    def push(self, new_env):
        if self.rand_seed is not None:
            new_env.seed(self.rand_seed)
        self.env_set.append(new_env)
        self.n_regions += 1
        if self.action_space is None:
            self.action_space = new_env.action_space
            self.observation_space = new_env.observation_space
            self._max_episode_steps = new_env._max_episode_steps
            #####
            if self._env_name == "Pendulum-v1":
                self._env_masses = 1.0 * new_env.m
            else:
                if getattr(new_env.unwrapped, "model", None) is not None:
                    self._env_masses = copy.deepcopy(new_env.unwrapped.model.body_mass.copy())

    def seed(self, seed, same=True):
        random.seed(seed)
        if len(self.env_set) > 0:
            for i, env in enumerate(self.env_set):
                try:
                    env.reset(seed=(int(not same) * i+1) * seed)
                except TypeError as err:
                    env.seed(seed=(int(not same) * i+1) * seed)
                if self.observation_noise is not None:
                    env.robot.random_state, _ = gym.utils.seeding.np_random((int(not same) * i+1) * seed)

    def change_env_params(self, env, params):
        if params is None:
            if self.use_encoder:
                params = np.ones_like(self.ub_params)
            else:
                params = np.ones(self.subdomains.domain_dim)
        else:
            #print(params)
            if self.use_encoder:
                with no_grad():
                    params = self._latent_decoder(FloatTensor(params).unsqueeze(0)).squeeze().numpy()
            #print(params)
        ###############
        keys = self.params_idxs.keys()
        ###############
        if self._env_name == "Pendulum-v1":
            for key in keys:
                rsetattr(env.env.env.env.env, key, params[self.params_idxs[key][0]] * self.nominal_params[key])
        elif getattr(env.unwrapped, "model", None) is not None:
            for key in keys:
                idxs = self.params_env_idxs[key]
                take_tuple = False
                for elt in idxs:
                    if isinstance(elt, list):
                        take_tuple = True
                        break
                if take_tuple:
                    idxs = tuple(idxs)
                if len(idxs) > 0:
                    ##
                    if "body_pos" in key:
                        orig_ang, orig_radius = self.nominal_params[key + "_angle"], self.nominal_params[key + "_radius"]
                        coef_orig_ang, coef_orig_radius = params[self.params_idxs[key]]
                        v_ffbase = rotate(v=np.array([0., -orig_radius*coef_orig_radius]), ang=orig_ang*coef_orig_ang)
                        v_mfbase = rotate(v=v_ffbase, ang=2.*np.pi/3.)
                        v_tfbase = rotate(v=v_ffbase, ang=4.*np.pi/3.)
                        ###
                        new_vals = np.array([np.round(v_ffbase, 3), np.round(v_mfbase, 3), np.round(v_tfbase, 3)]*2)
                        rgetattr(env.unwrapped.sim.model, key, None)[idxs] = new_vals
                        ###
                    else:
                        try:
                            rgetattr(env.unwrapped.sim.model, key, None)[idxs] = params[self.params_idxs[key]] * self.nominal_params[key]
                            if "actuator_gainprm" in key:
                                idxs = (idxs[0], 1)
                                rgetattr(env.unwrapped.sim.model, key.replace("actuator_gainprm", "actuator_biasprm"), None)[idxs] = - params[self.params_idxs[key]] * self.nominal_params[key]
                        except AttributeError:
                            rgetattr(env.unwrapped.model, key, None)[idxs] = params[self.params_idxs[key]] * self.nominal_params[key]
                else:
                    try:
                        rsetattr(env.unwrapped.sim.model, key, params[self.params_idxs[key]] * self.nominal_params[key])
                    except AttributeError:
                        rsetattr(env.unwrapped.model, key, params[self.params_idxs[key]] * self.nominal_params[key])

    def getState(self, env):
        return np.copy(env.env.env.state)

    def setState(self, qpos, qvel, env):
        env.env.env.state[:] = np.array([qpos, qvel])

    def bullet_getState(self, env):
        pos_obs = list(np.array([j.get_position() for j in env.ordered_joints], dtype=np.float32).flatten())
        vel_obs = list(np.array([j.get_velocity() for j in env.ordered_joints], dtype=np.float32).flatten())
        return DotDict({"qpos": np.array(pos_obs), "qvel": np.array(vel_obs)})

    def bullet_setState(self, qpos, qvel, env):
        for n, j in enumerate(env.ordered_joints):
            j.reset_current_position(qpos[n], qvel[n])

    def get_state(self):
        if "Pendulum" in self._env_name:
            st = self.getState(self.env_set[0])
        elif getattr(self.env_set[0].unwrapped, "sim", None) is None:
            st = self.bullet_getState(self.env_set[0])
        else:
            st = self.env_set[0].unwrapped.sim.get_state()
        return st

    def set_state(self, st):
        for env in self.env_set:
            if "Pendulum" in self._env_name:
                self.setState(st[0], st[1], env)
                s = self.getState(env)
                assert (np.round(s[0], 8) == np.round(st[0], 8)).all(), "{} != {}".format(s[0], st[0])
                assert (np.round(s[1], 8) == np.round(st[1], 8)).all(), "{} != {}".format(s[1], st[1])
            elif getattr(self.env_set[0].unwrapped, "sim", None) is None:
                self.bullet_setState(st.qpos, st.qvel, env)
                s = self.bullet_getState(env)
                assert (np.round(s.qpos, 8) == np.round(st.qpos, 8)).all(), "{} != {}".format(s.qpos, st.qpos)
                assert (np.round(s.qvel, 8) == np.round(st.qvel, 8)).all(), "{} != {}".format(s.qvel, st.qvel)
            else:
                try:
                    env.unwrapped.set_state(st.qpos, st.qvel)
                    s = env.unwrapped.sim.get_state()
                except TypeError as err:
                    ## DClaw
                    env.unwrapped.sim.set_state(st)
                    s = env.unwrapped.sim.get_state()
                assert (np.round(s.qpos, 8) == np.round(st.qpos, 8)).all(), "{} != {}".format(s.qpos, st.qpos)
                assert (np.round(s.qvel, 8) == np.round(st.qvel, 8)).all(), "{} != {}".format(s.qvel, st.qvel)

    def set_varpi_ranges(self, ranges):
        self.min_rel_param, self.max_rel_param = ranges.T
        assert (self.min_rel_param <= self.max_rel_param).all(), "The ranges must be a proper interval, i.e. [m, M] s.t. m <= M. Got m={} and M={} instead.".format(self.min_rel_param, self.max_rel_param)

    def sample_varpis(self, type="continuous"):
        ####
        if self.env_params is None:
            if type == "continuous":
                varpi_params = self.subdomains._varpi_rnd_generator.uniform(0., 1. + 1e-6, size=(self.n_regions, 2*self.min_rel_param.shape[0]))
                varpi_means, varpi_sigmas = np.split(varpi_params, 2, axis=-1)
                ####
                varpi_means = self.min_rel_param + (self.max_rel_param - self.min_rel_param) * varpi_means
                varpi_sigmas = varpi_sigmas * np.min(((self.max_rel_param - varpi_means), (varpi_means - self.min_rel_param)), axis=0) / np.sqrt(3.)
                ####
                self.varpis = np.concatenate((varpi_means, varpi_sigmas), axis=-1)
                self.env_params = varpi_means
            else:
                raise NotImplementedError
        return self.varpis, self.env_params

    def sample_env_params(self, use_varpi=False):
        ########
        if self.env_params is None:
            _ = self.sample_varpis(type="continuous")
            assert self.env_params.shape[0] == self.n_regions, "{} vs {}".format(self.env_params.shape, self.n_regions)
        ###
        varpi_idx = self.subdomains._points_rnd_gen.randint(self.n_regions)
        _, varpi = self.env_params[varpi_idx], self.varpis[varpi_idx]
        varpi_means, varpi_sigmas = np.split(varpi, 2, axis=-1)
        kappa = varpi_means + (2. * self.subdomains._points_rnd_gen.uniform(size=varpi_means.shape) - 1.) * np.sqrt(3.) * varpi_sigmas
        self.change_env_params(self.env_set[0], kappa)
        ###
        return kappa.reshape(1, -1), varpi.reshape(1, -1), varpi_idx

    def change_all_env_params(self, env_params):
        for env_idx, env in enumerate(self.env_set):
            self.change_env_params(env, env_params[env_idx])

    def _mujocopy_reset(self, same=True):
        if same:
            idx = 0
            state, _ = self.env_set[idx].reset()
            if "Pendulum" in self._env_name:
                st = self.getState(self.env_set[idx])
            elif getattr(self.env_set[idx].unwrapped, "sim", None) is None:
                st = self.bullet_getState(self.env_set[idx])
            else:
                st = self.env_set[idx].unwrapped.sim.get_state()
            #####
            for i, env in enumerate(self.env_set):
                if i != idx:
                    _ = env.reset()
                    if "Pendulum" in self._env_name:
                        self.setState(st[0], st[1], env)
                        s = self.getState(env)
                        assert (np.round(s[0], 8) == np.round(st[0], 8)).all(), "{} != {}".format(s[0], st[0])
                        assert (np.round(s[1], 8) == np.round(st[1], 8)).all(), "{} != {}".format(s[1], st[1])
                    elif getattr(self.env_set[idx].unwrapped, "sim", None) is None:
                        self.bullet_setState(st.qpos, st.qvel, env)
                        s = self.bullet_getState(env)
                        assert (np.round(s.qpos, 8) == np.round(st.qpos, 8)).all(), "{} != {}".format(s.qpos, st.qpos)
                        assert (np.round(s.qvel, 8) == np.round(st.qvel, 8)).all(), "{} != {}".format(s.qvel, st.qvel)
                    else:
                        try:
                            env.unwrapped.set_state(st.qpos, st.qvel)
                            s = env.unwrapped.sim.get_state()
                        except TypeError as err:
                            ## DClaw
                            env.unwrapped.sim.set_state(st)
                            s = env.unwrapped.sim.get_state()
                        assert (np.round(s.qpos, 8) == np.round(st.qpos, 8)).all(), "{} != {}".format(s.qpos, st.qpos)
                        assert (np.round(s.qvel, 8) == np.round(st.qvel, 8)).all(), "{} != {}".format(s.qvel, st.qvel)
            ####
            self._ep_dones = np.array([False for _ in range(1)])
            ####
            return state.reshape(1, -1).repeat(1, axis=0)
        else:
            states = []
            #####
            for i, env in enumerate(self.env_set):
                state, _ = env.reset()
                states.append(state)
            ####
            states = np.stack(states)
            assert states.shape[0] == 1, "{} vs {}".format(states.shape[0], 1)
            ####
            self._ep_dones = np.array([False for _ in range(1)])
            ####
            return states

    def _mujoco_reset(self, same=True):
        if same:
            idx = 0
            state, _ = self.env_set[idx].reset()
            st = (self.env_set[idx].data.qpos, self.env_set[idx].data.qvel)
            #####
            for i, env in enumerate(self.env_set):
                if i != idx:
                    _ = env.reset()
                    env.unwrapped.set_state(st[0], st[1])
                    s = (env.data.qpos, env.data.qvel)
                    assert (np.round(s[0], 8) == np.round(st[0], 8)).all(), "{} != {}".format(s[0], st[0])
                    assert (np.round(s[1], 8) == np.round(st[1], 8)).all(), "{} != {}".format(s[1], st[1])
            ####
            self._ep_dones = np.array([False for _ in range(1)])
            ####
            return state.reshape(1, -1).repeat(1, axis=0)
        else:
            states = []
            #####
            for i, env in enumerate(self.env_set):
                state, _ = env.reset()
                states.append(state)
            ####
            states = np.stack(states)
            assert states.shape[0] == 1, "{} vs {}".format(states.shape[0], 1)
            ####
            self._ep_dones = np.array([False for _ in range(1)])
            ####
            return states

    def step(self, actions):
        if len(actions.shape) == 1:
            actions = actions.reshape(1, -1)[~self._ep_dones]
        elif actions.shape[0] == 1:
            actions = actions.repeat(1, axis=0)[~self._ep_dones]
        ######
        if actions.shape[-1] == 1:
            actions = actions.squeeze(axis=-1)
        ######
        transitions, infos = [], []
        env_idx = 0
        for i, env in enumerate(self.env_set):
            if self._ep_dones[i]:
                continue
            else:
                ns, r, d, _, info = env.step(actions[env_idx])
                transitions.append((ns, r, d))
                self._ep_dones[i] = d
                infos.append(info)
                env_idx += 1
        ####
        next_states, rewards, dones = map(np.stack, zip(*transitions))
        return next_states, rewards, dones, False, infos

    def sample_env(self):
        return self.env_set[0], 0

    def get_dones(self):
        return np.copy(self._ep_dones)

    def get_stillActiveIdxs(self):
        idxs = np.array([i for i in range(1)])
        return idxs[~self._ep_dones]

    def close(self):
        for env in self.env_set:
            env.close()
        del self.env_set[:]

    def __len__(self):
        return len(self.env_set)

################################################################################
if __name__ == "__main__":
    env = SIRSAMDEnvSet(env_name="Hopper-v3", n_regions=3, seed=0)
    s = env.reset()
    act = env.action_space.sample()
    print("++ Action: ", act)
    p = env.sample_env_params(False)
    print("++ Parameters: ", p)
    ns, r, d, _, env_p, v = env.step(act)
    print(env_p); print(v)
    print(ns)
    print(env.subdomains.ranges)
    env_key_list = list(env.params_idxs.keys()); env_key_list; len(env_key_list)
    env.close()
