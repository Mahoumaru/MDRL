import torch
from torch.nn import MSELoss
import inspect
import numpy as np

from pytorch_rl_collection.model_networks.model_networks import SacActor as Actor
from pytorch_rl_collection.model_networks.model_networks import SacCritic as Critic
from pytorch_rl_collection.replay_buffers.replay_memory import ReplayMemory
from pytorch_rl_collection.utils import *
from pytorch_rl_collection.model_networks.model_networks import DynamicsNet, EnsembleOSINet

###########################################
criterion = MSELoss()

###########################################
class SDRSAC_AGENT:
    def __init__(self, states_dim, actions_dim, domains_dim, args, OptimCls=torch.optim.Adam,
                 ReplayBufferCls=ReplayMemory, value_range=None, add_noise=False, lambda_steplength=1e-5):
        self.steps = 0
        self.warmup = args.warmup
        self.states_dim = states_dim
        self.actions_dim = actions_dim
        self.verbose = args.verbose

        self.n_regions = args.n_regions

        self.net_cfg = {
            'hidden_layers': args.hidden_layers,
        }
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.entropy_temp_lr = args.entropy_temp_lr
        self.OptimCls = OptimCls
        print("++ Optimizer class: ", self.OptimCls)
        self.ReplayBufferCls = ReplayBufferCls
        self.rmsize = args.rmsize
        self.window_length = args.window_length
        self.alpha = args.alpha
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        # Hyper-parameters
        #self.batch_size = args.batch_size
        self.tau = args.tau
        self.discount = args.discount

        self.lambda_steplength = lambda_steplength
        ###
        self.domain_dim = domains_dim

        #
        ### self.use_cuda = args.cuda
        self.device = torch.device("cuda:{}".format(args.cuda_id) if args.cuda else "cpu")
        print("++ GPU Device: ", self.device)
        ###
        self.seed = args.seed
        self.set_seed(self.seed)
        ###
        self._random_generator = torch.Generator(device=self.device)
        self._points_random_generator = torch.Generator(device=self.device)
        if self.seed is not None:
            self._random_generator = self._random_generator.manual_seed(1234 * self.seed)
            self._points_random_generator = self._points_random_generator.manual_seed(1234 * self.seed)
        ####
        sim_observation_noise = 5e-3
        if value_range is not None and add_noise:
            self.noise_amplitude = torch.FloatTensor(sim_observation_noise * np.ptp(value_range, axis=1)).to(self.device)
        else:
            self.noise_amplitude = None

    def add_noise(self, states):
        if self.noise_amplitude is not None:
            states[:, :9] = states[:, :9] + self.noise_amplitude * torch.FloatTensor(states.shape[0], 9).to(self.device).uniform_(-0.5, 0.5, generator=self._points_random_generator)
        return states

    def set_varpi_ranges(self, ranges):
        self.min_rel_param, self.max_rel_param = ranges.T
        assert (self.min_rel_param <= self.max_rel_param).all(), "The ranges must be a proper interval, i.e. [m, M] s.t. m <= M. Got m={} and M={} instead.".format(self.min_rel_param, self.max_rel_param)
        self.min_rel_param, self.max_rel_param = torch.FloatTensor(self.min_rel_param).to(self.device), torch.FloatTensor(self.max_rel_param).to(self.device)

    def initialize(self):
        # An utility function to be able to initialize
        # everything at once
        self.initialize_critics()
        self.initialize_actor()
        self.initialize_entropy_temperature()
        self.initialize_lambda()
        self.initialize_targets()
        self.initialize_world_model()
        self.initialize_replay_buffer_set(size=self.n_regions)

    def initialize_critics(self):
        self.critic1 = Critic(self.states_dim, self.actions_dim, **self.net_cfg).to(self.device)
        self.critic2 = Critic(self.states_dim, self.actions_dim, **self.net_cfg).to(self.device)
        if self.verbose:
            print("++ Both Critics Initialized with following structure:")
            print(self.critic1)
            print("")
        self.critic1_optim  = self.OptimCls(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optim  = self.OptimCls(self.critic2.parameters(), lr=self.critic_lr)

    def initialize_actor(self):
        net_cfg = self.net_cfg.copy()
        net_cfg['policy_type'] = "Gaussian"
        self.actor = Actor(self.states_dim, self.actions_dim, **net_cfg).to(self.device)
        if self.verbose:
            print("++ Actor Initialized with following structure:")
            print(self.actor)
            print("")
        self.actor_optim  = self.OptimCls(self.actor.parameters(), lr=self.actor_lr)

    def initialize_entropy_temperature(self):
        if self.automatic_entropy_tuning:
            self.target_entropy = - self.actions_dim
            #self.log_alpha = torch.tensor([np.log(self.alpha)], requires_grad=True, device=self.device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = self.OptimCls([self.log_alpha], lr=self.entropy_temp_lr)

    def initialize_lambda(self):
        self.homotopy_lambda = 0.
        if self.verbose:
            print("++ Homotopy parameter Initialized with steplength: {}".format(self.lambda_steplength))

    def initialize_targets(self):
        self.critic1_target = Critic(self.states_dim, self.actions_dim, **self.net_cfg).to(self.device)
        self.critic2_target = Critic(self.states_dim, self.actions_dim, **self.net_cfg).to(self.device)
        if self.verbose:
            print("++ Both Target Critic networks Initialized.")
        hard_update(self.critic1_target, self.critic1) # Make sure target is with the same weight
        hard_update(self.critic2_target, self.critic2)

    def initialize_world_model(self):
        self.forwardDyn = DynamicsNet(
            self.states_dim, self.actions_dim,
            nb_latent=self.domain_dim, hidden_layers=[200, 200, 200, 200],
            init_w=3e-3, verbose=False
        ).to(self.device)
        self.reward_net = Critic(2*self.states_dim, self.actions_dim, **self.net_cfg).to(self.device)
        if self.verbose:
            print("++ Forward dynamics network Initialized with following structure:")
            print(self.forwardDyn)
            print("")
            print("++ Reward predictor network Initialized with following structure:")
            print(self.reward_net)
            print("")
        self.forwardDyn_optim = self.OptimCls(self.forwardDyn.parameters(), lr=self.critic_lr)
        self.reward_net_optim  = self.OptimCls(self.reward_net.parameters(), lr=self.critic_lr)

    def initialize_replay_buffer(self):
        self.replay_memory = self.ReplayBufferCls(capacity=self.rmsize, seed=self.seed, window_length=self.window_length)
        if self.verbose:
            print("++ Replay Memory Initialized.")

    def initialize_replay_buffer_set(self, size):
        self.replay_memory = self.ReplayBufferCls(size=size, capacity=self.rmsize, seed=self.seed, window_length=self.window_length)
        if self.verbose:
            print("++ Replay Memory Initialized.")

    def select_action(self, state, is_training=True, squeeze=True):
        if self.steps <= self.warmup:
            if len(state.shape) > 1:
                action = np.random.uniform(-1.,1., self.actions_dim)
                action = action.reshape(1, -1).repeat(state.shape[0], axis=0)
            else:
                action = np.random.uniform(-1.,1., self.actions_dim)
        else:
            state = torch.FloatTensor(state).to(self.device)
            #####
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            #####
            state = self.add_noise(state)
            if is_training:
                action, _, _ = self.actor(state)
                #####
                action = to_numpy(
                    action
                )
            else:
                _, _, action = self.actor(state)
                #####
                action = to_numpy(
                    action
                ).squeeze(axis=0)
            #action = np.clip(action, -1., 1.)

        self.steps += int(is_training)#1
        return action

    def store_transition(self, transition):
        state, action, reward, next_state, done = transition
        self.replay_memory.push(state, action, reward, next_state, done)

    def rmset_store_transition(self, transition, valid_idxs=None):
        state, action, reward, next_state, done, kappa, varpi = transition
        self.replay_memory.push(valid_idxs, state, action, reward, next_state, done, kappas=kappa, varpis=varpi)

    def rmset_sample_minibatch(self, batch_size):
        if self.steps > self.warmup:
            #print(self.steps, batch_size)
            for state_batch, action_batch, reward_batch, \
            next_state_batch, terminal_batch, kappa_batch, _ in self.replay_memory.sample(batch_size):
                state_batch = torch.FloatTensor(state_batch).to(self.device)
                next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
                action_batch = torch.FloatTensor(action_batch).to(self.device)
                reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
                terminal_batch = torch.FloatTensor(terminal_batch).to(self.device).unsqueeze(1)
                kappa_batch = torch.FloatTensor(kappa_batch).to(self.device)
                yield state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, kappa_batch
        else:
            yield None, None, None, None, None, None

    def sample_kappas(self, size=10):
        rand_gen = self._random_generator
        dim = self.domain_dim
        params = torch.FloatTensor(size, dim).to(self.device).uniform_(0., 1. + 1e-6, generator=rand_gen)
        ####
        params = self.min_rel_param + (self.max_rel_param - self.min_rel_param) * params
        ####
        return params

    def get_td_target(self, states, actions, rewards, next_states, masks):
        if states is not None:
            # Prepare the target q batch
            with torch.no_grad():
                bs = states.shape[0]
                #next_states = self.add_noise(next_states)
                sampled_kappas = self.sample_kappas(size=bs)
                sampled_kappas = sampled_kappas.unsqueeze(1).repeat(1, bs, 1)
                aug_states = states.unsqueeze(0).repeat(bs, 1, 1)
                #####
                sampled_ns, _, _ = self.forwardDyn(aug_states, actions.unsqueeze(0).repeat(bs, 1, 1), sampled_kappas)
                #####
                sampled_ns_action, sampled_ns_log_pi, _ = self.actor(sampled_ns)
                ## Q1
                next_q1_values = self.critic1_target([ sampled_ns, sampled_ns_action ])
                ## Q2
                next_q2_values = self.critic2_target([ sampled_ns, sampled_ns_action ])
                ## min(Q1, Q2)
                min_sampled_next_qf_target = (torch.min(next_q1_values, next_q2_values) - self.alpha * sampled_ns_log_pi).mean(0)
                ##########################################
                ##########################################
                next_states_action, next_states_log_pi, _ = self.actor(next_states)
                ## Q1
                next_q1_values = self.critic1_target([ next_states, next_states_action ])
                ## Q2
                next_q2_values = self.critic2_target([ next_states, next_states_action ])
                ## min(Q1, Q2)
                min_next_qf_target = torch.min(next_q1_values, next_q2_values) - self.alpha * next_states_log_pi

            target_q_batch = rewards + \
                self.discount * masks * (self.homotopy_lambda * min_next_qf_target + (1. - self.homotopy_lambda) * min_sampled_next_qf_target)
            return target_q_batch
        else:
            return None

    def update_critics(self, td_target, states, actions):
        if td_target is not None:
            self.critic1_optim.zero_grad()
            self.critic2_optim.zero_grad()
            #####
            #states = self.add_noise(states)
            #####
            q1_batch = self.critic1([ states, actions ])
            q2_batch = self.critic2([ states, actions ])
            value_loss = criterion(q1_batch, td_target) + criterion(q2_batch, td_target)
            value_loss.backward()
            #####
            self.critic1_optim.step()
            self.critic2_optim.step()

    def update_actor(self, states):
        if states is not None:
            self.actor_optim.zero_grad()
            #####
            actions, log_pi, _ = self.actor(self.add_noise(states))
            ######
            policy_loss = (self.alpha * log_pi) - torch.min(self.critic1([ states, actions ]), self.critic2([ states, actions ]))
            policy_loss = policy_loss.mean()
            ############################################
            policy_loss.backward()
            self.actor_optim.step()

    def update_world_model(self, states, actions, next_states, kappas, rewards):
        if states is not None:
            self.forwardDyn_optim.zero_grad()
            ##########################
            dyn_log_likelihood = self.forwardDyn.evaluate(states, actions, kappas, next_states)
            ####
            loss = - dyn_log_likelihood.mean()
            ##########################
            loss.backward()
            self.forwardDyn_optim.step()
            ############################################
            ##### Reward Net update
            ############################################
            self.reward_net_optim.zero_grad()
            ##########################
            predicted_rewards = self.reward_net([ torch.cat([states, next_states], dim=-1), actions ])
            ####
            loss = criterion(predicted_rewards, rewards)
            ##########################
            loss.backward()
            self.reward_net_optim.step()

    def update_temperature(self, states):
        if states is not None and self.automatic_entropy_tuning:
            self.alpha_optim.zero_grad()

            with torch.no_grad():
                _, log_pi, _ = self.actor(states)
            alpha_loss = - (self.log_alpha * (log_pi + self.target_entropy)).mean()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp().item()

    def increase_lambda(self):
        if self.steps > self.warmup:
            self.homotopy_lambda += self.lambda_steplength
            self.homotopy_lambda = min(self.homotopy_lambda, 1.)

    def update_target_networks(self, update):
        #if self.steps > self.warmup:
        if update:
            soft_update(self.critic1_target, self.critic1, self.tau)
            soft_update(self.critic2_target, self.critic2, self.tau)

    ### Utilities functions:
    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic1_target.train()
        self.critic2.train()
        self.critic2_target.train()

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic1_target.eval()
        self.critic2.eval()
        self.critic2_target.eval()

    def to_cuda(self):
        self.actor.cuda()
        self.critic1.cuda()
        self.critic1_target.cuda()
        self.critic2.cuda()
        self.critic2_target.cuda()

    def to_cpu(self):
        self.actor.cpu()
        self.critic1.cpu()
        self.critic1_target.cpu()
        self.critic2.cpu()
        self.critic2_target.cpu()

    def set_seed(self, s):
        np.random.seed(s)
        torch.manual_seed(s)
        if self.device == torch.device("cuda"):#self.use_cuda:
            torch.cuda.manual_seed(s)

    def load_weights(self, output_path):
        if output_path is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output_path))
        )

        self.critic1.load_state_dict(
            torch.load('{}/critic1.pkl'.format(output_path))
        )

        self.critic2.load_state_dict(
            torch.load('{}/critic2.pkl'.format(output_path))
        )

        self.forwardDyn.load_state_dict(
            torch.load('{}/forwardDyn.pkl'.format(output_path))
        )

        self.reward_net.load_state_dict(
            torch.load('{}/reward_net.pkl'.format(output_path))
        )


    def save_model(self, output_path, id=""):
        torch.save(
            self.actor.state_dict(),
            '{}/actor{}.pkl'.format(output_path, id)
        )
        torch.save(
            self.critic1.state_dict(),
            '{}/critic1.pkl'.format(output_path)
        )
        torch.save(
            self.critic2.state_dict(),
            '{}/critic2.pkl'.format(output_path)
        )
        torch.save(
            self.forwardDyn.state_dict(),
            '{}/forwardDyn.pkl'.format(output_path)
        )
        torch.save(
            self.reward_net.state_dict(),
            '{}/reward_net.pkl'.format(output_path)
        )
