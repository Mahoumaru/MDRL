import torch
from torch.nn import MSELoss, L1Loss, Softmax
import inspect
import numpy as np

#from pytorch_rl_collection.model_networks.model_networks import SacActor as Actor
#from pytorch_rl_collection.model_networks.model_networks import SacCritic as Critic
from pytorch_rl_collection.replay_buffers.replay_memory import ReplayMemory
from pytorch_rl_collection.utils import *#hard_update, soft_update
from pytorch_rl_collection.model_networks.model_networks import EnsembleSystemID

from torchoptim.optimizers import AdaTerm

###########################################
criterion_A = MSELoss(reduction='none')
criterion_B = L1Loss(reduction='none')
softmax_fct = Softmax(dim=1)

###########################################
class SIRSA_AGENT:
    def __init__(self, states_dim, actions_dim, domains_dim, args, OptimCls=torch.optim.Adam,
                 ReplayBufferCls=ReplayMemory, value_range=None, add_noise=False,
                 model_type="", threshold=int(5e5)):
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

        #
        self.model_type = model_type
        if self.model_type == "bilinear":
            from pytorch_rl_collection.model_networks.model_networks import BilinearSacActor as Actor
            from pytorch_rl_collection.model_networks.model_networks import BilinearSacCritic as Critic
        elif self.model_type == "hypernet":
            from pytorch_rl_collection.model_networks.model_networks import HypernetSacActor as Actor
            from pytorch_rl_collection.model_networks.model_networks import HypernetSacCritic as Critic
        else:
            from pytorch_rl_collection.model_networks.model_networks import SacActor as Actor
            from pytorch_rl_collection.model_networks.model_networks import SacCritic as Critic
        self.CriticCls = Critic
        self.ActorCls = Actor

        # Hyper-parameters
        #self.batch_size = args.batch_size
        self.tau = args.tau
        self.discount = args.discount
        self.SI_ensemble_size = args.n_osi
        self.threshold = threshold
        self.use_cvar = args.use_cvar
        self.cvar_alpha = args.cvar_alpha
        self.cvar_samples = args.cvar_samples
        print("++ Threshold: ", self.threshold)

        #
        ### self.use_cuda = args.cuda
        self.device = torch.device("cuda:{}".format(args.cuda_id) if args.cuda else "cpu")
        print("++ GPU Device: ", self.device)
        ###
        self.seed = args.seed
        self.set_seed(self.seed)
        ###
        self._th_random_generator = torch.Generator(device=self.device)
        self._varpi_random_generator = torch.Generator(device=self.device)
        self._points_random_generator = torch.Generator(device=self.device)
        if self.seed is not None:
            self._th_random_generator = self._th_random_generator.manual_seed(1234 * self.seed)
            self._varpi_random_generator = self._varpi_random_generator.manual_seed(1234 * self.seed)
            self._points_random_generator = self._points_random_generator.manual_seed(1234 * self.seed)
        ####
        self.continuous = args.continuous
        self.explicit_scal = args.explicit_scal
        self.use_ut = args.use_ut
        if self.continuous:
            self.varpi_size = 2 * domains_dim # = 2 * 2
            if self.use_ut:
                UNSCENTED_TRANSFO_DICT = fetch_uniform_unscented_transfo(dim=domains_dim)
                self.sigma_points = torch.FloatTensor(UNSCENTED_TRANSFO_DICT["sigma_points"]).to(self.device)
                #self.sigma_weights = torch.FloatTensor(UNSCENTED_TRANSFO_DICT["sigma_weights"]).to(self.device)
                self.sp_counter = 0
                self.sp_max_num = 1#self.sigma_points.shape[0]
        else:
            raise NotImplementedError
        ####
        sim_observation_noise = 5e-3
        if value_range is not None and add_noise:
            self.noise_amplitude = torch.FloatTensor(sim_observation_noise * np.ptp(value_range, axis=1)).to(self.device)
        else:
            self.noise_amplitude = None
        ####
        print("++ Noise Amplitude: ", self.noise_amplitude)

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
        self.initialize_targets()
        self.initialize_systemID()
        self.initialize_replay_buffer_set(size=self.n_regions)

    def initialize_critics(self):
        if self.continuous:
            net_cfg = self.net_cfg.copy()
            net_cfg['nb_varpi'] = (self.varpi_size // 2)
            net_cfg['per_layer_varpi'] = False
            self.critic1 = self.CriticCls(self.states_dim, self.actions_dim, output_dim=1, **net_cfg).to(self.device)
            self.critic2 = self.CriticCls(self.states_dim, self.actions_dim, output_dim=1, **net_cfg).to(self.device)
        else:
            self.critic1 = self.CriticCls(self.states_dim + self.varpi_size, self.actions_dim, output_dim=1, **self.net_cfg).to(self.device)
            self.critic2 = self.CriticCls(self.states_dim + self.varpi_size, self.actions_dim, output_dim=1, **self.net_cfg).to(self.device)
        if self.verbose:
            print("++ Both Critics Initialized with following structure:")
            print(self.critic1)
            print("")
        self.critic1_optim  = self.OptimCls(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optim  = self.OptimCls(self.critic2.parameters(), lr=self.critic_lr)

    def initialize_actor(self):
        net_cfg = self.net_cfg.copy()
        net_cfg['policy_type'] = "Gaussian"
        net_cfg['nb_varpi'] = self.varpi_size
        net_cfg['per_layer_varpi'] = False
        self.actor = self.ActorCls(self.states_dim, self.actions_dim, **net_cfg).to(self.device)
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

    def initialize_targets(self):
        if self.continuous:
            net_cfg = self.net_cfg.copy()
            net_cfg['nb_varpi'] = (self.varpi_size // 2)
            net_cfg['per_layer_varpi'] = False
            self.critic1_target = self.CriticCls(self.states_dim, self.actions_dim, output_dim=1, **net_cfg).to(self.device)
            self.critic2_target = self.CriticCls(self.states_dim, self.actions_dim, output_dim=1, **net_cfg).to(self.device)
        else:
            self.critic1_target = self.CriticCls(self.states_dim + self.varpi_size, self.actions_dim, output_dim=1, **self.net_cfg).to(self.device)
            self.critic2_target = self.CriticCls(self.states_dim + self.varpi_size, self.actions_dim, output_dim=1, **self.net_cfg).to(self.device)
        if self.verbose:
            print("++ Both Target Critic networks Initialized.")
        hard_update(self.critic1_target, self.critic1) # Make sure target is with the same weight
        hard_update(self.critic2_target, self.critic2)
        if self.verbose:
            print("++ hard copy done.")

    def initialize_systemID(self):
        input_dim = self.varpi_size + 2*self.states_dim + self.actions_dim
        self.systemID = EnsembleSystemID(ensemble_size=self.SI_ensemble_size,
            history_length=input_dim,
            nb_latent=self.varpi_size // 2,
            device=self.device,
            use_sigmoid=False
        )
        if self.verbose:
            print("++ System ID Ensemble Initialized with following structure:")
            print(self.systemID)
            print("")
        self.systemID_optim = self.systemID.get_optimizer(self.OptimCls, lr=self.critic_lr)

    def initialize_replay_buffer(self):
        self.replay_memory = self.ReplayBufferCls(capacity=self.rmsize, seed=self.seed, window_length=self.window_length)
        if self.verbose:
            print("++ Replay Memory Initialized.")

    def initialize_replay_buffer_set(self, size):
        self.replay_memory = self.ReplayBufferCls(
            size=size, capacity=self.rmsize,
            seed=self.seed, window_length=self.window_length,
            union_sampling=True, numpy_buffer=True
        )
        if self.verbose:
            print("++ Replay Memory Set Initialized with size {}.".format(size))

    def select_action(self, state, varpi, is_training=True, squeeze=True):
        if self.steps <= self.warmup:
            if len(state.shape) > 1:
                action = np.random.uniform(-1.,1., self.actions_dim)
                action = action.reshape(1, -1).repeat(state.shape[0], axis=0)
            else:
                action = np.random.uniform(-1.,1., self.actions_dim)
        else:
            if not isinstance(varpi, torch.Tensor):
                varpi = torch.FloatTensor(varpi).to(self.device)
            if varpi.shape[-1] != self.varpi_size:
                varpi = torch.cat((varpi, torch.zeros_like(varpi)), dim=-1)
            state = torch.FloatTensor(state).to(self.device)
            #####
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            if len(varpi.shape) == 1:
                varpi = varpi.unsqueeze(0).repeat(state.shape[0], 1)
            if state.shape[0] != varpi.shape[0]:
                varpi = varpi.repeat(state.shape[0], 1)
            #####
            assert state.shape[0] == varpi.shape[0], "{} vs {}".format(state.shape[0], varpi.shape[0])
            #####
            aug_state = torch.cat((state, varpi), dim=-1)
            if is_training:
                action, _, _ = self.actor(aug_state)
                #####
                action = to_numpy(
                    action
                )
            else:
                _, _, action = self.actor(aug_state)
                #####
                action = to_numpy(
                    action
                )
                if squeeze:
                    action = action.squeeze(axis=0)
            #action = np.clip(action, -1., 1.)

        self.steps += int(is_training)#1
        return action

    def update_uncertainty(self, state, action, next_state, varpi, verbose=False):
        if self.steps <= self.warmup:
            return varpi
        ######
        if not isinstance(varpi, torch.Tensor):
            varpi = torch.FloatTensor(varpi).to(self.device)
        if varpi.shape[-1] != self.varpi_size:
            varpi = torch.cat((varpi, torch.zeros_like(varpi)), dim=-1)
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.FloatTensor(action).to(self.device)
        #####
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        if len(varpi.shape) == 1:
            varpi = varpi.unsqueeze(0).repeat(state.shape[0], 1)
        if state.shape[0] != varpi.shape[0]:
            varpi = varpi.repeat(state.shape[0], 1)
        #####
        assert state.shape[0] == varpi.shape[0], "{} vs {}".format(state.shape[0], varpi.shape[0])
        assert state.shape[0] == action.shape[0], "{} vs {}".format(state.shape[0], action.shape[0])
        #####
        mean, sigma = torch.split(varpi, self.varpi_size//2, dim=-1)
        if verbose:
            print("Original varpi params: ", mean, sigma)
        ####
        aug_state = torch.cat((state, action, next_state, varpi), dim=-1)
        predicted_sigmas = self.systemID(aug_state)
        new_varpi_elements = torch.stack([mean + np.sqrt(3.) * predicted_sigmas[i].clip(min=-sigma, max=sigma)
            for i in range(self.SI_ensemble_size)])
        #new_varpi_elements = self.systemID(aug_state)
        #new_varpi_elements = self.min_rel_param + new_varpi_elements * (self.max_rel_param - self.min_rel_param)
        #new_varpi_elements.clip(min=self.min_rel_param, max=self.max_rel_param)
        mean, std = new_varpi_elements.mean(0), new_varpi_elements.std(0)
        if verbose:
            print("Predicted sigmas: ", predicted_sigmas.clip(min=-sigma, max=sigma).mean(0), predicted_sigmas.shape)
            #print("New varpi elements: ", new_varpi_elements[0], new_varpi_elements.shape)
            print("New varpi params: ", mean, std)
            print(jkklk)
        std = torch.min(std, torch.min(self.max_rel_param - mean, mean - self.min_rel_param)/np.sqrt(3.))
        new_varpi = torch.cat((mean, std), dim=-1).detach()
        return new_varpi

    def store_transition(self, transition):
        state, action, reward, next_state, done = transition
        self.replay_memory.push(state, action, reward, next_state, done)

    def rmset_store_transition(self, transition, valid_idxs=None):
        state, action, reward, next_state, done, kappa, varpi = transition
        self.replay_memory.push(valid_idxs, state, action, reward, next_state, done, kappas=kappa, varpis=varpi)

    def rmset_sample_minibatch(self, batch_size):
        if self.steps > self.warmup:
            #print(self.steps, batch_size)
            state_batch, action_batch, reward_batch, \
            next_state_batch, terminal_batch, kappa_batch, varpi_batch, batch_idxs = self.replay_memory.sample(batch_size)
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            terminal_batch = torch.FloatTensor(terminal_batch).to(self.device).unsqueeze(1)
            kappa_batch = torch.FloatTensor(kappa_batch).to(self.device)
            varpi_batch = torch.FloatTensor(varpi_batch).to(self.device)
            return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, kappa_batch, varpi_batch, batch_idxs
        else:
            return None, None, None, None, None, None, None, None

    def sample_varpis(self, size=10, kappas=None, her=False):
        if self.continuous:
            rand_gen = self._varpi_random_generator if her else self._th_random_generator
            if kappas is None:
                varpi_params = torch.FloatTensor(size, self.varpi_size).to(self.device).uniform_(0., 1. + 1e-6, generator=rand_gen)
                varpi_means, varpi_sigmas = torch.split(varpi_params, self.varpi_size//2, dim=-1)
                ####
                varpi_means = self.min_rel_param + (self.max_rel_param - self.min_rel_param) * varpi_means
                varpi_sigmas = varpi_sigmas * torch.min((self.max_rel_param - varpi_means), (varpi_means - self.min_rel_param)) / np.sqrt(3.)
                ####
                return torch.cat((varpi_means, varpi_sigmas), dim=-1)
            else:
                ## Sample varpi such that its support covers the alpha value
                if not isinstance(kappas, torch.Tensor):
                    kappas = torch.FloatTensor(kappas).to(self.device)
                ####
                size = kappas.shape[0]
                varpi_params = torch.FloatTensor(size, self.varpi_size).to(self.device).uniform_(0., 1. + 1e-6, generator=rand_gen)
                varpi_means, varpi_sigmas = torch.split(varpi_params, self.varpi_size//2, dim=-1)
                ####
                mean_min, mean_Max = 0.5 * (kappas + self.min_rel_param), 0.5 * (self.max_rel_param + kappas)
                varpi_means = mean_min + (mean_Max - mean_min) * varpi_means
                ####
                sigma_min = torch.abs(kappas - varpi_means) / np.sqrt(3.)
                sigma_Max = torch.min((self.max_rel_param - varpi_means), (varpi_means - self.min_rel_param)) / np.sqrt(3.)
                varpi_sigmas = sigma_min + (sigma_Max - sigma_min) * varpi_sigmas
                ####
                return torch.cat((varpi_means, varpi_sigmas), dim=-1)
        else:
            raise NotImplementedError

    def get_td_target(self, states, actions, rewards, next_states, masks, kappas, varpis):
        if self.continuous:
            return self._get_explicit_td_target_continuous(states, actions, rewards, next_states, masks, kappas, varpis)
        else:
            raise NotImplementedError

    def _get_explicit_td_target_continuous(self, states, actions, rewards, next_states, masks, kappas, varpis):
        if states is not None:
            # Prepare the target q batch
            with torch.no_grad():
                #####
                osi_varpis = self.update_uncertainty(states, actions, next_states, self.osi_varpis)
                next_states_action, next_states_log_pi, _ = self.actor(torch.cat((next_states, osi_varpis), dim=-1))
                #####
                aug_next_states = torch.cat(
                    (next_states, kappas), dim=-1
                )
                ## Q1
                next_q1_values = self.critic1_target([ aug_next_states, next_states_action ])
                ## Q2
                next_q2_values = self.critic2_target([ aug_next_states, next_states_action ])
                ## min(Q1, Q2)
                min_next_qf_target = torch.min(next_q1_values, next_q2_values) - self.alpha * next_states_log_pi

            target_q_batch = rewards + \
                self.discount * masks * min_next_qf_target
            return target_q_batch
        else:
            return None

    def update_critics(self, td_target, states, actions, kappas):
        if self.continuous:
            self._update_explicit_critics_continuous(td_target, states, actions, kappas)
        else:
            raise NotImplementedError

    def _update_explicit_critics_continuous(self, td_target, states, actions, kappas):
        if td_target is not None:
            #####
            #varpis_mean, _ = torch.split(kappas, self.varpi_size//2, dim=-1)
            aug_states = torch.cat(
                (states, kappas), dim=-1
            )
            #####
            self.critic1_optim.zero_grad()
            self.critic2_optim.zero_grad()
            q1_batch = self.critic1([ aug_states, actions ])
            q2_batch = self.critic2([ aug_states, actions ])
            value_loss = criterion_A(q1_batch, td_target) + criterion_A(q2_batch, td_target)
            #print(value_loss.shape, q1_batch.shape, q2_batch.shape, td_target.shape)
            #assert kappas.shape == value_loss.shape, "{} vs {}".format(kappas.shape, value_loss.shape)
            value_loss = value_loss.mean()
            value_loss.backward()
            self.critic1_optim.step()
            self.critic2_optim.step()

    def update_actor(self, states, kappas, varpis):
        if self.continuous:
            self._update_explicit_actor_continuous(states, kappas, varpis)
        else:
            raise NotImplementedError

    def _update_explicit_actor_continuous(self, states, kappas, varpis):
        if states is not None:
            self.actor_optim.zero_grad()
            ############################################
            if self.steps > self.threshold and self.use_cvar:
                bs = states.shape[0]
                ####
                mean, sigma = torch.split(self.osi_varpis, self.varpi_size//2, dim=-1)
                ####
                k = self.cvar_samples
                varpis_kappas = torch.FloatTensor(k, mean.shape[-1]).to(self.device).uniform_(0., 1. + 1e-6, generator=self._points_random_generator)
                varpis_kappas = mean + (2. * varpis_kappas.unsqueeze(1).repeat(1, bs, 1) - 1.) * np.sqrt(3.) * sigma
                ############################################
                actions, log_pi, _ = self.actor(
                    torch.cat(
                        (states,
                        self.osi_varpis)
                    , dim=-1)
                )
                ######
                actions = actions.unsqueeze(0).repeat(k, 1, 1)
                ######
                aug_states = torch.cat((states.unsqueeze(0).repeat(k, 1, 1), varpis_kappas), dim=-1)
                ###### estimate_cvar(returns, alpha, numpify=False)
                policy_loss = (
                    (self.alpha * log_pi) - estimate_cvar(
                        torch.min(
                            self.critic1([ aug_states, actions ]),
                            self.critic2([ aug_states, actions ])
                        ),
                        alpha=self.cvar_alpha
                    )
                )
            elif self.steps > self.threshold:
                bs = states.shape[0]
                ####
                mean, sigma = torch.split(self.osi_varpis, self.varpi_size//2, dim=-1)
                ####
                if self.use_ut and self.varpi_size >= 50: # Use UT
                    k = self.sigma_points[self.sp_counter].shape[0]
                    varpis_kappas = mean + (2. * self.sigma_points[self.sp_counter].unsqueeze(1).repeat(1, bs, 1) - 1.) * np.sqrt(3.) * sigma
                    #weights = self.sigma_weights.unsqueeze(1).repeat(1, bs, 1)
                    self.sp_counter = (self.sp_counter + 1) % self.sp_max_num
                else:
                    k = 50#self.varpi_size+1
                    varpis_kappas = torch.FloatTensor(k, mean.shape[-1]).to(self.device).uniform_(0., 1. + 1e-6, generator=self._points_random_generator)
                    varpis_kappas = mean + (2. * varpis_kappas.unsqueeze(1).repeat(1, bs, 1) - 1.) * np.sqrt(3.) * sigma
                    #weights = torch.zeros(k, 1).to(self.device).unsqueeze(1).repeat(1, bs, 1)
                ############################################
                actions, log_pi, _ = self.actor(
                    torch.cat(
                        (states,
                        self.osi_varpis)
                    , dim=-1)
                )
                ######
                actions = actions.unsqueeze(0).repeat(k, 1, 1)
                ######
                aug_states = torch.cat((states.unsqueeze(0).repeat(k, 1, 1), varpis_kappas), dim=-1)
                ######
                policy_loss = (
                    (self.alpha * log_pi) - torch.min(
                        self.critic1([ aug_states, actions ]),
                        self.critic2([ aug_states, actions ])
                    ).mean(0)
                )
            else:
                actions, log_pi, _ = self.actor(
                    torch.cat(
                        (states,
                        self.osi_varpis)
                    , dim=-1)
                )
                ######
                aug_states = torch.cat((states, kappas), dim=-1)
                ######
                policy_loss = (
                    (self.alpha * log_pi) - torch.min(
                        self.critic1([ aug_states, actions ]),
                        self.critic2([ aug_states, actions ])
                    )#.mean(0)
                )
            #print(policy_loss.shape)
            #policy_loss = policy_loss.mean(0)
            #print(policy_loss.shape)
            policy_loss = policy_loss.mean()
            ############################################
            policy_loss.backward()
            self.actor_optim.step()
            ############################################
            self.osi_varpis = None

    def update_systemID(self, states, actions, next_states, kappas, varpis, idxs):
        if states is not None:
            self.systemID_optim.zero_grad()
            ############################################
            prev_states, prev_actions, _, prev_next_states, _, prev_kappas, prev_varpis = self.replay_memory.get_previous_transition(idxs)
            prev_states = torch.FloatTensor(prev_states).to(self.device)
            prev_actions = torch.FloatTensor(prev_actions).to(self.device)
            prev_next_states = torch.FloatTensor(prev_next_states).to(self.device)
            prev_kappas = torch.FloatTensor(prev_kappas).to(self.device)
            prev_varpis = torch.FloatTensor(prev_varpis).to(self.device)
            ## Make sure each transition originates from the same batch element trajectory
            mask = ((prev_kappas == kappas).all(dim=-1) * (prev_next_states == states).all(dim=-1))
            #print(mask.shape)
            ############################################
            bs = states.shape[0]
            mini_bs = bs // self.SI_ensemble_size
            ####
            ## Use everything to optimize the systemID regardless of trajectory correspondance
            kappas = torch.cat(
                (
                    kappas, prev_kappas
                ), dim=0
            )
            ####
            mean, sigma = torch.split(
                torch.cat(
                    (
                        varpis, prev_varpis
                    ), dim=0
                ), self.varpi_size//2, dim=-1
            )
            ####
            aug_states = torch.cat(
                (
                    torch.cat((states, actions, next_states, varpis), dim=-1),
                    torch.cat((prev_states, prev_actions, prev_next_states, prev_varpis), dim=-1)
                ), dim=0
            )
            predicted_sigmas = self.systemID(aug_states) # (SI_ensemble_size x 2*bs x kappa_size)
            ####
            k = mean.shape[-1]
            loss = torch.cat([(mean + np.sqrt(3.) * predicted_sigmas[i] - kappas).view(2, bs, k)[:, i * mini_bs:(i+1) * mini_bs, :].square().sum(dim=-1).view(-1)
                for i in range(self.SI_ensemble_size)], dim=0)
            ####
            loss = loss.mean()
            ############################################
            loss.backward()
            self.systemID_optim.step()
            #########
            #########
            #####
            self.osi_varpis = self.update_uncertainty(prev_states, prev_actions, prev_next_states, varpis)
            ## If the previous transition does not match the current transition,
            ## then simply use the original varpis
            self.osi_varpis[~mask] = varpis[~mask]
            assert self.osi_varpis.shape == varpis.shape

    def update_temperature(self, states, varpis):
        if self.continuous:
            return self._update_temperature_continuous(states, varpis)
        else:
            raise NotImplementedError

    def _update_temperature_continuous(self, states, varpis):
        if states is not None and self.automatic_entropy_tuning:
            self.alpha_optim.zero_grad()

            with torch.no_grad():
                ######
                aug_states = torch.cat(
                    (states, varpis), dim=-1
                )
                _, log_pi, _ = self.actor(aug_states)
            alpha_loss = - (self.log_alpha * (log_pi + self.target_entropy)).mean()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp().item()
        return self.alpha

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
        self.systemID.train()

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic1_target.eval()
        self.critic2.eval()
        self.critic2_target.eval()
        self.systemID.eval()

    def to_cuda(self):
        self.actor.cuda()
        self.critic1.cuda()
        self.critic1_target.cuda()
        self.critic2.cuda()
        self.critic2_target.cuda()
        self.systemID.cuda()

    def to_cpu(self):
        self.actor.cpu()
        self.critic1.cpu()
        self.critic1_target.cpu()
        self.critic2.cpu()
        self.critic2_target.cpu()
        self.systemID.cpu()

    def set_seed(self, s):
        np.random.seed(s)
        torch.manual_seed(s)
        if self.device == torch.device("cuda"):#self.use_cuda:
            torch.cuda.manual_seed(s)

    def load_weights(self, output_path):
        if output_path is None: return
        if output_path[-1] == "/":
            output_path = output_path[:-1]
        print("++ Load models from ", output_path)

        print(self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output_path))
        ))

        self.critic1.load_state_dict(
            torch.load('{}/critic1.pkl'.format(output_path))
        )

        self.critic2.load_state_dict(
            torch.load('{}/critic2.pkl'.format(output_path))
        )

        print(self.systemID.load_state_dict(
            torch.load('{}/systemID.pkl'.format(output_path))
        ))


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
            self.systemID.state_dict(),
            '{}/systemID.pkl'.format(output_path)
        )
