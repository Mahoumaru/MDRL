import torch
from torch.nn import MSELoss, L1Loss, Softmax
import inspect
import numpy as np

#from pytorch_rl_collection.model_networks.model_networks import SacActor as Actor
#from pytorch_rl_collection.model_networks.model_networks import SacCritic as Critic
from pytorch_rl_collection.model_networks.model_networks import SacSolver as Solver
from pytorch_rl_collection.replay_buffers.replay_memory import ReplayMemory
from pytorch_rl_collection.utils import *#hard_update, soft_update
from pytorch_rl_collection.model_networks.model_networks import DynamicsNet, EnsembleOSINet

from torch.distributions import Normal

from torchoptim.optimizers import AdaTerm

from scipy.stats import qmc
from torch.quasirandom import SobolEngine

###########################################
criterion_A = MSELoss(reduction='none')
criterion_B = L1Loss(reduction='none')
softmax_fct = Softmax(dim=1)

###########################################
def kl_divergence(pi0, pi1):
    (mu0, sig0) = pi0
    (mu1, sig1) = pi1
    var0, var1 = sig0.square()+1e-6, sig1.square()+1e-6
    D = (mu1 - mu0).square().div(var1)#.sum(-1, keepdim=True)
    Tr = var0.div(var1)
    lndet = var1.log() - var0.log()
    return 0.5 * (Tr + D + lndet).sum(-1, keepdim=True)

###########################################
#### MD updates
def inverse_sigmoid(y, eps=1e-8):
    return torch.log(y + eps) - torch.log(1. - y + eps)

def mirror_descent_update(variables, lr=1e-3, fn=torch.sigmoid, inv_fn=inverse_sigmoid):
    for v in variables:
        g = v.grad
        if g is not None:
           v.data = fn(inv_fn(v.data) - lr * g)

###########################################
class EMDSAC_AGENT:
    def __init__(self, states_dim, actions_dim, domains_dim, args, OptimCls=torch.optim.Adam,
                 ReplayBufferCls=ReplayMemory, lambda_steplength=1e-5, value_range=None, add_noise=False,
                 model_type=""):
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
        self.beta = args.alpha
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
        self.lambda_steplength = lambda_steplength
        self.SI_ensemble_size = args.n_osi
        self.n_proposed_varpis = 30

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
                #self.sampler = qmc.Halton(d=domains_dim, scramble=True, seed=1234 * self.seed)
                self.sampler = SobolEngine(dimension=domains_dim, scramble=True, seed=1234 * self.seed)
        else:
            self.varpi_size = self.n_regions
        ####
        self.osi = None
        ####
        sim_observation_noise = 5e-3
        #value_range = env.robot.get_config('dclaw').qpos_range
        if value_range is not None and add_noise:
            self.noise_amplitude = torch.FloatTensor(sim_observation_noise * np.ptp(value_range, axis=1)).to(self.device)
        else:
            self.noise_amplitude = None
        ####
        print("++ Noise Amplitude: ", self.noise_amplitude)

    def reset_steps(self):
        self.steps = 0

    def set_warmup(self, warmup):
        self.warmup = warmup

    def add_noise(self, states):
        if self.noise_amplitude is not None:
            states[:, :9] = states[:, :9] + self.noise_amplitude * torch.FloatTensor(states.shape[0], 9).to(self.device).uniform_(-0.5, 0.5, generator=self._points_random_generator)
        return states

    def set_varpi_ranges(self, ranges):
        self.min_rel_param, self.max_rel_param = ranges.T
        assert (self.min_rel_param <= self.max_rel_param).all(), "The ranges must be a proper interval, i.e. [m, M] s.t. m <= M. Got m={} and M={} instead.".format(self.min_rel_param, self.max_rel_param)
        self.min_rel_param, self.max_rel_param = torch.FloatTensor(self.min_rel_param).to(self.device), torch.FloatTensor(self.max_rel_param).to(self.device)
        ####
        #self.p_k = (self.max_rel_param - self.min_rel_param) / 2.
        #self.p_b = (self.max_rel_param + self.min_rel_param) / 2.

    def initialize(self):
        # An utility function to be able to initialize
        # everything at once
        self.initialize_critics()
        self.initialize_actor()
        self.initialize_solver()
        self.initialize_entropy_temperature()
        self.initialize_targets()
        self.initialize_systemID()
        self.initialize_replay_buffer_set(size=self.n_regions)

    def initialize_critics(self):
        if self.explicit_scal:
            self._initialize_explicit_critics()
            self.get_td_target = self._get_explicit_td_target
            self.update_critics = self._update_explicit_critics
            self.update_actor = self._update_explicit_actor
            self.update_solver = self._update_explicit_solver
        else:
            self._initialize_implicit_critics()
            self.get_td_target = self._get_implicit_td_target
            self.update_critics = self._update_implicit_critics
            self.update_actor = self._update_implicit_actor
            self.update_solver = self._update_implicit_solver

    def _initialize_implicit_critics(self):
        if self.continuous:
            net_cfg = self.net_cfg.copy()
            net_cfg['nb_varpi'] = self.varpi_size
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

    def _initialize_explicit_critics(self):
        if self.continuous:
            net_cfg = self.net_cfg.copy()
            net_cfg['nb_varpi'] = self.varpi_size + (self.varpi_size // 2)
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

    def initialize_solver(self):
        net_cfg = self.net_cfg.copy()
        net_cfg['solver_type'] = "Deterministic"
        net_cfg['sigmoid_fn'] = False
        net_cfg['double'] = True
        net_cfg['nb_kappa'] = self.varpi_size // 2
        self.solver = Solver(self.states_dim, **net_cfg).to(self.device)
        if self.verbose:
            print("++ Solver Initialized with following structure:")
            print(self.solver)
            print("")
        self.solver_optim  = self.OptimCls(self.solver.parameters(), lr=self.actor_lr)

    def initialize_entropy_temperature(self):
        if self.automatic_entropy_tuning:
            self.target_entropy = - self.actions_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = self.OptimCls([self.log_alpha], lr=self.entropy_temp_lr)

    def initialize_targets(self):
        if self.explicit_scal:
            self._initialize_explicit_targets()
        else:
            self._initialize_implicit_targets()
        hard_update(self.critic1_target, self.critic1) # Make sure target is with the same weight
        hard_update(self.critic2_target, self.critic2)
        if self.verbose:
            print("++ Hard copy done.")

    def _initialize_implicit_targets(self):
        if self.continuous:
            net_cfg = self.net_cfg.copy()
            net_cfg['nb_varpi'] = self.varpi_size
            self.critic1_target = self.CriticCls(self.states_dim, self.actions_dim, output_dim=1, **net_cfg).to(self.device)
            self.critic2_target = self.CriticCls(self.states_dim, self.actions_dim, output_dim=1, **net_cfg).to(self.device)
        else:
            self.critic1_target = self.CriticCls(self.states_dim + self.varpi_size, self.actions_dim, output_dim=1, **self.net_cfg).to(self.device)
            self.critic2_target = self.CriticCls(self.states_dim + self.varpi_size, self.actions_dim, output_dim=1, **self.net_cfg).to(self.device)
        if self.verbose:
            print("++ Both Target Critic networks Initialized.")

    def _initialize_explicit_targets(self):
        if self.continuous:
            net_cfg = self.net_cfg.copy()
            net_cfg['nb_varpi'] = self.varpi_size + (self.varpi_size // 2)
            net_cfg['per_layer_varpi'] = False
            self.critic1_target = self.CriticCls(self.states_dim, self.actions_dim, output_dim=1, **net_cfg).to(self.device)
            self.critic2_target = self.CriticCls(self.states_dim, self.actions_dim, output_dim=1, **net_cfg).to(self.device)
        else:
            self.critic1_target = self.CriticCls(self.states_dim + self.varpi_size, self.actions_dim, output_dim=1, **self.net_cfg).to(self.device)
            self.critic2_target = self.CriticCls(self.states_dim + self.varpi_size, self.actions_dim, output_dim=1, **self.net_cfg).to(self.device)
        if self.verbose:
            print("++ Both Target Critic networks Initialized.")

    def initialize_systemID(self):
        self.systemID = EnsembleOSINet(self.SI_ensemble_size,
            self.states_dim, self.actions_dim,
            nb_latent=self.varpi_size // 2, hidden_layers=[200, 200, 200],
            #osi_type="Gaussian", init_w=3e-3,
            device=self.device,
            verbose=False
        )#.to(self.device)
        self.forwardDyn = DynamicsNet(
            self.states_dim, self.actions_dim,
            nb_latent=self.varpi_size // 2, hidden_layers=[200, 200, 200, 200],
            init_w=3e-3, verbose=False
        ).to(self.device)
        if self.verbose:
            print("++ System ID network Initialized with following structure:")
            print(self.systemID)
            print("")
            print("++ Forward dynamics network Initialized with following structure:")
            print(self.forwardDyn)
            print("")
        self.systemID_optim = self.systemID.get_optimizer(self.OptimCls, lr=self.critic_lr)
        #self.OptimCls(self.systemID.parameters(), lr=self.critic_lr)
        self.forwardDyn_optim = self.OptimCls(self.forwardDyn.parameters(), lr=self.critic_lr)

    def initialize_replay_buffer(self):
        self.replay_memory = self.ReplayBufferCls(capacity=self.rmsize, seed=self.seed, window_length=self.window_length)
        if self.verbose:
            print("++ Replay Memory Initialized.")

    def initialize_replay_buffer_set(self, size):
        self.replay_memory = self.ReplayBufferCls(size=size, capacity=self.rmsize, seed=self.seed, window_length=self.window_length)
        #, numpy_buffer=True)
        if self.verbose:
            print("++ Replay Memory Initialized.")

    def initialize_lambda(self):
        self.homotopy_lambda = 0.
        if self.verbose:
            print("++ Homotopy parameter Initialized with steplength: {}".format(self.lambda_steplength))

    def get_ccs_varpi(self, state, varpi):
        if not isinstance(varpi, torch.Tensor):
            varpi = torch.FloatTensor(varpi).to(self.device)
        if not isinstance(state, torch.Tensor):
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
        ####
        wrapped_mu, wrapped_sig = self.solver(torch.cat((state, varpi), dim=-1))
        mu = self.min_rel_param + 0.5 * (1. + wrapped_mu) * (self.max_rel_param - self.min_rel_param)
        sig = (0.5 * (1. + wrapped_sig) * (self.max_rel_param - self.min_rel_param) / (2.*np.sqrt(3.))) * (1. - wrapped_mu.abs())
        return torch.cat((mu, sig), dim=-1), None

    def select_action(self, state, varpi, kappa=None, is_training=True, squeeze=True):
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
            state = self.add_noise(state)
            if is_training:
                #####
                aug_state = torch.cat((state, varpi), dim=-1)
                action, _, _ = self.actor(aug_state)
                #####
                action = to_numpy(
                    action
                )
            else:
                aug_state = torch.cat((state, varpi), dim=-1)
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

    def update_uncertainty(self, state, action, next_state, varpi):
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
        aug_state = torch.cat((state, action, next_state, varpi), dim=-1)
        new_varpi = self.systemID.get_varpi(aug_state, bounds=(self.min_rel_param, self.max_rel_param))
        #####
        mean, std = torch.split(new_varpi, self.varpi_size//2, dim=-1)
        ####
        # Ensure within domain coverage
        std = torch.min(std, torch.min(self.max_rel_param - mean, mean - self.min_rel_param)/np.sqrt(3.))
        new_varpi = torch.cat((mean, std), dim=-1).detach()
        return new_varpi

    def get_new_varpi(self, state, action, next_state, varpi):
        if not isinstance(varpi, torch.Tensor):
            varpi = torch.FloatTensor(varpi).to(self.device)
        if varpi.shape[-1] != self.varpi_size:
            varpi = torch.cat((varpi, torch.zeros_like(varpi)), dim=-1)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        #####
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
        if len(varpi.shape) == 1:
            varpi = varpi.unsqueeze(0).repeat(state.shape[0], 1)
        if state.shape[0] != varpi.shape[0]:
            varpi = varpi.repeat(state.shape[0], 1)
        #####
        assert state.shape[0] == varpi.shape[0], "{} vs {}".format(state.shape[0], varpi.shape[0])
        #####
        with torch.no_grad():
            _, uncertainty, _ = self.osi(
                state,
                action,
                next_state,
                varpi
            )
        ######
        mu, sig = uncertainty.mean, uncertainty.stddev
        a = torch.min(torch.max(self.min_rel_param, mu - np.sqrt(3.) * sig), self.max_rel_param)
        b = torch.max(torch.min(self.max_rel_param, mu + np.sqrt(3.) * sig), self.min_rel_param)
        ####
        a, b = to_numpy(a), to_numpy(b)
        return np.concatenate(
            (0.5 * (a + b), np.sqrt(1./12.) * (b - a)), axis=-1
        )

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
            next_state_batch, terminal_batch, kappa_batch, varpi_batch in self.replay_memory.sample(batch_size):
                state_batch = torch.FloatTensor(state_batch).to(self.device)
                next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
                action_batch = torch.FloatTensor(action_batch).to(self.device)
                reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
                terminal_batch = torch.FloatTensor(terminal_batch).to(self.device).unsqueeze(1)
                kappa_batch = torch.FloatTensor(kappa_batch).to(self.device)
                varpi_batch = torch.FloatTensor(varpi_batch).to(self.device)
                yield state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, kappa_batch, varpi_batch
        else:
            yield None, None, None, None, None, None, None

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

    def compare_varpis_ccs(self, varpi1, varpi2, states, target=False):
        if varpi1.shape != varpi2.shape:
            if len(varpi2.shape) == 1:
                varpi2 = varpi2.unsqueeze(0).repeat(varpi1.shape[0], 1) ## bs x 2*dim
            else:
                varpi2 = varpi2.repeat(varpi1.shape[0], 1) ## bs x 2*dim
        assert varpi1.shape == varpi2.shape, "varpi1 shape = {} vs varpi2 shape = {}".format(varpi1.shape, varpi2.shape)
        ####
        bs = varpi1.shape[0]
        k = self.sigma_points[self.sp_counter].shape[0]
        d = self.varpi_size // 2
        ###########
        mean, sigma = torch.split(varpi1, self.varpi_size//2, dim=-1)
        varpis_kappas = mean + (2. * self.sigma_points[self.sp_counter].unsqueeze(1).repeat(1, bs, 1) - 1.) * np.sqrt(3.) * sigma ## k x bs x d
        del mean; del sigma
        #print("varpis_kappas:", varpis_kappas.shape)
        ###########
        sampled_varpis = torch.cat((varpi1.unsqueeze(1), varpi2.unsqueeze(1)), dim=1) ## bs x 2 x 2*d
        #print("sampled_varpis:", sampled_varpis.shape)
        ####
        actions, log_pi = self.actor.evaluate(
            torch.cat(
                (states.unsqueeze(1).repeat(1, 2, 1), ## bs x 2 x |S|
                sampled_varpis) ## bs x 2 x 2*d
            , dim=-1),
            self.actor.get_standard_normal(sample_shape=(bs, self.actions_dim)).unsqueeze(1).repeat(1, 2, 1)
        ) ## bs x 2 x |A|
        aug_actions = actions.unsqueeze(0).repeat(k, 1, 1, 1) ## k x bs x 2 x |A|
        #print("aug_actions:", aug_actions.shape, "; log_pi:", log_pi.shape)
        ####
        aug_states = torch.cat(
            (states.unsqueeze(0).repeat(k, 1, 1), ## k x bs x |S|
            varpis_kappas, ## k x bs x d,
            varpi1.unsqueeze(0).repeat(k, 1, 1)
            )
        , dim=-1).unsqueeze(-2).repeat(1, 1, 2, 1) ## k x bs x 2 x (|S|+d)
        #print("aug_states:", aug_states.shape)
        del varpis_kappas
        ######
        if target:
            vals = torch.min(self.critic1_target([ aug_states, aug_actions ]).mean(0), self.critic2_target([ aug_states, aug_actions ]).mean(0))
        else:
            vals = torch.min(self.critic1([ aug_states, aug_actions ]).mean(0), self.critic2([ aug_states, aug_actions ]).mean(0))
        vals = vals - self.alpha * log_pi  ## bs x 2 x 1
        #print("vals", vals.shape)
        ######
        del aug_states, aug_actions
        ######
        max_idxs = torch.max(vals, dim=1).indices
        #print(max_idxs.shape)
        opt_varpis = sampled_varpis.gather(1, max_idxs.unsqueeze(2).repeat(1, 1, 2*d)).squeeze(1)
        actions = actions.gather(1, max_idxs.unsqueeze(2).repeat(1, 1, self.actions_dim)).squeeze(1)
        log_pi = log_pi.gather(1, max_idxs.unsqueeze(2).repeat(1, 1, 1)).squeeze(1)
        #print(opt_varpis.shape, actions.shape, log_pi.shape)
        ####
        assert opt_varpis.shape == varpi1.shape
        #print(jkjkl)
        return opt_varpis, actions, log_pi

    def _get_implicit_td_target(self, states, actions, rewards, next_states, masks, kappas, her_varpis, varpis=None):
        if self.continuous:
            return self._get_implicit_td_target_continuous(states, actions, rewards, next_states, masks, kappas, her_varpis, varpis)
        else:
            raise NotImplementedError

    def _get_implicit_td_target_continuous(self, states, actions, rewards, next_states, masks, kappas, her_varpis, varpis=None):
        if states is not None:
            # Prepare the target q batch
            with torch.no_grad():
                #####
                ccs_varpis, _ = self.get_ccs_varpi(next_states, her_varpis)
                #next_states = next_states.repeat(2, 1)
                next_states_action, next_states_log_pi, _ = self.actor(torch.cat((next_states, ccs_varpis), dim=-1))#(aug_next_states)#
                #####
                aug_next_states = torch.cat(
                    (next_states, her_varpis), dim=-1
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

    def _get_explicit_td_target(self, states, actions, rewards, next_states, masks, kappas, her_varpis, varpis=None):
        if self.continuous:
            return self._get_explicit_td_target_continuous(states, actions, rewards, next_states, masks, kappas, her_varpis, varpis)
        else:
            raise NotImplementedError

    def _get_explicit_td_target_continuous(self, states, actions, rewards, next_states, masks, kappas, her_varpis, varpis=None):
        if states is not None:
            # Prepare the target q batch
            with torch.no_grad():
                #####
                #next_states = self.add_noise(next_states)
                #####
                ccs_varpis, _ = self.get_ccs_varpi(next_states, her_varpis)
                #####
                next_states_action, next_states_log_pi, _ = self.actor(torch.cat((next_states, ccs_varpis), dim=-1))
                #ccs_varpis, next_states_action, next_states_log_pi = self.compare_varpis_ccs(her_varpis, ccs_varpis, next_states, target=False)
                #####
                aug_next_states = torch.cat(
                    (next_states, kappas, her_varpis), dim=-1
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

    def _update_implicit_critics(self, td_target, states, actions, kappas, her_varpis):
        if self.continuous:
            self._update_implicit_critics_continuous(td_target, states, actions, kappas, her_varpis)
        else:
            raise NotImplementedError

    def _update_implicit_critics_continuous(self, td_target, states, actions, kappas, her_varpis):
        if td_target is not None:
            #####
            #varpis_mean, _ = torch.split(kappas, self.varpi_size//2, dim=-1)
            aug_states = torch.cat(
                (states, her_varpis), dim=-1
            )
            #actions = actions.repeat(2, 1)
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

    def _update_explicit_critics(self, td_target, states, actions, kappas, her_varpis):
        if self.continuous:
            self._update_explicit_critics_continuous(td_target, states, actions, kappas, her_varpis)
        else:
            raise NotImplementedError

    def _update_explicit_critics_continuous(self, td_target, states, actions, kappas, her_varpis):
        if td_target is not None:
            #####
            #states = self.add_noise(states)
            #####
            #varpis_mean, _ = torch.split(varpis, self.varpi_size//2, dim=-1)
            aug_states = torch.cat(
                (states, kappas, her_varpis), dim=-1
            )
            #####
            self.critic1_optim.zero_grad()
            self.critic2_optim.zero_grad()
            q1_batch = self.critic1([ aug_states, actions ])
            q2_batch = self.critic2([ aug_states, actions ])
            value_loss = criterion_A(q1_batch, td_target) + criterion_A(q2_batch, td_target)
            #assert kappas.shape == value_loss.shape, "{} vs {}".format(kappas.shape, value_loss.shape)
            value_loss = value_loss.mean()
            value_loss.backward()
            self.critic1_optim.step()
            self.critic2_optim.step()

    def _update_implicit_actor(self, states, kappas, her_varpis):
        if self.continuous:
            self._update_implicit_actor_continuous(states, kappas, her_varpis)
        else:
            raise NotImplementedError

    def _update_implicit_actor_continuous(self, states, kappas, her_varpis):
        if states is not None:
            self.actor_optim.zero_grad()
            ############################################
            actions, log_pi, _ = self.actor(
                torch.cat(
                    (states,
                    her_varpis)
                , dim=-1)
            )
            ######
            aug_states = torch.cat((states, her_varpis), dim=-1)
            ######
            policy_loss = (self.alpha * log_pi) - torch.min(self.critic1([ aug_states, actions ]), self.critic2([ aug_states, actions ]))
            policy_loss = policy_loss.mean()
            ############################################
            policy_loss.backward()
            self.actor_optim.step()

    def _update_explicit_actor(self, states, kappas, her_varpis):
        if self.continuous:
            self._update_explicit_actor_continuous(states, kappas, her_varpis)
        else:
            raise NotImplementedError

    def _update_explicit_actor_continuous(self, states, kappas, her_varpis):
        if states is not None:
            self.actor_optim.zero_grad()
            ############################################
            bs = states.shape[0]
            ####
            mean, sigma = torch.split(her_varpis, self.varpi_size//2, dim=-1)
            ####
            if self.use_ut: # Use UT
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
                    her_varpis)
                , dim=-1)
            )
            ######
            #ccs_varpis, ccs_log_xi = self.get_ccs_varpi(states, actions, her_varpis)
            ######
            actions = actions.unsqueeze(0).repeat(k, 1, 1)
            ######
            aug_states = torch.cat(
                (states.unsqueeze(0).repeat(k, 1, 1),
                varpis_kappas,
                her_varpis.unsqueeze(0).repeat(k, 1, 1))
            , dim=-1)
            ######
            policy_loss = (
                (self.alpha * log_pi) - torch.min(
                    self.critic1([ aug_states, actions ]),
                    self.critic2([ aug_states, actions ])
                ).mean(0)
            )
            policy_loss = policy_loss.mean()
            ############################################
            policy_loss.backward()
            self.actor_optim.step()

    def get_previous_transitions(self, current_idxs, current_states, current_kappas):
        state_batch = None
        for state_mbatch, action_mbatch, _, \
        next_state_mbatch, _, kappa_mbatch, _ in self.replay_memory.get_previous_transition(current_idxs):
            if state_batch is None:
                state_batch = torch.FloatTensor(state_mbatch).to(self.device)
                next_state_batch = torch.FloatTensor(next_state_mbatch).to(self.device)
                action_batch = torch.FloatTensor(action_mbatch).to(self.device)
                #reward_batch = torch.FloatTensor(reward_mbatch).to(self.device).unsqueeze(1)
                kappa_batch = torch.FloatTensor(kappa_mbatch).to(self.device)
                #varpi_batch = torch.FloatTensor(varpi_mbatch).to(self.device)
            else:
                state_batch = torch.cat((state_batch, torch.FloatTensor(state_mbatch).to(self.device)), dim=0)
                next_state_batch = torch.cat((next_state_batch, torch.FloatTensor(next_state_mbatch).to(self.device)), dim=0)
                action_batch = torch.cat((action_batch, torch.FloatTensor(action_mbatch).to(self.device)), dim=0)
                #reward_batch = torch.cat((reward_batch, torch.FloatTensor(reward_mbatch).to(self.device).unsqueeze(1)), dim=0)
                kappa_batch = torch.cat((kappa_batch, torch.FloatTensor(kappa_mbatch).to(self.device)), dim=0)
                #varpi_batch = ((varpi_batch, torch.FloatTensor(varpi_mbatch).to(self.device)))
        mask_batch = (next_state_batch == current_states).all(dim=-1) * (kappa_batch == current_kappas).all(dim=-1)
        return state_batch[mask_batch], action_batch[mask_batch], next_state_batch[mask_batch], mask_batch

    def _update_implicit_solver(self, states, her_varpis):
        if self.continuous:
            self._update_implicit_solver_continuous(states, her_varpis)
        else:
            raise NotImplementedError

    def _update_implicit_solver_continuous(self, states, her_varpis):
        if states is not None:
            self.solver_optim.zero_grad()
            ############################################
            wrapped_mu, wrapped_sig = self.solver(torch.cat((states, her_varpis), dim=-1))
            mu = self.min_rel_param + 0.5 * (1. + wrapped_mu) * (self.max_rel_param - self.min_rel_param)
            sig = (0.5 * (1. + wrapped_sig) * (self.max_rel_param - self.min_rel_param) / (2.*np.sqrt(3.))) * (1. - wrapped_mu.abs())
            ccs_varpis = torch.cat((mu, sig), dim=-1)
            #print(ccs_varpis.shape)
            ####
            actions, log_pi, _ = self.actor(torch.cat((states, ccs_varpis), dim=-1))
            ######
            aug_states = torch.cat((states, her_varpis), dim=-1)
            ######
            solver_loss = (self.alpha * log_pi) - torch.min(self.critic1([ aug_states, actions ]), self.critic2([ aug_states, actions ]))
            solver_loss = solver_loss.mean()
            ############################################
            solver_loss.backward()
            self.solver_optim.step()

    def _update_explicit_solver(self, states, her_varpis):
        if self.continuous:
            self._update_explicit_solver_continuous(states, her_varpis)
        else:
            raise NotImplementedError

    def _update_explicit_solver_continuous(self, states, her_varpis):
        if states is not None:
            self.solver_optim.zero_grad()
            ############################################
            #states, actions, next_states, masks = self.get_previous_transitions(current_idxs, current_states, current_kappas)
            ############################################
            bs = states.shape[0]
            ####
            mean, sigma = torch.split(her_varpis, self.varpi_size//2, dim=-1)
            ####
            if self.use_ut: # Use UT
                k = self.sigma_points[self.sp_counter].shape[0]
                varpis_kappas = mean + (2. * self.sigma_points[self.sp_counter].unsqueeze(1).repeat(1, bs, 1) - 1.) * np.sqrt(3.) * sigma
                self.sp_counter = (self.sp_counter + 1) % self.sp_max_num
            else:
                k = 50
                varpis_kappas = torch.FloatTensor(k, mean.shape[-1]).to(self.device).uniform_(0., 1. + 1e-6, generator=self._points_random_generator)
                varpis_kappas = mean + (2. * varpis_kappas.unsqueeze(1).repeat(1, bs, 1) - 1.) * np.sqrt(3.) * sigma
            ############################################
            wrapped_mu, wrapped_sig = self.solver(torch.cat((states, her_varpis), dim=-1))
            mu = self.min_rel_param + 0.5 * (1. + wrapped_mu) * (self.max_rel_param - self.min_rel_param)
            sig = (0.5 * (1. + wrapped_sig) * (self.max_rel_param - self.min_rel_param) / (2.*np.sqrt(3.))) * (1. - wrapped_mu.abs())
            ccs_varpis = torch.cat((mu, sig), dim=-1)
            #print(ccs_varpis.shape)
            ####
            actions, log_pi, _ = self.actor(torch.cat((states, ccs_varpis), dim=-1))
            ######
            aug_actions = actions.unsqueeze(0).repeat(k, 1, 1)
            #####
            aug_states = torch.cat(
                (
                    states.unsqueeze(0).repeat(k, 1, 1),
                    varpis_kappas,
                    her_varpis.unsqueeze(0).repeat(k, 1, 1)
                ), dim=-1
            )
            ####
            solver_loss = (self.alpha * log_pi) - torch.min(
                self.critic1([ aug_states, aug_actions ]),
                self.critic2([ aug_states, aug_actions ])
            ).mean(0)
            ####
            #print(solver_loss.shape)
            solver_loss = solver_loss.mean()
            ############################################
            solver_loss.backward()
            self.solver_optim.step()

    def update_systemID(self, states, actions, next_states, kappas, varpis):
        if states is not None:
            self.systemID_optim.zero_grad()
            self.forwardDyn_optim.zero_grad()
            ############################################
            bs = states.shape[0]
            mini_bs = bs // self.SI_ensemble_size
            ####
            target_mean, target_std = torch.split(varpis, self.varpi_size//2, dim=-1)
            ####
            aug_states = torch.cat((states, actions, next_states, varpis), dim=-1)
            ######################
            predicted_kappas = self.systemID(aug_states)
            k = predicted_kappas.shape[-1]
            predicted_kappas = self.min_rel_param + 0.5 * (self.max_rel_param - self.min_rel_param) * (1. + predicted_kappas)
            ####
            mean, std = predicted_kappas.mean(dim=0), (predicted_kappas.std(dim=0) + 1e-10)
            ####
            dyn_log_likelihood = []
            for i in range(self.SI_ensemble_size):
                dyn_log_likelihood.append(
                    (self.forwardDyn.evaluate(states, actions, predicted_kappas[i], next_states)#.view(-1)
                    - Normal(mean, std).log_prob(predicted_kappas[i]).sum(-1, keepdim=True)
                    + Normal(target_mean, target_std + 1e-10).log_prob(predicted_kappas[i]).sum(-1, keepdim=True))[i * mini_bs:(i+1) * mini_bs, :]
                )
            dyn_log_likelihood = torch.cat(dyn_log_likelihood, dim=0)
            #print(dyn_log_likelihood.shape)
            ####
            loss = - dyn_log_likelihood.mean()
            ############################################
            loss.backward()
            self.systemID_optim.step()
            self.forwardDyn_optim.step()

    def update_temperature(self, states, kappas, her_varpis):
        if self.continuous:
            self._update_temperature_continuous(states, kappas, her_varpis)
        else:
            raise NotImplementedError

    def _update_temperature_continuous(self, states, kappas, her_varpis):
        if states is not None and self.automatic_entropy_tuning:
            self.alpha_optim.zero_grad()

            with torch.no_grad():
                ######
                aug_states = torch.cat(
                    (states, her_varpis), dim=-1
                )
                _, log_pi, _ = self.actor(aug_states)
                #_, ccs_log_xi = self.get_ccs_varpi(states, actions, her_varpis)
            alpha_loss = - (self.log_alpha * (log_pi + self.target_entropy)).mean()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp().item()

    def update_target_networks(self, update):
        #if self.steps > self.warmup:
        if update:
            soft_update(self.critic1_target, self.critic1, self.tau)
            soft_update(self.critic2_target, self.critic2, self.tau)
            #print(jklkl)

    def increase_lambda(self):
        if self.steps > self.warmup:
            self.homotopy_lambda += self.lambda_steplength
            self.homotopy_lambda = min(self.homotopy_lambda, 1.)

    ### Utilities functions:
    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic1_target.train()
        self.critic2.train()
        self.critic2_target.train()
        self.systemID.train()
        self.solver.train()
        self.forwardDyn.train()

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic1_target.eval()
        self.critic2.eval()
        self.critic2_target.eval()
        self.systemID.eval()
        self.solver.eval()
        self.forwardDyn.eval()

    def to_cuda(self):
        self.actor.cuda()
        self.critic1.cuda()
        self.critic1_target.cuda()
        self.critic2.cuda()
        self.critic2_target.cuda()
        self.systemID.cuda()
        self.solver.cuda()
        self.forwardDyn.cuda()

    def to_cpu(self):
        self.actor.cpu()
        self.critic1.cpu()
        self.critic1_target.cpu()
        self.critic2.cpu()
        self.critic2_target.cpu()
        self.systemID.cpu()
        self.solver.cpu()
        self.forwardDyn.cpu()

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

        self.forwardDyn.load_state_dict(
            torch.load('{}/forwardDyn.pkl'.format(output_path))
        )

        self.solver.load_state_dict(
            torch.load('{}/solver.pkl'.format(output_path))
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
            self.systemID.state_dict(),
            '{}/systemID.pkl'.format(output_path)
        )
        torch.save(
            self.forwardDyn.state_dict(),
            '{}/forwardDyn.pkl'.format(output_path)
        )
        torch.save(
            self.solver.state_dict(),
            '{}/solver.pkl'.format(output_path)
        )
