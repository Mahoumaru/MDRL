import torch
from torch.nn import MSELoss
from torch.nn.utils.rnn import pad_sequence
import inspect
import numpy as np

#from pytorch_rl_collection.model_networks.model_networks import SacActor as Actor
#from pytorch_rl_collection.model_networks.model_networks import SacCritic as Critic
from pytorch_rl_collection.replay_buffers.replay_memory import ReplayMemoryLSTM2
from pytorch_rl_collection.utils import *#hard_update, soft_update

###########################################
criterion = MSELoss()

###########################################
class SAC_RNN_AGENT:
    def __init__(self, states_dim, actions_dim, args, OptimCls=torch.optim.Adam,
                 ReplayBufferCls=ReplayMemoryLSTM2, value_range=None, add_noise=False, OptimArgs={}):
        self.steps = 0
        self.warmup = args.warmup
        self.states_dim = states_dim
        self.actions_dim = actions_dim
        self.verbose = args.verbose

        self.net_cfg = {
            'hidden_layers': args.hidden_layers,
        }
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.entropy_temp_lr = args.entropy_temp_lr
        self.OptimCls = OptimCls
        self.OptimArgs = OptimArgs
        print("++ Optimizer class: ", self.OptimCls, "with args: ", self.OptimArgs)
        self.ReplayBufferCls = ReplayBufferCls
        self.rmsize = args.rmsize
        self.window_length = args.window_length
        self.alpha = args.alpha
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        ##
        self.model_type = args.rnn_type
        if self.model_type == "lstm":
            from pytorch_rl_collection.model_networks.model_networks import SacActorLSTM as Actor
            from pytorch_rl_collection.model_networks.model_networks import SacCriticLSTM as Critic
        elif self.model_type == "lstm2":
            from pytorch_rl_collection.model_networks.model_networks import SacActorLSTM as Actor
            from pytorch_rl_collection.model_networks.model_networks import SacCriticLSTM2 as Critic
        elif self.model_type == "gru":
            from pytorch_rl_collection.model_networks.model_networks import SacActorGRU as Actor
            from pytorch_rl_collection.model_networks.model_networks import SacCriticGRU as Critic
        else:
            print("++ {} is not a valid RNN SAC type".format(self.model_type))
            raise NotImplementedError
        #####
        self.CriticCls = Critic
        self.ActorCls = Actor

        # Hyper-parameters
        #self.batch_size = args.batch_size
        self.tau = args.tau
        self.discount = args.discount

        #
        dof = np.inf
        self.target_updater1 = TSoft_Update(self.tau, dof=dof, eps=1e-8, name="Target_Updater1")
        self.target_updater2 = TSoft_Update(self.tau, dof=dof, eps=1e-8, name="Target_Updater2")
        ### self.use_cuda = args.cuda
        self.device = torch.device("cuda:{}".format(args.cuda_id) if args.cuda else "cpu")
        self.seed = args.seed
        self.set_seed(self.seed)
        ###
        self._points_random_generator = torch.Generator(device=self.device)
        if self.seed is not None:
            self._points_random_generator = self._points_random_generator.manual_seed(1234 * self.seed)
        ####
        sim_observation_noise = 5e-3
        if value_range is not None and add_noise:
            self.noise_amplitude = torch.FloatTensor(sim_observation_noise * np.ptp(value_range, axis=1)).to(self.device)
        else:
            self.noise_amplitude = None

    def add_noise(self, states):
        if self.noise_amplitude is not None:
            states[:, :, :9] = states[:, :, :9] + self.noise_amplitude * torch.FloatTensor(states.shape[0], 9).to(self.device).uniform_(-0.5, 0.5, generator=self._points_random_generator)
        return states

    def reset_hidden_state(self):
        hidden_dim = self.net_cfg['hidden_layers'][0]
        self.last_action = torch.FloatTensor(np.random.uniform(-1., 1., self.actions_dim)).to(self.device).unsqueeze(0).unsqueeze(0)
        if self.model_type == "gru":
            self.hidden_state = torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(self.device)
        else:
            self.hidden_state = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(self.device), \
                    torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(self.device))

    def initialize(self):
        # An utility function to be able to initialize
        # everything at once
        self.initialize_critics()
        self.initialize_actor()
        self.initialize_targets()
        self.initialize_replay_buffer()
        self.initialize_entropy_temperature()

    def initialize_critics(self):
        self.critic1 = self.CriticCls(self.states_dim, self.actions_dim, **self.net_cfg).to(self.device)
        self.critic2 = self.CriticCls(self.states_dim, self.actions_dim, **self.net_cfg).to(self.device)
        if self.verbose:
            print("++ Both Critics Initialized with following structure:")
            print(self.critic1)
            print("")
        self.critic1_optim  = self.OptimCls(self.critic1.parameters(), lr=self.critic_lr, **self.OptimArgs)
        self.critic2_optim  = self.OptimCls(self.critic2.parameters(), lr=self.critic_lr, **self.OptimArgs)

    def initialize_actor(self):
        net_cfg = self.net_cfg.copy()
        net_cfg['policy_type'] = "Gaussian"
        self.actor = self.ActorCls(self.states_dim, self.actions_dim, **net_cfg).to(self.device)
        if self.verbose:
            print("++ Actor Initialized with following structure:")
            print(self.actor)
            print("")
        self.actor_optim  = self.OptimCls(self.actor.parameters(), lr=self.actor_lr, **self.OptimArgs)

    def initialize_entropy_temperature(self):
        if self.automatic_entropy_tuning:
            self.target_entropy = - self.actions_dim
            #self.log_alpha = torch.tensor([np.log(self.alpha)], requires_grad=True, device=self.device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = self.OptimCls([self.log_alpha], lr=self.entropy_temp_lr, **self.OptimArgs)
            self.alpha = self.log_alpha.exp().item()

    def initialize_targets(self):
        self.critic1_target = self.CriticCls(self.states_dim, self.actions_dim, **self.net_cfg).to(self.device)
        self.critic2_target = self.CriticCls(self.states_dim, self.actions_dim, **self.net_cfg).to(self.device)
        if self.verbose:
            print("++ Both Target Critic networks Initialized.")
        hard_update(self.critic1_target, self.critic1) # Make sure target is with the same weight
        hard_update(self.critic2_target, self.critic2)

    def initialize_replay_buffer(self):
        self.replay_memory = self.ReplayBufferCls(capacity=self.rmsize, seed=self.seed, window_length=self.window_length)
        if self.verbose:
            print("++ Replay Memory Initialized.")

    def select_action(self, state, is_training=True):
        if self.steps <= self.warmup:
            action = np.random.uniform(-1.,1.,self.actions_dim)
            #####
            self.last_action = torch.FloatTensor(action).to(self.device).unsqueeze(0).unsqueeze(0)
            #####
        else:
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0).unsqueeze(0)
            state = self.add_noise(state)
            if is_training:
                action, _, hidden_out = self.actor.get_action(state, self.last_action, self.hidden_state, lengths=None)
            else:
                _, action, hidden_out = self.actor.get_action(state, self.last_action, self.hidden_state, lengths=None)
            #####
            self.last_action = torch.clone(action.detach())
            if self.model_type == "gru":
                self.hidden_state = hidden_out.detach()
            else:
                self.hidden_state = (hidden_out[0].detach(), hidden_out[1].detach())
            #####
            action = to_numpy(
                action
            ).squeeze()#.squeeze(axis=0).squeeze(axis=0)
            #action = np.clip(action, -1., 1.)

        self.steps += int(is_training)#1
        return action

    def store_transition(self, transition):
        state, action, last_action, reward, next_state, done = transition
        self.replay_memory.push(state, action, last_action, reward, next_state, done)

    def store_whole_episode(self, episode):
        ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action, \
                episode_reward, episode_next_state, episode_done = episode
        self.replay_memory.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action, \
                episode_reward, episode_next_state, episode_done)

    def sample_minibatch(self, batch_size):
        #if self.steps > self.warmup:
        if len(self.replay_memory) > batch_size:
            #print(self.steps)
            """if self.model_type == "lstm":
                state_batch, action_batch, last_action_batch, reward_batch, \
                next_state_batch, terminal_batch = self.replay_memory.sample(batch_size)
                hidden_in, hidden_out = None, None
                ######
                state_batch = torch.FloatTensor(state_batch).to(self.device).unsqueeze(1)
                next_state_batch = torch.FloatTensor(next_state_batch).to(self.device).unsqueeze(1)
                action_batch = torch.FloatTensor(action_batch).to(self.device).unsqueeze(1)
                last_action_batch = torch.FloatTensor(last_action_batch).to(self.device).unsqueeze(1)
                reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(-1).unsqueeze(1)
                terminal_batch = torch.FloatTensor(terminal_batch).to(self.device).unsqueeze(-1).unsqueeze(1)
                ######
                lengths = None
            else:"""
            hidden_in, hidden_out, state_batch, action_batch, last_action_batch, reward_batch, \
            next_state_batch, terminal_batch, lengths = self.replay_memory.sample(batch_size)
            ######
            #lengths = []
            #for ep in range(batch_size):
            #    lengths.append(state_batch[ep].shape[0])
            ######
            state_batch = pad_sequence(state_batch, batch_first=True).to(self.device)
            next_state_batch = pad_sequence(next_state_batch, batch_first=True).to(self.device)
            action_batch = pad_sequence(action_batch, batch_first=True).to(self.device)
            last_action_batch = pad_sequence(last_action_batch, batch_first=True).to(self.device)
            reward_batch = pad_sequence(reward_batch, batch_first=True).to(self.device).unsqueeze(-1)
            terminal_batch = pad_sequence(terminal_batch, batch_first=True).to(self.device).unsqueeze(-1)
            ######
            #state_batch = torch.FloatTensor(state_batch).to(self.device)
            #next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            #action_batch = torch.FloatTensor(action_batch).to(self.device)
            #last_action_batch = torch.FloatTensor(last_action_batch).to(self.device)
            #reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(-1)
            #terminal_batch = torch.FloatTensor(terminal_batch).to(self.device).unsqueeze(-1)
            ######
            #print(lengths)
            #print(hidden_in.shape, hidden_out.shape)
            #print(hidden_in[0].shape, hidden_in[1].shape)
            #print(hidden_out[0].shape, hidden_out[1].shape)
            #print(state_batch.shape, action_batch.shape, last_action_batch.shape, reward_batch.shape)
            #print(jkklk)
            ######
            return hidden_in, hidden_out, state_batch, action_batch, last_action_batch, reward_batch, next_state_batch, terminal_batch, lengths
        else:
            return None, None, None, None, None, None, None, None, None

    def get_td_target(self, states, actions, rewards, next_states, masks, hidden_out, lengths):
        if states is not None:
            # Prepare the target q batch
            with torch.no_grad():
                next_states_actions, next_states_log_pi, _, _ = self.actor.evaluate(next_states, actions, hidden_out, lengths)
                ## Q1
                next_q1_values, _ = self.critic1_target(next_states, next_states_actions, actions, hidden_out, lengths)
                ## Q2
                next_q2_values, _ = self.critic2_target(next_states, next_states_actions, actions, hidden_out, lengths)
                ## min(Q1, Q2)
                min_next_qf_target = torch.min(next_q1_values, next_q2_values) - self.alpha * next_states_log_pi

            target_q_batch = rewards + \
                self.discount * masks * min_next_qf_target
            return target_q_batch
        else:
            return None

    def update_critics(self, td_target, states, actions, last_actions, hidden_in, lengths):
        if td_target is not None:
            self.critic1_optim.zero_grad()
            self.critic2_optim.zero_grad()
            q1_batch, _ = self.critic1(states, actions, last_actions, hidden_in, lengths)
            q2_batch, _ = self.critic2(states, actions, last_actions, hidden_in, lengths)
            value_loss = criterion(q1_batch, td_target) + criterion(q2_batch, td_target)
            value_loss.backward()
            self.critic1_optim.step()
            self.critic2_optim.step()

    def update_actor(self, states, last_actions, hidden_in, lengths):
        if states is not None:
            self.actor_optim.zero_grad()

            actions, log_pi, _, _ = self.actor.evaluate(self.add_noise(states), last_actions, hidden_in, lengths)
            policy_loss = (self.alpha * log_pi) - torch.min(
                self.critic1(states, actions, last_actions, hidden_in, lengths)[0],
                self.critic2(states, actions, last_actions, hidden_in, lengths)[0]
            )

            policy_loss = policy_loss.mean()
            policy_loss.backward()
            self.actor_optim.step()

    def update_temperature(self, states, last_actions, hidden_in, lengths):
        if states is not None and self.automatic_entropy_tuning:
            self.alpha_optim.zero_grad()

            with torch.no_grad():
                _, log_pi, _, _ = self.actor.evaluate(states, last_actions, hidden_in, lengths)
            alpha_loss = - (self.log_alpha * (log_pi + self.target_entropy)).mean()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp().item()


    def update_target_networks(self, update):
        #if self.steps > self.warmup:
        if update:
            #soft_update(self.critic1_target, self.critic1, self.tau)
            self.target_updater1.target_update(self.critic1_target, self.critic1, self.tau)
            #soft_update(self.critic2_target, self.critic2, self.tau)
            self.target_updater2.target_update(self.critic2_target, self.critic2, self.tau)

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


    def save_model(self, output_path):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output_path)
        )
        torch.save(
            self.critic1.state_dict(),
            '{}/critic1.pkl'.format(output_path)
        )
        torch.save(
            self.critic2.state_dict(),
            '{}/critic2.pkl'.format(output_path)
        )
