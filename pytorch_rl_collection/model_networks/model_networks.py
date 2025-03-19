import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.distributions import Normal
from torch.distributions.utils import _standard_normal

from pytorch_rl_collection.utils import fetch_uniform_unscented_transfo
from pytorch_rl_collection.GenUtransfo import GenUnscented

from copy import deepcopy

###########################################
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

###########################################
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return th.Tensor(size).uniform_(-v, v)

###########################################
def weights_init_(m):
    if isinstance(m, nn.Linear):
        th.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            th.nn.init.constant_(m.bias, 0)

###########################################
def std_weights_init_(m):
    if isinstance(m, nn.Linear):
       th.nn.init.constant_(m.weight, 0)
       if m.bias is not None:
           th.nn.init.constant_(m.bias, LOG_SIG_MIN)

###########################################
def zero_weights_init_(m):
    if isinstance(m, nn.Linear):
        th.nn.init.constant_(m.weight, 0)
        if m.bias is not None:
            th.nn.init.constant_(m.bias, 0)

###########################################
def linear_weights_init(m):
    if isinstance(m, nn.Linear):
        stdv = 1. / np.sqrt(m.weight.size(1))
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)

###########################################
def conv_weights_init(m):
    if isinstance(m, nn.Conv2d):
        th.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            th.nn.init.zeros_(m.bias)

###########################################
class SacActor(nn.Module):
    def __init__(self, nb_states, nb_actions, nb_varpi=0, hidden_layers=[256, 256], policy_type="Gaussian", VarGrad=False, per_layer_varpi=False):
        assert policy_type in ["Deterministic", "Gaussian"]
        super(SacActor, self).__init__()
        self._nb_states = nb_states
        self._nb_varpi = nb_varpi
        self.policy_type = policy_type
        self.VarGrad = VarGrad
        layers = []
        self.per_layer_varpi = per_layer_varpi
        concat_dim = 0
        if self.per_layer_varpi:
            concat_dim = nb_varpi
        ######
        layers.append(nn.Linear(nb_states + nb_varpi, hidden_layers[0]))
        n_hidden_layers = len(hidden_layers)
        for i in range(n_hidden_layers-1):
            layers.append(nn.Linear(hidden_layers[i] + concat_dim, hidden_layers[i+1]))
        layers.append(nn.Linear(hidden_layers[-1] + concat_dim, nb_actions))
        if policy_type == "Gaussian":
            layers.append(nn.Linear(hidden_layers[-1] + concat_dim, nb_actions))
            self._n = 2
        else:
            self._n = 1
        self.network = nn.ModuleList(layers)
        self._network_size = len(self.network)
        ###
        self.apply(weights_init_)

    def forward(self, x):
        out = (x,)
        ######
        N = self._network_size
        if self.policy_type == "Deterministic":
            if self.per_layer_varpi:
                _, varpi = th.split(x, [self._nb_states, self._nb_varpi], dim=-1)
                for i, fc in enumerate(self.network):
                    out = (
                        th.cat((F.relu(fc(*out)), varpi), dim=-1) if i < N - 1 else th.tanh(fc(*out)),
                    )
            else:
                for i, fc in enumerate(self.network):
                    out = (F.relu(fc(*out)) if i < N - 1 else th.tanh(fc(*out)),)
            ###
            return out[0]
        else:
            if self.per_layer_varpi:
                _, varpi = th.split(x, [self._nb_states, self._nb_varpi], dim=-1)
                for i, fc in enumerate(self.network):
                    if i < N - 2:
                        out = (th.cat((F.relu(fc(*out)), varpi), dim=-1),)
                    else:
                        break
            else:
                for i, fc in enumerate(self.network):
                    if i < N - 2:
                        out = (F.relu(fc(*out)),)
                    else:
                        break
            ####
            out = out[0]
            mean = self.network[-2](out)
            log_std = self.network[-1](out)
            log_std = th.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            ###
            normal = Normal(mean, log_std.exp())
            xt = normal.rsample()
            if self.VarGrad:
                normal = Normal(mean.detach(), log_std.exp().detach())
            log_prob = normal.log_prob(xt)
            out = th.tanh(xt)
            log_prob -= th.log(1. - out.square() + 1e-6) # Enforcing action bounds
            log_prob = log_prob.sum(-1, keepdim=True)
            mean = th.tanh(mean)
            ###
            return out, log_prob, mean

    def get_deterministic_action(self, x):
        if self.policy_type == "Deterministic":
            return self(x)
        else:
            out = (x,)
            ######
            N = self._network_size
            for i, fc in enumerate(self.network):
                if i < N - 2:
                    out = (F.relu(fc(*out)),)
                else:
                    break
            ####
            out = out[0]
            mean = self.network[-2](out)
            mean = th.tanh(mean)
            ###
            return None, None, mean

    def get_standard_normal(self, sample_shape):
        return _standard_normal(sample_shape, dtype=self.network[0].weight.dtype, device=self.network[0].weight.device)

    def evaluate(self, x, standard_normal_variable):
        if self.policy_type == "Deterministic":
            return self(x)
        else:
            out = (x,)
            ######
            N = self._network_size
            if self.per_layer_varpi:
                _, varpi = th.split(x, [self._nb_states, self._nb_varpi], dim=-1)
                for i, fc in enumerate(self.network):
                    if i < N - 2:
                        out = (th.cat((F.relu(fc(*out)), varpi), dim=-1),)
                    else:
                        break
            else:
                for i, fc in enumerate(self.network):
                    if i < N - 2:
                        out = (F.relu(fc(*out)),)
                    else:
                        break
            ####
            out = out[0]
            mean = self.network[-2](out)
            log_std = self.network[-1](out)
            log_std = th.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            ###
            normal = Normal(mean, log_std.exp())
            xt = normal.loc + standard_normal_variable * normal.scale
            log_prob = normal.log_prob(xt)
            out = th.tanh(xt)
            log_prob -= th.log(1. - out.square() + 1e-6) # Enforcing action bounds
            log_prob = log_prob.sum(-1, keepdim=True)
            ###
            return out, log_prob

    def get_policy_distribution(self, x):
        if self.policy_type == "Deterministic":
            return self(x)
        else:
            out = (x,)
            ######
            N = self._network_size
            if self.per_layer_varpi:
                _, varpi = th.split(x, [self._nb_states, self._nb_varpi], dim=-1)
                for i, fc in enumerate(self.network):
                    if i < N - 2:
                        out = (th.cat((F.relu(fc(*out)), varpi), dim=-1),)
                    else:
                        break
            else:
                for i, fc in enumerate(self.network):
                    if i < N - 2:
                        out = (F.relu(fc(*out)),)
                    else:
                        break
            ####
            out = out[0]
            mean = self.network[-2](out)
            log_std = self.network[-1](out)
            log_std = th.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            ###
            normal = Normal(mean, log_std.exp())
            return normal.loc, normal.scale

###########################################
## Adapted from https://github.com/maywind23/LSTM-RL/blob/master/common/policy_networks.py#L366
class SacActorLSTM(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden_layers=[256, 256], policy_type="Gaussian", init_w=3e-3):
        assert policy_type in ["Gaussian"]#["Deterministic", "Gaussian"]
        super(SacActorLSTM, self).__init__()
        ######
        self.policy_type = policy_type
        ######
        layers = []
        assert len(hidden_layers) == 2, "The LSTM Actor hidden_layers should \
            have 2 elements. Got {} of length {}".format(hidden_layers, len(hidden_layers))
        #assert hidden_layers[0] == hidden_layers[1], "Both elements of the hidden_layers should \
        #    be equal. Got {} instead".format(hidden_layers)
        layers.append(nn.Linear(nb_states, hidden_layers[0])) # branch 1 FC
        layers.append(nn.Linear(nb_states + nb_actions, hidden_layers[0])) # branch 2 FC
        layers.append(nn.LSTM(hidden_layers[0], hidden_layers[-1])) # branch 2 LSTM
        layers.append(nn.Linear(hidden_layers[0]+hidden_layers[-1], hidden_layers[-1])) # Merged FC
        layers.append(nn.Linear(hidden_layers[-1], hidden_layers[-1]))
        ### Mean and standard deviation
        layers.append(nn.Linear(hidden_layers[-1], nb_actions))
        if policy_type == "Gaussian":
            layers.append(nn.Linear(hidden_layers[-1], nb_actions))
            self._n = 2
        else:
            self._n = 1
        ####
        self.network = nn.ModuleList(layers)
        self._network_size = len(self.network)
        ###
        # Only the output layer seems to have a specific weights initialization
        for i in range(self._n):
            #self.network[-self._n+i].apply(weights_init_)
            self.network[-self._n+i].weight.data.uniform_(-init_w, init_w)
            if self.network[-self._n+i].bias is not None:
                self.network[-self._n+i].bias.data.uniform_(-init_w, init_w)
        #self.network[-2].apply(weights_init_)
        #self.network[-1].apply(weights_init_)

    def forward(self, state, last_action, hidden_in, lengths=None):
        """
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
            for lstm, both needs to be permuted as: (sequence_length, batch_size, -1)
        hidden_in: Tuple (h0, c0) for LSTM initialization, each of shape (1, batch_size, hidden_size)
        lengths: (Optional) 1D tensor or list of actual episode lengths (before padding) for each sample in the
            batch. If provided, it is used with pack_padded_sequence so that the LSTM ignores the padded part.
        """
        state = state.permute(1,0,2) # (T, B, state_dim)
        last_action = last_action.permute(1,0,2) # (T, B, action_dim)
        # branch 1
        fc_branch = F.relu(self.network[0](state)) # (T, B, hidden_layers[0])
        # branch 2
        lstm_branch = th.cat([state, last_action], -1) # (T, B, state_dim+action_dim)
        lstm_branch = F.relu(self.network[1](lstm_branch)) # (T, B, hidden_layers[0])
        if lengths is None:
            lstm_branch, lstm_hidden = self.network[2](lstm_branch, hidden_in)  # no activation after lstm
        else:
            # lengths must be on CPU and be a 1D tensor or list.
            packed_input = pack_padded_sequence(lstm_branch, lengths, enforce_sorted=False)
            lstm_branch, lstm_hidden = self.network[2](packed_input, hidden_in)  # no activation after lstm
            # Unpack the sequence back to (T, B, hidden_layers[1])
            lstm_branch, _ = pad_packed_sequence(lstm_branch, total_length=fc_branch.shape[0])
        # merged
        merged_branch=th.cat([fc_branch, lstm_branch], -1)
        x = F.relu(self.network[3](merged_branch))
        x = F.relu(self.network[4](x))
        x = x.permute(1,0,2)  # permute back

        mean    = self.network[-self._n](x)
        # mean    = F.leaky_relu(self.network[-2](x))
        if self.policy_type == "Gaussian":
            log_std = self.network[-self._n+1](x)
            log_std = th.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        else:
            log_std = None

        return mean, log_std, lstm_hidden

    def evaluate(self, state, last_action, hidden_in, lengths):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in, lengths)
        ####
        if self.policy_type == "Gaussian":
            normal = Normal(mean, log_std.exp())
            xt = normal.rsample()
            log_prob = normal.log_prob(xt)
            action = th.tanh(xt)
            log_prob -= th.log(1. - action.square() + 1e-6) # Enforcing action bounds
            log_prob = log_prob.sum(-1, keepdim=True)
            mean = th.tanh(mean)
            return action, log_prob, mean, hidden_out
        else:
            return None, None, th.tanh(mean), hidden_out

    def get_action(self, state, last_action, hidden_in, lengths):
        action, _, mean, hidden_out = self.evaluate(state, last_action, hidden_in, lengths)
        #return action[0][0], hidden_out
        return action, mean, hidden_out

###########################################
## Adapted from https://github.com/maywind23/LSTM-RL/blob/master/common/policy_networks.py
class SacActorGRU(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden_layers=[256, 256], policy_type="Gaussian", init_w=3e-3):
        assert policy_type in ["Deterministic", "Gaussian"]
        super(SacActorGRU, self).__init__()
        ######
        self.policy_type = policy_type
        ######
        layers = []
        assert len(hidden_layers) == 2, "The GRU Actor hidden_layers should \
            have 2 elements. Got {} of length {}".format(hidden_layers, len(hidden_layers))
        #assert hidden_layers[0] == hidden_layers[1], "Both elements of the hidden_layers should \
        #    be equal. Got {} instead".format(hidden_layers)
        layers.append(nn.Linear(nb_states, hidden_layers[0])) # branch 1 FC
        layers.append(nn.Linear(nb_states + nb_actions, hidden_layers[0])) # branch 2 FC
        layers.append(nn.GRU(hidden_layers[0], hidden_layers[-1])) # branch 2 GRU
        layers.append(nn.Linear(hidden_layers[0]+hidden_layers[-1], hidden_layers[-1])) # Merged FC
        layers.append(nn.Linear(hidden_layers[-1], hidden_layers[-1]))
        ### Mean and standard deviation
        layers.append(nn.Linear(hidden_layers[-1], nb_actions))
        if policy_type == "Gaussian":
            layers.append(nn.Linear(hidden_layers[-1], nb_actions))
            self._n = 2
        else:
            self._n = 1
        ####
        self.network = nn.ModuleList(layers)
        self._network_size = len(self.network)
        ###
        # Only the output layer seems to have a specific weights initialization
        for i in range(self._n):
            #self.network[-self._n+i].apply(weights_init_)
            self.network[-self._n+i].weight.data.uniform_(-init_w, init_w)
            if self.network[-self._n+i].bias is not None:
                self.network[-self._n+i].bias.data.uniform_(-init_w, init_w)
        #self.network[-2].apply(weights_init_)
        #self.network[-1].apply(weights_init_)

    def forward(self, state, last_action, hidden_in, lengths=None):
        """
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
            for lstm, both needs to be permuted as: (sequence_length, batch_size, -1)
        hidden_in: Tuple (h0, c0) for LSTM initialization, each of shape (1, batch_size, hidden_size)
        lengths: (Optional) 1D tensor or list of actual episode lengths (before padding) for each sample in the
            batch. If provided, it is used with pack_padded_sequence so that the LSTM ignores the padded part.
        """
        state = state.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # branch 1
        fc_branch = F.relu(self.network[0](state))
        # branch 2
        gru_branch = th.cat([state, last_action], -1)
        gru_branch = F.relu(self.network[1](gru_branch)) # (T, B, hidden_layers[0])
        if lengths is None:
            gru_branch, gru_hidden = self.network[2](gru_branch, hidden_in)  # no activation after lstm
        else:
            # lengths must be on CPU and be a 1D tensor or list.
            packed_input = pack_padded_sequence(gru_branch, lengths, enforce_sorted=False)
            gru_branch, gru_hidden = self.network[2](packed_input, hidden_in)  # no activation after lstm
            # Unpack the sequence back to (T, B, hidden_layers[1])
            gru_branch, _ = pad_packed_sequence(gru_branch, total_length=fc_branch.shape[0])
        # merged
        merged_branch=th.cat([fc_branch, gru_branch], -1)
        x = F.relu(self.network[3](merged_branch))
        x = F.relu(self.network[4](x))
        x = x.permute(1,0,2)  # permute back

        mean    = self.network[-self._n](x)
        # mean    = F.leaky_relu(self.network[-2](x))
        if self.policy_type == "Gaussian":
            log_std = self.network[-self._n+1](x)
            log_std = th.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        else:
            log_std = None

        return mean, log_std, gru_hidden

    def evaluate(self, state, last_action, hidden_in, lengths):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in, lengths)
        ####
        if self.policy_type == "Gaussian":
            normal = Normal(mean, log_std.exp())
            xt = normal.rsample()
            log_prob = normal.log_prob(xt)
            action = th.tanh(xt)
            log_prob -= th.log(1. - action.square() + 1e-6) # Enforcing action bounds
            log_prob = log_prob.sum(-1, keepdim=True)
            mean = th.tanh(mean)
            return action, log_prob, mean, hidden_out
        else:
            return None, None, th.tanh(mean), hidden_out

    def get_action(self, state, last_action, hidden_in, lengths):
        action, _, mean, hidden_out = self.evaluate(state, last_action, hidden_in, lengths)
        #return action[0][0], hidden_out
        return action, mean, hidden_out

###########################################
class SacCritic(nn.Module):
    def __init__(self, nb_states, nb_actions, nb_varpi=0, output_dim=1, hidden_layers=[256, 256], per_layer_varpi=False):
        super(SacCritic, self).__init__()
        self._nb_states = nb_states
        self._nb_varpi = nb_varpi
        layers = []
        self.per_layer_varpi = per_layer_varpi
        concat_dim = 0
        if self.per_layer_varpi:
            concat_dim = nb_varpi
        ######
        layers.append(nn.Linear(nb_states + nb_varpi + nb_actions, hidden_layers[0]))
        n_hidden_layers = len(hidden_layers)
        for i in range(n_hidden_layers-1):
            layers.append(nn.Linear(hidden_layers[i] + concat_dim, hidden_layers[i+1]))
        layers.append(nn.Linear(hidden_layers[-1] + concat_dim, output_dim))
        self.network = nn.ModuleList(layers)
        self._network_size = len(self.network)
        ###
        self.apply(weights_init_)

    def forward(self, xs):
        x, a = xs
        out = (th.cat(xs, -1),)
        ######
        N = self._network_size
        if self.per_layer_varpi:
            _, v = th.split(x, [self._nb_states, self._nb_varpi], dim=-1)
            for i, fc in enumerate(self.network):
                out = (th.cat((F.relu(fc(*out)), v), dim=-1) if i < N - 1 else fc(*out),)
        else:
            for i, fc in enumerate(self.network):
                out = (F.relu(fc(*out)) if i < N - 1 else fc(*out),)
        return out[0]

###########################################
## Adapted from https://github.com/maywind23/LSTM-RL/blob/master/common/buffers.py
class SacCriticLSTM(nn.Module):
    """
    Critic Q network with LSTM structure.
    The network follows two-branch structure as in paper:
    Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
    """
    def __init__(self, nb_states, nb_actions, output_dim=1, hidden_layers=[256, 256]):
        super(SacCriticLSTM, self).__init__()
        ######
        layers = []
        assert len(hidden_layers) == 2, "The LSTM critic hidden_layers should \
            have 2 elements. Got {} of length {}".format(hidden_layers, len(hidden_layers))
        layers.append(nn.Linear(nb_states + nb_actions, hidden_layers[0])) # branch 1 FC
        layers.append(nn.Linear(nb_states + nb_actions, hidden_layers[0])) # branch 2 FC
        layers.append(nn.LSTM(hidden_layers[0], hidden_layers[1])) # branch 2 LSTM
        layers.append(nn.Linear(2*hidden_layers[0], hidden_layers[1])) # Merged FC
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.network = nn.ModuleList(layers)
        self._network_size = len(self.network)
        ###
        # Only the output layer seems to have a specific weights initialization
        self.network[-1].apply(linear_weights_init)

    def forward(self, state, action, last_action, hidden_in, lengths):
        """
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
            for lstm, both needs to be permuted as: (sequence_length, batch_size, -1)
        hidden_in: Tuple (h0, c0) for LSTM initialization, each of shape (1, batch_size, hidden_size)
        lengths: (Optional) 1D tensor or list of actual episode lengths (before padding) for each sample in the
            batch. If provided, it is used with pack_padded_sequence so that the LSTM ignores the padded part.
        """
        state = state.permute(1,0,2)
        action = action.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # branch 1
        fc_branch = th.cat([state, action], -1)
        fc_branch = F.relu(self.network[0](fc_branch))
        # branch 2
        lstm_branch = th.cat([state, last_action], -1)
        lstm_branch = F.relu(self.network[1](lstm_branch))  # linear layer for 3d input only applied on the last dim # (T, B, hidden_layers[0])
        if lengths is None:
            lstm_branch, lstm_hidden = self.network[2](lstm_branch, hidden_in)  # no activation after lstm
        else:
            # lengths must be on CPU and be a 1D tensor or list.
            packed_input = pack_padded_sequence(lstm_branch, lengths, enforce_sorted=False)
            lstm_branch, lstm_hidden = self.network[2](packed_input, hidden_in)  # no activation after lstm
            # Unpack the sequence back to (T, B, hidden_layers[1])
            lstm_branch, _ = pad_packed_sequence(lstm_branch, total_length=fc_branch.shape[0])
        # merged
        merged_branch=th.cat([fc_branch, lstm_branch], -1)

        x = F.relu(self.network[3](merged_branch))
        x = self.network[-1](x)
        x = x.permute(1,0,2)  # back to same axes as input
        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)

###########################################
## Adapted from https://github.com/maywind23/LSTM-RL/blob/master/common/buffers.py
class SacCriticLSTM2(nn.Module):
    """
    Critic Q network with LSTM structure.
    The network follows single-branch structure as in paper:
    Memory-based control with recurrent neural networks
    """
    def __init__(self, nb_states, nb_actions, output_dim=1, hidden_layers=[256, 256]):
        super(SacCriticLSTM2, self).__init__()
        ######
        layers = []
        assert len(hidden_layers) == 2, "The LSTM critic hidden_layers should \
            have 2 elements. Got {} of length {}".format(hidden_layers, len(hidden_layers))
        layers.append(nn.Linear(nb_states + 2*nb_actions, hidden_layers[0])) # Single branch FC1
        layers.append(nn.LSTM(hidden_layers[0], hidden_layers[1])) # Single branch LSTM
        layers.append(nn.Linear(hidden_layers[0], hidden_layers[1])) # Single branch FC2
        layers.append(nn.Linear(hidden_layers[-1], output_dim)) # Single branch FC3
        self.network = nn.ModuleList(layers)
        self._network_size = len(self.network)
        ###
        # Only the output layer seems to have a specific weights initialization
        self.network[-1].apply(linear_weights_init)

    def forward(self, state, action, last_action, hidden_in, lengths):
        """
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
            for lstm, both needs to be permuted as: (sequence_length, batch_size, -1)
        hidden_in: Tuple (h0, c0) for LSTM initialization, each of shape (1, batch_size, hidden_size)
        lengths: (Optional) 1D tensor or list of actual episode lengths (before padding) for each sample in the
            batch. If provided, it is used with pack_padded_sequence so that the LSTM ignores the padded part.
        """
        seq_len = state.shape[1]
        state = state.permute(1,0,2)
        action = action.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # single branch
        x = th.cat([state, action, last_action], -1)
        x = F.relu(self.network[0](x))# (T, B, hidden_layers[0])
        if lengths is None:
            x, lstm_hidden = self.network[1](x, hidden_in)  # no activation after lstm
        else:
            # lengths must be on CPU and be a 1D tensor or list.
            packed_input = pack_padded_sequence(x, lengths, enforce_sorted=False)
            x, lstm_hidden = self.network[1](packed_input, hidden_in)  # no activation after lstm
            # Unpack the sequence back to (T, B, hidden_layers[1])
            x, _ = pad_packed_sequence(x, total_length=seq_len)

        x = F.relu(self.network[2](x))
        x = self.network[3](x)
        x = x.permute(1,0,2)  # back to same axes as input
        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)

###########################################
class SacCriticGRU(nn.Module):
    def __init__(self, nb_states, nb_actions, output_dim=1, hidden_layers=[256, 256]):
        super(SacCriticGRU, self).__init__()
        ######
        layers = []
        assert len(hidden_layers) == 2, "The GRU critic hidden_layers should \
            have 2 elements. Got {} of length {}".format(hidden_layers, len(hidden_layers))
        layers.append(nn.Linear(nb_states + nb_actions, hidden_layers[0])) # branch 1 FC
        layers.append(nn.Linear(nb_states + nb_actions, hidden_layers[0])) # branch 2 FC
        layers.append(nn.GRU(hidden_layers[0], hidden_layers[1])) # branch 2 GRU
        layers.append(nn.Linear(2*hidden_layers[0], hidden_layers[1])) # Merged FC
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.network = nn.ModuleList(layers)
        self._network_size = len(self.network)
        ###
        # Only the output layer seems to have a specific weights initialization
        self.network[-1].apply(linear_weights_init)

    def forward(self, state, action, last_action, hidden_in, lengths):
        """
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
            for gru, both needs to be permuted as: (sequence_length, batch_size, -1)
        hidden_in: Tuple (h0, c0) for GRU initialization, each of shape (1, batch_size, hidden_size)
        lengths: (Optional) 1D tensor or list of actual episode lengths (before padding) for each sample in the
            batch. If provided, it is used with pack_padded_sequence so that the GRU ignores the padded part.
        """
        state = state.permute(1,0,2)
        action = action.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # branch 1
        fc_branch = th.cat([state, action], -1)
        fc_branch = F.relu(self.network[0](fc_branch))
        # branch 2
        gru_branch = th.cat([state, last_action], -1)
        gru_branch = F.relu(self.network[1](gru_branch))  # linear layer for 3d input only applied on the last dim # (T, B, hidden_layers[0])
        if lengths is None:
            gru_branch, gru_hidden = self.network[2](gru_branch, hidden_in)  # no activation after gru
        else:
            # lengths must be on CPU and be a 1D tensor or list.
            packed_input = pack_padded_sequence(gru_branch, lengths, enforce_sorted=False)
            gru_branch, gru_hidden = self.network[2](packed_input, hidden_in)  # no activation after gru
            # Unpack the sequence back to (T, B, hidden_layers[1])
            gru_branch, _ = pad_packed_sequence(gru_branch, total_length=fc_branch.shape[0])
        # merged
        merged_branch=th.cat([fc_branch, gru_branch], -1)

        x = F.relu(self.network[3](merged_branch))
        x = self.network[-1](x)
        x = x.permute(1,0,2)  # back to same axes as input
        return x, gru_hidden    # gru_hidden is actually tuple: (hidden, cell)

###########################################
class SacSolver(nn.Module):
    def __init__(self, nb_states, nb_kappa=0, hidden_layers=[256, 256],
                solver_type="Deterministic", double=False, sigmoid_fn=True,
                no_output_activa_fct=False):
        assert solver_type in ["Deterministic", "Gaussian"]
        super(SacSolver, self).__init__()
        self.solver_type = solver_type
        self._nb_states = nb_states
        self._nb_kappa = nb_kappa
        layers = []
        ######
        if no_output_activa_fct:
            self.output_fn = lambda x: x
            self.output_fn_der = lambda x: 0.
        elif sigmoid_fn:
            self.output_fn = th.sigmoid
            self.output_fn_der = lambda x: (x + 1e-6).log() + th.log(1. - x + 1e-6)
        else:
            self.output_fn = th.tanh
            self.output_fn_der = lambda x: th.log(1. - x.square() + 1e-6)
        ######
        if not double:
            layers.append(nn.Linear(nb_states + nb_kappa, hidden_layers[0]))
        else:
            layers.append(nn.Linear(nb_states + 2*nb_kappa, hidden_layers[0]))
        n_hidden_layers = len(hidden_layers)
        for i in range(n_hidden_layers-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        layers.append(nn.Linear(hidden_layers[-1], nb_kappa)) # head for mu
        layers.append(nn.Linear(hidden_layers[-1], nb_kappa)) # head for sigma
        self._n = 2
        if solver_type == "Gaussian":
            layers.append(nn.Linear(hidden_layers[-1], nb_kappa)) # std head for mu
            layers.append(nn.Linear(hidden_layers[-1], nb_kappa)) # std head for sigma
        self.network = nn.ModuleList(layers)
        self._network_size = len(self.network)
        ####
        self.apply(weights_init_)

    def forward(self, x):
        out = (x,)
        ######
        N = self._network_size
        if self.solver_type == "Deterministic":
            for i, fc in enumerate(self.network):
                if i < N - 2:
                    out = (F.relu(fc(*out)),)
                else:
                    break
            ####
            out = out[0]
            mean = self.output_fn(self.network[-2](out))
            sig = self.output_fn(self.network[-1](out))
            ####
            return mean, sig
        else:
            for i, fc in enumerate(self.network):
                if i < N - 4:
                    out = (F.relu(fc(*out)),)
                else:
                    break
            ####
            out = out[0]
            mu_mean = self.network[-4](out)
            sig_mean = self.network[-3](out)
            mu_log_std = self.network[-2](out)
            sig_log_std = self.network[-1](out)
            ####
            mu_log_std = th.clamp(mu_log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            sig_log_std = th.clamp(sig_log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            ###
            mu_normal = Normal(mu_mean, mu_log_std.exp())
            sig_normal = Normal(sig_mean, sig_log_std.exp())
            mu = mu_normal.rsample()
            sig = sig_normal.rsample()
            mu_log_prob = mu_normal.log_prob(mu)
            sig_log_prob = sig_normal.log_prob(sig)
            log_prob = th.cat((mu_log_prob, sig_log_prob), dim=-1)
            ####
            mu_out = self.output_fn(mu)
            sig_out = self.output_fn(sig)
            out = th.cat((mu_out, sig_out), dim=-1)
            log_prob -= self.output_fn_der(out)
            log_prob = log_prob.sum(-1, keepdim=True)
            ###
            mu_mean = self.output_fn(mu_mean)
            sig_mean = self.output_fn(sig_mean)
            ###
            return mu_out, sig_out, log_prob, mu_mean, sig_mean

    def sample(self, x, n=1):
        if self.solver_type == "Deterministic":
            return self(x)
        else:
            out = (x,)
            ######
            N = self._network_size
            for i, fc in enumerate(self.network):
                if i < N - 4:
                    out = (F.relu(fc(*out)),)
                else:
                    break
            ####
            out = out[0]
            mu_mean = self.network[-4](out)
            sig_mean = self.network[-3](out)
            mu_log_std = self.network[-2](out)
            sig_log_std = self.network[-1](out)
            ####
            mu_log_std = th.clamp(mu_log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            sig_log_std = th.clamp(sig_log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            ###
            mu_normal = Normal(mu_mean, mu_log_std.exp())
            sig_normal = Normal(sig_mean, sig_log_std.exp())
            mu = mu_normal.rsample([n])
            sig = sig_normal.rsample([n])
            ####
            mu_out = self.output_fn(mu)
            sig_out = self.output_fn(sig)
            ####
            return mu_out, sig_out

###########################################
class EnsembleSacSolver(nn.Module):
    def __init__(self, ensemble_size, nb_states, nb_kappa=0, hidden_layers=[256, 256],
                solver_type="Deterministic", double=False, sigmoid_fn=True,
                no_output_activa_fct=False, device='cpu'):
        super(EnsembleSacSolver, self).__init__()
        self.ensemble_size = ensemble_size
        solver_list = []
        for _ in range(ensemble_size):
            solver_list.append(
                SacSolver(nb_states, nb_kappa=nb_kappa, hidden_layers=hidden_layers,
                        solver_type=solver_type, double=double, sigmoid_fn=sigmoid_fn,
                        no_output_activa_fct=no_output_activa_fct).to(device)
            )
        self.base_model = deepcopy(solver_list[0])
        self.params, self.buffers = th.func.stack_module_state(solver_list)

    def forward(self, x):
        def fmodel(params, buffers, data):
            return th.func.functional_call(self.base_model, (params, buffers), (data,))
        return th.vmap(fmodel, in_dims=(0, 0, None))(self.params, self.buffers, x)

    def get_optimizer(self, OptimCls, lr):
        return OptimCls(self.params.values(), lr=lr)

###########################################
class Encoder(nn.Module):
    def __init__(self, nb_envparams, nb_latent, hidden_layers=[300, 200], init_w=3e-3, verbose=False):
        super(Encoder, self).__init__()
        layers = []
        layers.append(nn.Linear(nb_envparams, hidden_layers[0]))
        n_hidden_layers = len(hidden_layers)
        for i in range(n_hidden_layers-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        layers.append(nn.Linear(hidden_layers[-1], nb_latent))
        self.network = nn.ModuleList(layers)
        self.apply(weights_init_)
        self._network_size = len(self.network)

    def forward(self, p):
        out = p
        N = self._network_size
        for i, fc in enumerate(self.network):
            out = F.relu(fc(out)) if i < N - 1 else fc(out)
        return out

###########################################
class Decoder(nn.Module):
    def __init__(self, nb_envparams, nb_latent, lb=None, ub=None, hidden_layers=[200, 300], init_w=3e-3, verbose=False):
        super(Decoder, self).__init__()
        layers = []
        layers.append(nn.Linear(nb_latent, hidden_layers[0]))
        n_hidden_layers = len(hidden_layers)
        for i in range(n_hidden_layers-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        layers.append(nn.Linear(hidden_layers[-1], nb_envparams))
        self.network = nn.ModuleList(layers)
        self.apply(weights_init_)
        self._network_size = len(self.network)
        ###
        self._ub = ub if ub is not None else np.inf
        self._lb = lb if lb is not None else 0.

    def to(self, device):
        if isinstance(self._ub, th.Tensor):
            self._ub = self._ub.to(device)
        if isinstance(self._lb, th.Tensor):
            self._lb = self._lb.to(device)
        return super(Decoder, self).to(device)

    def forward(self, p):
        out = p
        N = self._network_size
        for i, fc in enumerate(self.network):
            out = F.relu(fc(out)) if i < N - 1 else F.softplus(fc(out))
        #####
        out = th.clamp(out, min=self._lb, max=self._ub)
        return out

###########################################
class OSINet(nn.Module):
    def __init__(self, nb_states, nb_actions, nb_latent, hidden_layers=[200, 200, 200], osi_type="Gaussian", init_w=3e-3, verbose=False):
        assert osi_type in ["Deterministic", "Gaussian"]
        super(OSINet, self).__init__()
        self.osi_type = osi_type
        self._nb_states = nb_states
        self._nb_actions = nb_actions
        self._nb_kappa = nb_latent
        layers = []
        ######
        self.tanh_der = lambda x: th.log(1. - x.square() + 1e-6)
        ######
        layers.append(nn.Linear(2*nb_states + nb_actions + 2*nb_latent, hidden_layers[0]))
        n_hidden_layers = len(hidden_layers)
        for i in range(n_hidden_layers-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        layers.append(nn.Linear(hidden_layers[-1], nb_latent)) # mean head
        if osi_type == "Gaussian":
            layers.append(nn.Linear(hidden_layers[-1], nb_latent)) # std head
            ######
            self.sp, self.sw = GenUnscented(sigma=1., dim=nb_latent).get_sigma_points_and_weights()
            self.sp, self.sw = th.FloatTensor(self.sp), th.FloatTensor(self.sw).reshape(-1, 1)
        ####
        self.network = nn.ModuleList(layers)
        self._network_size = len(self.network)
        ####
        self.apply(weights_init_)

    def to(self, device):
        if self.osi_type == "Gaussian":
            self.sp = self.sp.to(device)
            self.sw = self.sw.to(device)
        return super(OSINet, self).to(device)

    def forward(self, x):
        out = (x,)
        ######
        N = self._network_size
        if self.osi_type == "Deterministic":
            for i, fc in enumerate(self.network):
                if i < N - 1:
                    out = (F.relu(fc(*out)),)
                else:
                    break
            ####
            out = out[0]
            mean = th.tanh(self.network[-1](out))
            ####
            return mean
        else:
            for i, fc in enumerate(self.network):
                if i < N - 2:
                    out = (F.relu(fc(*out)),)
                else:
                    break
            ####
            out = out[0]
            mean = self.network[-2](out)
            log_std = self.network[-1](out)
            ####
            log_std = th.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            ###
            normal = Normal(mean, log_std.exp())
            out = normal.rsample()
            log_prob = normal.log_prob(out)
            ####
            out = th.tanh(out)
            log_prob -= self.tanh_der(out) # Enforcing parameter bounds
            log_prob = log_prob.sum(-1, keepdim=True)
            ###
            mean = th.tanh(mean)
            ###
            return out, log_prob, mean

    def get_varpi(self, x, bounds):
        out = (x,)
        bs = x.shape[0]
        lb, ub = bounds
        ######
        N = self._network_size
        if self.osi_type == "Gaussian":
            for i, fc in enumerate(self.network):
                if i < N - 2:
                    out = (F.relu(fc(*out)),)
                else:
                    break
            ####
            out = out[0]
            mean = self.network[-2](out)
            log_std = self.network[-1](out)
            ####
            log_std = th.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            ###
            sp, sw = self.sp.unsqueeze(1).repeat(1, bs, 1), self.sw.unsqueeze(1).repeat(1, bs, 1)
            ###
            mean = mean.unsqueeze(0).repeat(self.sp.shape[0], 1, 1)
            std = log_std.exp().unsqueeze(0).repeat(self.sp.shape[0], 1, 1)
            ###
            x = lb + 0.5 * (ub - lb) * (1. + th.tanh(mean + std * sp))
            ut_mean = (x * sw).sum(0)
            ut_std = ((x - ut_mean).square() * sw).sum(0).sqrt()
            return th.cat((ut_mean, ut_std), dim=-1)
        else:
            raise NotImplementedError

###########################################
class EnsembleOSINet(nn.Module):
    def __init__(self, ensemble_size, nb_states, nb_actions, nb_latent, hidden_layers=[200, 200, 200],
        device='cpu', init_w=3e-3, verbose=False):
        super(EnsembleOSINet, self).__init__()
        #self._ran_gen = np.random.RandomState(1234 * seed)
        self.ensemble_size = ensemble_size
        osi_type = "Deterministic" # Only Deterministic
        osi_list = []
        for _ in range(ensemble_size):
            osi_list.append(
                OSINet(nb_states, nb_actions, nb_latent, hidden_layers=hidden_layers, init_w=init_w, verbose=verbose,
                    osi_type=osi_type).to(device)
            )
        ####
        self.base_model = deepcopy(osi_list[0])
        self.params, self.buffers = th.func.stack_module_state(osi_list)

    def forward(self, h):
        def fmodel(params, buffers, data):
            return th.func.functional_call(self.base_model, (params, buffers), (data,))
        return th.vmap(fmodel, in_dims=(0, 0, None))(self.params, self.buffers, h)

    def get_optimizer(self, OptimCls, lr):
        return OptimCls(self.params.values(), lr=lr)

    def get_varpi(self, h, bounds):
        pred_ps = self(h)
        ####
        #nb_latent = pred_ps.shape[-1]
        lb, ub = bounds
        #inp_transi, inp_mean, inp_sig = th.split(h, [h.shape[-1]-2*nb_latent, nb_latent, nb_latent])
        ####
        pred_ps = lb + 0.5 * (ub - lb) * (1. + pred_ps)
        mean, std = pred_ps.mean(0), pred_ps.std(0)
        # Ensure within domain coverage
        std = th.min(std, th.min(ub - mean, mean - lb)/np.sqrt(3.))
        return th.cat((mean, std), dim=-1).detach()

###########################################
class DynamicsNet(nn.Module):
    def __init__(self, nb_states, nb_actions, nb_latent, hidden_layers=[200, 200, 200, 200], init_w=3e-3, verbose=False):
        super(DynamicsNet, self).__init__()
        layers = []
        layers.append(nn.Linear(nb_states + nb_actions + nb_latent, hidden_layers[0]))
        n_hidden_layers = len(hidden_layers)
        for i in range(n_hidden_layers-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        layers.append(nn.Linear(hidden_layers[-1], nb_states)) #mean
        layers.append(nn.Linear(hidden_layers[-1], nb_states)) #std
        self.network = nn.ModuleList(layers)
        self.apply(weights_init_)
        self._network_size = len(self.network)

    def forward(self, s, a, lat):
        out = th.cat([s, a, lat], dim=-1)
        N = self._network_size
        for i, fc in enumerate(self.network):
            #out = F.relu(fc(out)) if i < N - 1 else fc(out)
            if i < N - 2:
                out = F.relu(fc(out))
            else:
                break
        ####
        mean = self.network[-2](out)
        log_std = self.network[-1](out)
        log_std = th.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        ####
        normal = Normal(mean, log_std.exp())
        xt = normal.rsample()
        log_prob = normal.log_prob(xt).sum(-1, keepdim=True)
        return xt, normal, mean#out#

    def evaluate(self, s, a, lat, ns):
        out = th.cat([s, a, lat], dim=-1)
        N = self._network_size
        for i, fc in enumerate(self.network):
            #out = F.relu(fc(out)) if i < N - 1 else fc(out)
            if i < N - 2:
                out = F.relu(fc(out))
            else:
                break
        ####
        mean = self.network[-2](out)
        log_std = self.network[-1](out)
        log_std = th.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        ####
        log_prob = Normal(mean, log_std.exp()).log_prob(ns).sum(-1, keepdim=True)
        ####
        return log_prob

###########################################
class SystemID(nn.Module):
    def __init__(self, history_length, nb_latent, hidden_layers=[256, 256], init_w=3e-3, verbose=False, use_sigmoid=True):
        super(SystemID, self).__init__()
        #####
        self.use_sigmoid = use_sigmoid
        if self.use_sigmoid:
            self._output_activ_fn = th.sigmoid
        else:
            self._output_activ_fn = lambda x: x
        #####
        layers = []
        layers.append(nn.Linear(history_length, hidden_layers[0]))
        n_hidden_layers = len(hidden_layers)
        for i in range(n_hidden_layers-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        layers.append(nn.Linear(hidden_layers[-1], nb_latent))
        self.network = nn.ModuleList(layers)
        self.apply(weights_init_)
        self._network_size = len(self.network)

    def forward(self, h):
        out = h
        N = self._network_size
        for i, fc in enumerate(self.network):
            out = F.relu(fc(out)) if i < N - 1 else self._output_activ_fn(fc(out))
        return out

###########################################
class EnsembleSystemID(nn.Module):
    def __init__(self, ensemble_size, history_length, nb_latent, hidden_layers=[256, 256], init_w=3e-3, verbose=False,
        device='cpu', seed=0, use_sigmoid=True):
        super(EnsembleSystemID, self).__init__()
        #self._ran_gen = np.random.RandomState(1234 * seed)
        self.ensemble_size = ensemble_size
        osi_list = []
        for _ in range(ensemble_size):
            osi_list.append(
                SystemID(history_length, nb_latent, hidden_layers=hidden_layers, init_w=init_w, verbose=verbose,
                    use_sigmoid=use_sigmoid).to(device)
            )
        #self.osi_list = nn.ModuleList(self.osi_list)
        self.base_model = deepcopy(osi_list[0])
        #self.base_model = self.base_model.to('meta')
        self.params, self.buffers = th.func.stack_module_state(osi_list)

    def forward(self, h):
        def fmodel(params, buffers, data):
            return th.func.functional_call(self.base_model, (params, buffers), (data,))
        return th.vmap(fmodel, in_dims=(0, 0, None))(self.params, self.buffers, h)

    def get_optimizer(self, OptimCls, lr):
        return OptimCls(self.params.values(), lr=lr)#([{"params" : m.parameters()} for m in self.osi_list])

    def get_varpi(self, h, bounds):
        pred_ps = self(h)
        ####
        nb_latent = pred_ps.shape[-1]
        lb, ub = bounds
        inp_transi, inp_mean, inp_sig = th.split(h, [h.shape[-1]-2*nb_latent, nb_latent, nb_latent], dim=-1)
        ####
        pred_ps = th.stack([inp_mean + np.sqrt(3.) * pred_ps[i].clip(min=-inp_sig, max=inp_sig)
            for i in range(self.ensemble_size)])
        mean, std = pred_ps.mean(0), pred_ps.std(0)
        std = th.min(std, th.min(ub - mean, mean - lb)/np.sqrt(3.))
        new_varpi = th.cat((mean, std), dim=-1).detach()
        return new_varpi
