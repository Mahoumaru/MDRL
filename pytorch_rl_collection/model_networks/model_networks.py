import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

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
class SacActor(nn.Module):
    def __init__(self, nb_states, nb_actions, nb_varpi=0, hidden_layers=[256, 256], policy_type="Gaussian", per_layer_varpi=False):
        assert policy_type in ["Deterministic", "Gaussian"]
        super(SacActor, self).__init__()
        self._nb_states = nb_states
        self._nb_varpi = nb_varpi
        self.policy_type = policy_type
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
            log_prob = normal.log_prob(xt)
            out = th.tanh(xt)
            log_prob -= th.log(1. - out.square() + 1e-6) # Enforcing action bounds
            log_prob = log_prob.sum(-1, keepdim=True)
            mean = th.tanh(mean)
            ###
            return out, log_prob, mean

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
