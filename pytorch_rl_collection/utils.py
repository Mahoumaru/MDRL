import os
import torch
import numpy as np
import pandas as pd

from itertools import count, combinations

from torchvision import transforms
from torchdata.skeleton import SkeletonDataset

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
#import seaborn as sns

import random
from scipy.special import gamma

from functools import reduce as functools_reduce

from scipy.stats import gengamma

import yaml
import errno
import shutil
import fnmatch
from collections import deque

from collections import defaultdict
from typing import DefaultDict

###########################################
def interquartile_mean(x, axis=0):
    #n = len(x)
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    n = x.shape[axis]
    x = np.sort(x, axis=axis)
    start_idx = n//4
    end_idx = 3*n//4
    if n == 1:
        return np.sum(x, axis=axis)
    elif n%4 == 0:
        #y = x[start_idx:end_idx]
        y = np.take(x, np.arange(start_idx, end_idx), axis=axis)
    else:
        f = n/4. - start_idx
        #y = x[1+start_idx:end_idx]
        y = np.take(x, np.arange(1+start_idx, end_idx), axis=axis)
        #yp = np.concatenate((x[start_idx:1+start_idx], x[end_idx:n-start_idx]))
        yp = np.concatenate(
            (np.take(x, np.arange(start_idx, 1+start_idx), axis=axis),
            np.take(x, np.arange(end_idx, n-start_idx), axis=axis)
            ), axis=axis
        )
        y = np.concatenate((y, (1. - f) * yp), axis=axis)
    return (2./n) * np.sum(y, axis=axis)

###########################################
def convert_string_to_numpy(serie):
    result = np.fromstring(
        serie.replace('\n','')
         .replace('[','')
         .replace(']','')
         .replace('  ',' '), sep=' ')
    return result

###########################################
def fetch_uniform_unscented_transfo(dim):
    if dim == 1:
        return UNIFORM_1D_UNSCENTED_TRANSFO
    elif dim == 2:
        return UNIFORM_2D_UNSCENTED_TRANSFO
    else:
        filename = "ut_uniform_sigmapoints_dim{}*.csv".format(dim)
        df = None
        sig_points = []
        for file in os.listdir("./"):
            if fnmatch.fnmatch(file, filename): #"dim{}.".format(dim) in file or
                #print(file)
                df = pd.read_csv("./" + file)
                # if df is None else pd.concat([df, pd.read_csv("./" + file)], ignore_index=True, sort=True)
                sig_points.append(list(df["Points"].apply(convert_string_to_numpy)))
        return {
            "sigma_points": np.stack(sig_points),
            "sigma_weights": np.array(list(df["Weights"])).reshape(-1, 1)
        }

###########################################
def fetch_gaussian_unscented_transfo(dim):
    filename = "./ut_gaussian_sigmapoints_dim{}.csv".format(dim)
    df = pd.read_csv(filename)
    return {
        "sigma_points": np.stack(list(df["Points"].apply(convert_string_to_numpy))),
        "sigma_weights": np.array(list(df["Weights"])).reshape(-1, 1)
    }


###########################################
UNIFORM_2D_UNSCENTED_TRANSFO = {
   "sigma_points": np.stack([np.array([
      [0.68726563, 0.3149943 ],
      [0.31273094, 0.68499416],
      [0.0837525, 0.08276018],
      [0.91625065, 0.9172508 ],
      [0.5, 0.5]
   ]),
   np.array([[0.31483606, 0.08374587], [0.49997485, 0.49999967], [0.08281405, 0.6872253 ], [0.9171803, 0.31277505], [0.68519473, 0.91625404]]),
   np.array([[0.50005543, 0.08374585], [0.9171905, 0.9162541], [0.08282213, 0.6872275 ], [0.31479177, 0.31277242], [0.68514025, 0.49999997]]),
   np.array([[0.08281568, 0.31277248], [0.68517584, 0.6872277 ], [0.91718465, 0.0837471 ], [0.50000113, 0.91625285], [0.31482267, 0.49999997]]),
   np.array([[0.08374607, 0.68517685], [0.9162539, 0.49999928], [0.68722767, 0.08281646], [0.31277245, 0.91718334], [0.49999988, 0.31482396]])
   ]),
   "sigma_weights": np.array([
       [0.2], [0.2], [0.2], [0.2], [0.2]
   ])
}

###########################################
UNIFORM_12D_UNSCENTED_TRANSFO = {
    "sigma_points": np.array([
       [0.4825696, 0.39863625, 0.37467188, 0.5426106, 0.31509018,
              0.3055977, 0.655561, 0.9641412, 0.6789752, 0.23768367,
              0.42177457, 0.42593372],
       [0.6753581, 0.5680598, 0.6760362, 0.9299419, 0.37077278,
              0.38517657, 0.55952156, 0.5156115, 0.54118603, 0.02634549,
              0.03421248, 0.05375181],
       [0.05053902, 0.04580784, 0.0861031, 0.03479515, 0.09838173,
              0.6255063, 0.7726154, 0.8537047, 0.8849401, 0.693928  ,
              0.712565, 0.23866546],
       [0.9573251, 0.351685, 0.91564643, 0.5948111, 0.9674654 ,
              0.82074934, 0.34257117, 0.35880437, 0.64071673, 0.44902414,
              0.47375178, 0.84358054],
       [0.20999861, 0.38459867, 0.05482093, 0.92490864, 0.26254836,
              0.5103237, 0.31969428, 0.810979, 0.5838547, 0.4004288 ,
              0.883583, 0.987958  ],
       [0.23462747, 0.03168043, 0.5478716, 0.6003765, 0.6753166 ,
              0.96681976, 0.6885581, 0.28202674, 0.25069684, 0.89676446,
              0.9809549, 0.2714406 ],
       [0.26306134, 0.78461564, 0.15141296, 0.31310567, 0.02084234,
              0.5488273, 0.29828447, 0.24494644, 0.27642772, 0.23059781,
              0.7362666, 0.5084079 ],
       [0.930825, 0.16592227, 0.40561226, 0.94213635, 0.07724971,
              0.1969262, 0.21029477, 0.9754964, 0.87370586, 0.24005377,
              0.14166622, 0.68177825],
       [0.37680143, 0.7512623, 0.31480423, 0.19658251, 0.5237622 ,
              0.68832624, 0.91274, 0.6493651, 0.29914546, 0.78006023,
              0.6721791, 0.7828326 ],
       [0.3787701, 0.28533688, 0.70500535, 0.6532736, 0.93625367,
              0.09650398, 0.7814272, 0.3129192, 0.5707053, 0.16982971,
              0.18294613, 0.93014115],
       [0.92529124, 0.78330797, 0.09602346, 0.27434152, 0.94468814,
              0.2560451, 0.10038205, 0.65575594, 0.12429945, 0.47728872,
              0.25915447, 0.48288834],
       [0.41902512, 0.27525678, 0.25971878, 0.17915985, 0.74802893,
              0.9511871, 0.11347282, 0.04779045, 0.32974052, 0.9801207 ,
              0.9539112, 0.01806752],
       [0.61745226, 0.589964, 0.9302783, 0.2937055, 0.7491484 ,
              0.14799781, 0.02285253, 0.5875084, 0.3609395, 0.4323507 ,
              0.21368824, 0.09731908],
       [0.74433243, 0.94745666, 0.8478876, 0.48174676, 0.09034557,
              0.03347546, 0.38187066, 0.45899945, 0.1076868, 0.8232129 ,
              0.05997606, 0.4640945 ],
       [0.08686988, 0.73280406, 0.51795465, 0.7803529, 0.63353723,
              0.03608505, 0.96627384, 0.10200313, 0.8259556, 0.07817614,
              0.77651805, 0.63356984],
       [0.6262578, 0.21427834, 0.94729626, 0.93639326, 0.48788485,
              0.23529169, 0.5106627, 0.8411764, 0.99430895, 0.9566184 ,
              0.72932565, 0.5408837 ],
       [0.2309317, 0.08337366, 0.8020198, 0.03542607, 0.51411515,
              0.83649766, 0.06490649, 0.18812118, 0.01577695, 0.78919226,
              0.52479327, 0.90014887],
       [0.46356145, 0.17677549, 0.9014735, 0.18239248, 0.28826863,
              0.7365547, 0.9548927, 0.26065245, 0.06422651, 0.51823795,
              0.3373103, 0.6745703 ],
       [0.7251891, 0.56477034, 0.23024663, 0.37302312, 0.83869123,
              0.49816626, 0.18295088, 0.48445848, 0.50189006, 0.18788832,
              0.75615335, 0.7541359 ],
       [0.91815156, 0.9797177, 0.50290275, 0.09217902, 0.7483245 ,
              0.65996903, 0.7652308, 0.8257573, 0.8901361, 0.62933916,
              0.39735892, 0.83686745],
       [0.02652479, 0.87484956, 0.6261615, 0.5679004, 0.4191364 ,
              0.5558777, 0.5640027, 0.02435847, 0.36800143, 0.76798457,
              0.07040332, 0.32403234],
       [0.74239063, 0.88911676, 0.06478059, 0.6739034, 0.25522146,
              0.9218637, 0.38292423, 0.601526, 0.7041047, 0.05679836,
              0.8721615, 0.18095568],
       [0.14979939, 0.6474414, 0.6280319, 0.78841716, 0.20805351,
              0.28999442, 0.8656647, 0.79671276, 0.16966546, 0.45214313,
              0.45679086, 0.17931823],
       [0.7497112, 0.53330195, 0.3966733, 0.7053901, 0.79121196,
              0.79050314, 0.65080416, 0.13781676, 0.86395794, 0.74403906,
              0.50940585, 0.2107786 ],
       [0.51463497, 0.43997973, 0.51656705, 0.4031271, 0.5356616 ,
              0.40573534, 0.4318407, 0.5193669, 0.5789508, 0.48189452,
              0.34314853, 0.47788048]
    ]),
    "sigma_weights": np.array([
        [0.04]*25
    ])
}

###########################################
UNIFORM_1D_UNSCENTED_TRANSFO = {
   "sigma_points": np.array([
      [0.3719886839389801],
      [0.4315377175807953],
      [0.12908919155597687],
      [0.7305907607078552],
      [0.04963883385062218],
      [0.2758784294128418],
      [0.8118380308151245],
      [0.9650654196739197],
      [0.7343740463256836],
      [0.5]
   ]),
   "sigma_weights": np.array([
       [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1]
   ])
}

###########################################
# Function for Exception Handling
def handler(func, path, exc_info):
    print("We got the following exception")
    print(exc_info)

###########################################
def remove_path(path):
    """path could either be relative or absolute. """
    # check if file or directory exists
    if os.path.isfile(path) or os.path.islink(path):
        # remove file
        os.remove(path)
        print("++ Deleted {} directory successfully".format(path))
    elif os.path.isdir(path):
        # remove directory and all its content
        shutil.rmtree(path, ignore_errors=False, onerror=handler)
        print("++ Deleted {} directory successfully".format(path))
    else:
        #raise ValueError("Path {} is not a file or dir.".format(path))
        print("++ Path {} is not a file or dir.".format(path))

###########################################
def to_numpy(var):
    using_cuda = var.device.type == 'cuda'
    return var.cpu().data.numpy() if using_cuda else var.data.numpy()

###########################################
def estimate_cvar(returns, alpha, numpify=False):
    if not isinstance(returns, torch.Tensor):
        returns = torch.FloatTensor(returns)
        numpify = True
    sorted_returns = torch.sort(returns, dim=0).values
    index = max(int(alpha * len(returns)), 1) # ensure at least 1
    cvar = torch.mean(sorted_returns[:index], dim=0) # if trying to max q
    #cvar = torch.mean(sorted_returns[index:], dim=0) # if trying to min q
    #print(returns.shape, cvar.shape)
    return cvar.squeeze().numpy() if numpify else cvar

###########################################
def max_fct(x, y):
    """
     Necessary for auto_LiRPA since bounds of th.max() and th.abs() operations are
    currently undefined.
    See: https://www.physicsforums.com/threads/proofs-of-max-and-min-formulas-for-2-numbers.318767/
    """
    abs_val = (y - x).square().sqrt() # replaces "(y - x).abs()"
    return 0.5 * (x + y + abs_val)

###########################################
def min_fct(x, y):
    """
     Necessary for auto_LiRPA since bounds of th.min() and th.abs() operations are
    currently undefined.
    See: https://www.physicsforums.com/threads/proofs-of-max-and-min-formulas-for-2-numbers.318767/
    """
    abs_val = (y - x).square().sqrt() # replaces "(y - x).abs()"
    return 0.5 * (x + y - abs_val)

###########################################
class TSoft_Update:
    def __init__(self, tau, dof=1., eps=1e-8, name="Target_Updater"):
        self.dof = dof
        self.tau = tau
        self.eps = eps
        #####
        if tau == 1.:
            self.target_update = self.hard_update
            print("++ {} set to use Hard update".format(name))
        elif dof == np.inf:
            self.target_update = self.soft_update
            print("++ {} set to use Soft update".format(name))
        else:
            self.target_update = self.tsoft_update
            print("++ {} set to use t-Soft update".format(name))
        #####
        self.state: DefaultDict[torch.Tensor, Any] = defaultdict(dict)

    def tsoft_update(self, target, source, tau=None):
        for id, (target_param, param) in enumerate(zip(target.parameters(), source.parameters())):
            state = self.state[id]
            # State initialization
            if len(state) == 0:
                state["W_t"] = (1. - self.tau) / self.tau
                state["sigma_sq"] = torch.zeros(1, device=target_param.data.device) + self.eps
            ####
            dof = self.dof
            sigma_sq = state["sigma_sq"]
            Wt = state['W_t']
            D_ = target_param.data.sub(param.data).square_().mean()#.item()
            wt = (1. + dof) / (D_.div(sigma_sq + self.eps) + dof)
            tau_w = wt / (Wt + wt)
            ####
            target_param.data.copy_(
                target_param.data * (1. - tau_w) + param.data * tau_w
            )
            ####
            tau_sig_w = wt * self.tau * dof / (1. + dof)
            state["sigma_sq"] = (1. - tau_sig_w) * sigma_sq + tau_sig_w * D_
            ####
            state['W_t'] = (1. - self.tau) * (Wt + wt)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def hard_update(self, target, source, tau=None):
        for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(param.data)

###########################################
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

###########################################
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

###########################################
def set_flat_params_to(model, flatten_parameters):
    prev_idx = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flatten_parameters[prev_idx:prev_idx+flat_size].view(param.size())
        )
        prev_idx += flat_size

###########################################
def make_Dirs(d):
    for i, p in enumerate(d.split("/")):
        p = "/".join(d.split("/")[:i]) + "/" + p
        if not os.path.isdir(p):
            try:
                os.mkdir(p)
            except OSError as e:
                print(d, "/".join(d.split("/")[:i]), p)
                if e.errno != errno.EEXIST:
                    raise
                pass

###########################################
def save_data(epochs, data, sdir, name):
    np.savetxt(sdir + name + ".csv", np.array([epochs, data]).T, delimiter=",")
    plt.clf()
    plt.plot(epochs, data)
    plt.xlabel("Epoch")
    plt.ylabel(name)
    plt.tight_layout()
    plt.savefig(sdir + name + ".pdf")

###########################################
def get_path_directory(algo_name, env_name, seed, multi_dim, continuous=False, use_encoder=True, presuffix="", suffix="_10082023"):
    if algo_name in ["ddpg", "sac"]:
        path = "./trained_agents/{}/{}/{}/".format(algo_name, env_name, seed)
    elif "mdsac" in algo_name or "drsac" in algo_name:
        malgo_name = algo_name + presuffix#""
        if "drsac" in algo_name:
            malgo_name = algo_name
        ####
        if continuous and "mdsac" in algo_name:
            malgo_name += "_cont"
        elif "mdsac" in algo_name:
            malgo_name += "_disc"
        ####
        ###### Read config file
        if use_encoder:
            config_filename = env_name.split("-")[0] + ".yaml"
        else:
            config_filename = env_name.split("-")[0] + "_for_test" + ".yaml"
        ######
        if "DClaw" in config_filename:
            config_filename = "DClaw.yaml" if use_encoder else "DClaw_for_test.yaml"
        with open("./pytorch_rl_collection/configs/" + config_filename, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        ######
        if multi_dim:
            config = config["multi_dim"]
        else:
            config = config["2d"]
        ######
        params_idxs = config["params_idxs"]
        ######
        print("++ Randomized params: ", params_idxs.keys())
        env_key_list = list(params_idxs.keys())
        #######
        path = "./trained_agents/{}/{}_{}/{}/".format(malgo_name, env_name, ("_".join(env_key_list) if len(env_key_list) <= 2 else "all") + suffix, seed)
    else:
        print(algo_name)
        raise NotImplementedError
    return path

###########################################
def gaussian_quadrature(N, coefs):
    """
    This function computes the abscissas and weights of the Gaussian quadrature given
    the coefficients of the three term recurrence relation for the distribution of interest.
    """
    if coefs.shape[0] < N:
        raise NotImplementedError
    ####
    J = np.zeros((N, N))
    # Jacobi matrix diagonal
    for n in range(N):
        J[n, n] = coefs[n, 0]
    # Jacobi matrix subdiagonal
    for n in range(1, N):
        J[n, n-1] = np.sqrt(coefs[n, 1])
        J[n-1, n] = J[n, n-1]
    # Eigenvalues and Eigenvectors
    (D, V) = np.linalg.eig(J)
    I = np.argsort(D)
    D = np.sort(D)
    V = V[:, I]
    s = D
    w = coefs[0, 1] * (V[0, :]**2)
    return s, w

###########################################
def ttr_jacobi(N, a=None, b=None):
    if a is None:
        a = 0.
        b = a
    if b is None:
        b = a
    if N <= 0 or a <= -1 or b <= -1:
        raise NotImplementedError
    ####
    alpha0 = (b-a)/(a+b+2)
    beta0 = (2.**(a+b+1)) * gamma(a+1) * gamma(b+1) / gamma(a+b+2)
    if N == 1:
        return np.array([[alpha0, beta0]])
    ####
    nn = np.array([n for n in range(1, N)])
    j = 2.*nn+a+b
    jna = ((b**2 - a**2) * np.ones((1, N-1)) / (j * (j+2.))).reshape(-1)
    ##
    beta1 = 4.*(a+1.)*(b+1.)/(((a+b+2.)**2) * (a+b+3.))
    nn = np.array([n for n in range(2, N)])
    j = j[nn-1]
    beta_jnb = 4.*(nn+a)*(nn+b)*nn*(nn+a+b)/((j**2)*(j+1.)*(j-1.))
    # Coefficients
    alpha_n = np.array([[alpha0] + list(jna)])
    beta_n = np.array([[beta0, beta1] + list(beta_jnb)])
    ##
    return np.concatenate((alpha_n.T, beta_n.T), axis=1)

###########################################
def ut_uniform(a, b, n):
    """
    # ut_uniform Computes n UT sigma points and weights for
    # uniform distribution in [a,b] interval.
    # The UT is a form of gaussian quadrature. For the uniform
    # distribution, the sigma points will be the zeros of
    # the n-th order Legendre polynomial. In this implementation,
    # we use the Jacobi weighting function with parameters a=0, b=0.
    """
    ###
    # Gaussian quadrature nodes and weights.
    s, w = gaussian_quadrature(n, ttr_jacobi(n, 0., 0.))
    ###
    weights = w / 2.
    sigma_points = (a * (1. - s) + b * (1. + s)) / 2.
    ###
    return sigma_points, weights

###########################################
def evaluate(env, agent, algo_name="DDPG", render=False, alpha=0., varpi=None):
    episode_reward = 0.
    done = False
    state, _ = env.reset()
    if 'preference' in agent.select_action.__code__.co_varnames:
        pref = agent.sample_preference()
        distIndex = None
    elif 'tilde_alpha' in agent.select_action.__code__.co_varnames:
        distIndex = agent.sample_distIndex()
        pref = None
    else:
        pref = None
        distIndex = None
    for t in count(0):
        if render:
            env.render()
        ## + Select action according to the actor or
        ## randomly during the warmup steps.
        #print(state.shape)
        if algo_name == "MODDPG":#pref is not None:
            action = agent.select_action(state, pref, is_training=False)
        elif algo_name == "MDRDDPG":#distIndex is not None:
            #action, _, _ = agent.select_action(state, distIndex, is_training=False)
            action, disturbance = agent.select_action(state, distIndex, is_training=False)
            action = (1. - distIndex) * action + distIndex * disturbance
        elif "MDARDDPG" in algo_name:
            #alpha = agent.uncertainty_alpha
            #action = agent.select_action(state, np.array([alpha, alpha]), is_training=False)
            action = agent.select_action(state, np.array([0., 0.]), is_training=False)#agent.select_action(state, np.array([0., alpha]), is_training=False)
            #action = (1. - 2. * alpha) * action
            action = action.squeeze()
        elif "MDVARDDPG" in algo_name or "MDVARSAC" in algo_name:
            action = agent.select_action(state, np.concatenate((np.ones(agent.alpha_dim) * alpha, np.zeros(agent.alpha_dim))), is_training=False)
            action = action.squeeze()
            action = (1. - 2. * alpha) * action
        elif "MDDDPG" in algo_name:
            action = agent.select_action(state, np.zeros(2*agent.alpha_dim), is_training=False)
        elif "MDSAC" in algo_name:
            if "DEGEN" in algo_name:
                action = agent.select_action(state, np.ones(1), is_training=False)
            else:
                if varpi is not None:
                    action = agent.select_action(state, varpi, is_training=False)
                else:
                    action = agent.select_action(state, np.concatenate((np.ones(1), np.zeros(1))), is_training=False)
        else:
            action = agent.select_action(state, is_training=False)
        ## + Execute the action and observe reward r_t and new state
        try:
            new_state, reward, done, _, _ = env.step(action)
        except ValueError:
            print(action.shape, env.action_space.shape)
            raise ValueError
        episode_reward += reward
        ##
        state = new_state
        if done or (t+1) == env._max_episode_steps:
            break
    return episode_reward

###########################################
def md_evaluate(envs, varpis, agent, algo_name="MDSAC", osi_agent=None, degen=False):
    n_envs = len(envs)
    episode_rewards = np.zeros(n_envs)
    dones = np.array([False]*n_envs)
    #########
    env_name = str(envs[0]).split('<')[-1].split('>')[0]
    #########
    #print(varpis)
    #########
    last_osi_varpi = None
    if osi_agent is not None:
        compute_osi = True
    else:
        compute_osi = False
    #########
    states = []
    for env in envs:
        state, _ = env.reset()
        states.append(state)
    states = np.array(states)
    if varpis is not None:
        varpis = np.stack(varpis)
        assert states.shape[0] == varpis.shape[0], "{} vs {}".format(states.shape, varpis.shape)
    #########
    if compute_osi:
        min_p = agent.min_rel_param.cpu().numpy().reshape(-1)
        max_p = agent.max_rel_param.cpu().numpy().reshape(-1)
    #########
    for t in count(0):
        prev_dones = np.copy(dones)
        ## + Select action according to the actor or
        ## randomly during the warmup steps.
        #print(state.shape)
        if "MDSAC" in algo_name or "SIRSA" in algo_name:
            actions = agent.select_action(states, varpis, is_training=False, squeeze=False)
        elif "SDRSAC" in algo_name:
            actions = agent.select_action(states, is_training=False, squeeze=False)
        else:
            actions = agent.select_action(states, is_training=False)
        ## + Execute the action and observe reward r_t and new state
        try:
            new_states, rewards = [], []
            j = 0
            for i, env in enumerate(envs):
                if not dones[i]:
                    new_state, reward, done, _, _ = env.step(actions[j])
                    new_states.append(new_state)
                    rewards.append(reward)
                    dones[i] = done
                    j += 1
            ####
        except ValueError:
            print(action.shape, envs[0].action_space.shape)
            raise ValueError
        episode_rewards[~prev_dones] += np.array(rewards)
        ####
        if dones[-1]:
            compute_osi = False
        if compute_osi:
            with torch.no_grad():
                if degen:
                    varp = np.concatenate((varpis[-1], np.zeros_like(varpis[-1])), axis=-1)
                else:
                    varp = varpis[-1]
                _, uncertainty, _ = osi_agent(
                    torch.FloatTensor(states[-1]).unsqueeze(0).to(agent.device),
                    torch.FloatTensor(actions[-1]).unsqueeze(0).to(agent.device),
                    torch.FloatTensor(new_states[-1]).unsqueeze(0).to(agent.device),
                    torch.FloatTensor(varp).unsqueeze(0).to(agent.device)
                )
                ####
                #####
                mu, sig = uncertainty.mean.squeeze(0), uncertainty.stddev.squeeze(0)
                mu, sig = to_numpy(mu), to_numpy(sig)
                a = np.min((np.max((mu - np.sqrt(3.) * sig, min_p), axis=0), max_p), axis=0)
                b = np.max((np.min((mu + np.sqrt(3.) * sig, max_p), axis=0), min_p), axis=0)
                if degen:
                    varpis[-1] = np.min((np.max((mu, min_p), axis=0), max_p), axis=0)
                else:
                    varpis[-1] = np.concatenate(
                        (0.5 * (a + b), np.sqrt(1./12.) * (b - a)), axis=-1
                    )
            last_osi_varpi = (a, b, mu, sig)#varpis[-1]
        #####
        mutable_dones = dones[~prev_dones]
        ####
        if mutable_dones.shape[0] != np.array(new_states).shape[0]:
            print(dones)
            print(prev_dones)
            print(j, idx_to_delete)
            print("++ mut. dones and new_states shape: ", mutable_dones.shape, np.array(new_states).shape)
            print("++ prev mut. dones: ", prev_mutable_dones)
            print("++ mut. dones: ", mutable_dones)
        states = np.array(new_states)[~mutable_dones]
        if varpis is not None:
            varpis = varpis[~mutable_dones]
        if (dones == True).all() or (t+1) == envs[0]._max_episode_steps:
            break
    return episode_rewards, last_osi_varpi

###########################################
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools_reduce(_getattr, [obj] + attr.split('.'))

###########################################
class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

##################################################
# From https://github.com/robtandy/randomdict/blob/b3cba5ec51d583b81e599ead2680075194c01d5b/randomdict.py
class RandomDict(dict):
    def __init__(self, seed=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._keys = {}  # Maps keys to their index in _random_vector
        self._random_vector = []
        self.last_index = -1
        self.rng = random.Random(seed)

        # Populate _keys and _random_vector along with the inherited dict
        for key in self.keys():
            self._random_vector.append(key)
            self.last_index += 1
            self._keys[key] = self.last_index

    def copy(self):
        """ Return a shallow copy of the RandomDict """
        new_rd = RandomDict(super().copy())
        new_rd._keys = self._keys.copy()
        new_rd._random_vector = self._random_vector[:]
        new_rd.last_index = self.last_index
        return new_rd

    @classmethod
    def fromkeys(cls, keys, value=None):
        """Create a RandomDict from an iterable of keys, all mapped to the same value."""
        rd = cls()
        for key in keys:
            rd[key] = value
        return rd

    def __setitem__(self, key, value):
        """ Insert or update a key-value pair """
        super().__setitem__(key, value)
        i = self._keys.get(key, -1)

        if i == -1:
            # Add new key
            self.last_index += 1
            self._random_vector.append(key)
            self._keys[key] = self.last_index

    def __delitem__(self, key):
        """ Delete item by swapping with the last element in the random vector """
        if key not in self._keys:
            raise KeyError(key)

        # Get the index of the item to delete
        i = self._keys[key]

        # Remove the last item from the random vector
        move_key = self._random_vector.pop()

        # Only swap if we are not deleting the last item
        if i != self.last_index:
            # Move the last item into the location of the deleted item
            self._random_vector[i] = move_key
            self._keys[move_key] = i

        self.last_index -= 1
        del self._keys[key]
        super().__delitem__(key)

    def random_key(self):
        """ Return a random key from this dictionary in O(1) time """
        if len(self) == 0:
            raise KeyError("RandomDict is empty")
        i = self.rng.randint(0, self.last_index)
        return self._random_vector[i]

    def get_key_from_idx(self, idx):
        return self._random_vector[idx]

    def get_idx_from_key(self, key):
        return self._keys[key]

    def random_value(self):
        """ Return a random value from this dictionary in O(1) time """
        return self[self.random_key()]

    def random_item(self):
        """ Return a random key-value pair from this dictionary in O(1) time """
        k = self.random_key()
        return k, self[k]

##################################################
class MultiDimSubDomains:
    def __init__(self, n_regions, ranges, seed=0, verbose=False, use_quasi_random=False):
        if use_quasi_random:
            print("--- Initialize {} {}-dimensional sub-domains with Quasi-Randomness ---".format(n_regions, ranges.shape[0]))
        else:
            print("--- Initialize {} {}-dimensional sub-domains ---".format(n_regions, ranges.shape[0]))
        self.domain_dim = ranges.shape[0]
        self.n_regions = n_regions
        self.ranges = np.sort(ranges, axis=-1)
        self.verbose = verbose
        ####
        self.regions_size = (self.ranges[:,1] - self.ranges[:,0]) / n_regions
        #####
        self._seed = seed
        if seed is None:
            self._points_rnd_gen = np.random.RandomState()
            self._reg_idxs_rnd_gen = np.random.RandomState()
            self._varpi_rnd_generator = np.random.RandomState()
        else:
            self._points_rnd_gen = np.random.RandomState(1234 * seed)
            self._reg_idxs_rnd_gen = np.random.RandomState(1234 * seed)
            self._varpi_rnd_generator = np.random.RandomState(1234 * seed)
        #####
        self._use_quasi_random = use_quasi_random
        if self._use_quasi_random:
            assert self.n_regions == 1, "Number of regions must be 1, instead of {}, if quasi-random is to be used.".format(self.n_regions)
            self._seed = self._points_rnd_gen.uniform()
            self._points_rnd_gen = QuasiRandomGenerator(seed=self._seed, dim=self.domain_dim)
        #####
        self._create_subdomains()

    def _create_subdomains(self):
        n_regions = self.n_regions
        ranges = self.ranges
        regions_size = self.regions_size
        #####
        regions = []
        if n_regions > 1:
            for i in range(n_regions-1):
                regions.append([ranges[:,0] + i * regions_size, ranges[:,0] + (i+1) * regions_size])
            #####
            regions.append([ranges[:,0] + (i+1) * regions_size, ranges[:,1]])
        else:
            regions.append([ranges[:,0], ranges[:,1]])
        #####
        self.regions = np.array(regions).transpose(0, 2, 1)
        self.regions_low, self.regions_high = self.regions.transpose(2, 0, 1)
        assert self.regions.shape[-1] == 2, "The last axis of self.regions is not 2 dimensional, but {} dimensional".format(self.regions.shape)

    def sample_from_domains(self):
        n_regions = self.n_regions
        regions_low, regions_high = self.regions_low, self.regions_high
        dim = regions_low.shape[-1]
        #####
        points_in_regions = []
        for i in range(n_regions):
            if self._use_quasi_random:
                points_in_regions.append(self._points_rnd_gen.uniform(low=regions_low[i], high=regions_high[i]))
            else:
                idxs = self._reg_idxs_rnd_gen.choice(n_regions, dim-1)
                idxs = np.concatenate(([i], idxs))
                points_in_regions.append(self._points_rnd_gen.uniform(low=np.diag(regions_low[idxs]), high=np.diag(regions_high[idxs])))
        #######
        return np.array(points_in_regions)

    def get_point_domain_id(self, point):
        for i in range(self.n_regions):
            if self.regions_low[i, 0] <= point[0] <= self.regions_high[i, 0]:
                return None, i, None


##################################################
class SubDomains:
    def __init__(self, n_regions, ranges, seed=0, verbose=False, use_quasi_random=False):
        print("--- Initialize {} 2-dimensional sub-domains ---".format(n_regions))
        assert ranges.shape[0] == 2, "The parameters must be 2 dimensional, not {} dimensional".format(ranges.shape[0])
        self.domain_dim = ranges.shape[0]
        self.n_regions = n_regions
        self.ranges = np.sort(ranges, axis=-1)
        self.verbose = verbose
        ####
        self.region_area = 1. / n_regions
        self.unit_lp_circle_regions = np.array([[i*self.region_area, (i+1)*self.region_area] for i in range(self.n_regions)])
        self.unit_lp_circle_regions = np.sqrt(self.unit_lp_circle_regions).reshape(n_regions, 2, 1).repeat(2, axis=2)
        ####
        assert len(self.unit_lp_circle_regions) == n_regions
        #####
        self._seed = seed
        if seed is None:
            self._points_rnd_gen = np.random.RandomState()
            self._varpi_rnd_generator = np.random.RandomState()
        else:
            self._points_rnd_gen = np.random.RandomState(1234 * seed)
            self._varpi_rnd_generator = np.random.RandomState(1234 * seed)
        #####
        self._rv = gengamma(a=0.5, c=2., loc=np.zeros(2), scale=np.ones(2))
        #####
        self._create_subdomains()

    def _create_subdomains(self):
        n_regions = self.n_regions
        regions = self.unit_lp_circle_regions
        ranges = self.ranges
        #####
        self.max_norm = np.linalg.norm(
           regions[0][0] - regions[-1][1]
        )
        #####
        self.regions_norms = []
        self.regions_inf_norms = []
        for i in range(n_regions):
            lowb, upb = regions[i]
            self.regions_norms.append([np.linalg.norm(lowb - regions[0][0]), np.linalg.norm(upb - regions[0][0])])
            self.regions_inf_norms.append([np.linalg.norm(lowb - regions[0][0], ord=np.inf), np.linalg.norm(upb - regions[0][0], ord=np.inf)])
        ########
        self.regions_norms = np.array(self.regions_norms)
        self.regions_inf_norms = np.array(self.regions_inf_norms)

    def sample_from_domains(self, planar=False, idx=None):
        ranges = self.ranges
        points = np.sqrt(self._points_rnd_gen.rand())
        ####
        points_in_regions = []
        l0 = self.unit_lp_circle_regions[0][0]
        l1 = self.unit_lp_circle_regions[-1][1]
        ###########
        if idx is None:
            for i in range(self.n_regions):
                region_lb, region_ub = self.regions_norms[i]
                ####
                v = self._rv.rvs(random_state=self._points_rnd_gen)
                v = v / np.linalg.norm(v)
                assert (v >= 0.).all(), "{}: {}".format(i, v)
                ####
                point_norm = (region_lb + points * (region_ub - region_lb))
                ####
                k = np.min(l1 / v)
                point_norm = point_norm * (k / self.max_norm) # Sample from edge subdivision
                ####
                point = l0 + point_norm * v
                point = ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * point
                points_in_regions.append(point)
            #######
            return np.array(points_in_regions)
        else:
            region_lb, region_ub = self.regions_norms[idx]
            ####
            v = self._rv.rvs(random_state=self._points_rnd_gen)
            v = v / np.linalg.norm(v)
            assert (v >= 0.).all(), "{}: {}".format(idx, v)
            ####
            point_norm = (region_lb + points * (region_ub - region_lb))
            ####
            k = np.min(l1 / v)
            point_norm = point_norm * (k / self.max_norm) # Sample from edge subdivision
            ####
            point = l0 + point_norm * v
            point = ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * point
            points_in_regions.append(point)
            #######
            return np.array(points_in_regions).reshape(-1)

    def sample_domains_varpis(self, degen=False):
        if degen:
            orig_varpi_mean = self._varpi_rnd_generator.uniform(0., 1. + 1e-6, (1, 1))
            orig_varpi_sigma = np.zeros_like(orig_varpi_mean)
        else:
            orig_varpi_params = self._varpi_rnd_generator.uniform(0., 1. + 1e-6, (1, 2))
            orig_varpi_mean, orig_varpi_sigma = np.split(orig_varpi_params, 2, axis=-1)
        ####
        max_norm = self.max_norm
        varpis = []
        for i in range(self.n_regions):
            region_lb, region_ub = self.regions_norms[i]
            ####
            varpi_mean = region_lb + orig_varpi_mean * (region_ub - region_lb)
            assert (varpi_mean < max_norm).all(), "{}: {} vs {}; {}; {}".format(i, varpi_mean, max_norm, region_lb, orig_varpi_mean)
            varpi_sigma = orig_varpi_sigma * np.min(((max_norm - varpi_mean), varpi_mean), axis=0) / np.sqrt(3.)
            ####
            varpis.append(np.array((varpi_mean, varpi_sigma)).reshape(-1))
        ####
        varpis = np.array(varpis)
        return varpis

    def sample_using_varpis(self, varpis):
        ranges = self.ranges
        points = np.sqrt(self._points_rnd_gen.rand())
        ####
        points_in_regions = []
        l0 = self.unit_lp_circle_regions[0][0]
        l1 = self.unit_lp_circle_regions[-1][1]
        for i in range(self.n_regions):
            ####
            varpi_mean, varpi_sigma = np.split(varpis[i], 2, axis=-1)
            ub = varpi_mean + np.sqrt(3.) * varpi_sigma
            lb = varpi_mean - np.sqrt(3.) * varpi_sigma
            ####
            v = self._rv.rvs(random_state=self._points_rnd_gen)
            v = v / np.linalg.norm(v)
            assert (v >= 0.).all(), "{}: {}".format(i, v)
            ####
            point_norm = (lb + points * (ub - lb))
            ####
            k = np.min(l1 / v)
            point_norm = point_norm * (k / self.max_norm) # Sample from edge subdivision
            ####
            point = l0 + point_norm * v
            point = ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * point
            points_in_regions.append(point)
        #######
        return np.array(points_in_regions)

    def get_point_domain_id(self, point):
        range_lb, range_ub = self.ranges.T
        point = np.clip(point, a_min=range_lb, a_max=range_ub)
        ####
        unit_point = (point - range_lb) / (range_ub - range_lb)
        norm_point = np.linalg.norm(unit_point, ord=np.inf)
        #norm_point = np.round(norm_point, 3)
        ####
        idx = None
        one_hot_repr = []
        for i in range(self.n_regions):
            norm_a, norm_b = self.regions_inf_norms[i]
            if i == self.n_regions-1:
                cond = norm_a <= norm_point <= norm_b
            else:
                cond = norm_a <= norm_point < norm_b
            #######
            if cond:
                idx = i
                one_hot_repr.append(1)
            else:
                one_hot_repr.append(0)
        ####
        return np.array(one_hot_repr), idx, norm_point

###########################################
def unif_wasserstein_dist(pdist, qdist):
    d = (pdist.shape[-1]//2)
    mup, sigmap = torch.split(pdist, d, dim=-1)
    muq, sigmaq = torch.split(qdist, d, dim=-1)
    ####
    lower_bound_p, upper_bound_p = mup - np.sqrt(3.) * sigmap, mup + np.sqrt(3.) * sigmap + 1e-6
    #print("pdist: ", lower_bound_p, upper_bound_p)
    lower_bound_q, upper_bound_q = muq - np.sqrt(3.) * sigmaq, muq + np.sqrt(3.) * sigmaq + 1e-6
    #print("qdist: ", lower_bound_q, upper_bound_q)
    ####
    lower_bounds = torch.stack([lower_bound_p, lower_bound_q])
    upper_bounds = torch.stack([upper_bound_p, upper_bound_q])
    #print(lower_bounds.shape)
    a_idx = torch.argmin(lower_bounds, keepdim=True, dim=0)
    #print(a_idx, a_idx.shape)
    alpha_idx = torch.argmax(lower_bounds, keepdim=True, dim=0)
    #print(alpha_idx, alpha_idx.shape)
    a, b = torch.gather(lower_bounds, 0, a_idx), torch.gather(upper_bounds, 0, a_idx)
    #a, b = lower_bounds[a_idx], upper_bounds[a_idx]
    alpha, beta = torch.gather(lower_bounds, 0, alpha_idx), torch.gather(upper_bounds, 0, alpha_idx)
    #print("a: ", a)
    #print("b: ", b)
    #print("alpha: ", alpha)
    #print("beta: ", beta)
    ####
    case1 = ( beta <= b )
    case2 = ( (~case1) * (alpha <= b) * (b <= beta) )
    case3 = ( alpha >= b )
    #print("Case 1: ", case1)
    #print("Case 2: ", case2)
    #print("Case 3: ", case3)
    ####
    case1 = case1.float()
    case2 = case2.float()
    case3 = case3.float()
    ####
    T1 = (1. / (b - a)) * ( alpha.square() + 0.5 * a.square() - 2. * alpha.mul(a) - 0.5 * b.square() + a.mul(b) ) # Case 1 et 2
    T2 = b.sub(beta)
    T2 = torch.sign(T2) * T2 # Case 1 et 2
    T3 = (1. / (beta - alpha)) * ( 0.5 * beta.square() + 0.5 * alpha.square() - alpha.mul(beta) ) # Case 1
    T4 = (1. / (beta - alpha)) * ( b.square() + 0.5 * alpha.square() - 2. * alpha.mul(b) - 0.5 * beta.square() + alpha.mul(beta) ) # Case 2
    T5 = 0.5 * ( (alpha + beta) - (a + b) ) # Case 3
    ####
    out = ( (case1 + case2) * (T1 + T2) + case1 * T3 + case2 * T4 + case3 * T5 ).squeeze(dim=0).mean(dim=-1)
    ####
    return out

################################################################################
class QuasiRandomGenerator:
    def __init__(self, seed=0.5, dim=2):
        self.seed = seed
        self.dim = dim
        self.counter = 0
        ###
        g = self.get_phi(dim)
        self.alpha = np.zeros(dim)
        for j in range(dim):
            self.alpha[j] = pow(1./g, j+1)

    def reset(self):
        self.counter = 0

    def set_seed(self, seed):
        self.seed = seed
        self.reset()

    def get_phi(self, d):
        x = 2.0000
        for i in range(10):
          x = pow(1+x, 1/(d+1))
        return x

    def sample(self, n):
        d = self.dim
        z = np.zeros((n, d))
        for i in range(self.counter, self.counter+n):
            z[i-self.counter] = (self.seed + self.alpha*(i+1)) % 1
        ###
        self.counter += n
        ###
        return z.reshape(-1) if n == 1 else z

    def uniform(self, low=0., high=1., size=1):
        return low + (high - low) * self.sample(size)
