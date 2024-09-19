import torch as th
import numpy as np
from scipy.special import binom, factorial2

import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import argparse

##############################################
parser = argparse.ArgumentParser(description='Unscented Transform Learner Arguments')
parser.add_argument('--dim', default=2, type=int, help='Dimension of the space (default: 2)')
parser.add_argument('--seed', default=10, type=int, help='Random seed (default: 10)')
parser.add_argument('--normal', action='store_true', help='Run with the gaussian distribution. (default: False)')

##############################################
def convert_string_to_numpy(serie):
    result = serie.apply(lambda x:
                           np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' '))
    return result

### Tensor norm
def tensor_lpqr_norm(tensor, p=2, q=1, r=1):
    res = th.linalg.vector_norm(tensor, ord=r, dim=-1)
    res = th.linalg.vector_norm(res, ord=q, dim=-1)
    res = th.linalg.vector_norm(res, ord=p, dim=-1)
    return res

### Moments
def unifdist_moments(k, a=0., b=1.):
    m = 0.
    for i in range(k+1):
        m += (a**i) * (b**(k-i))
    return m / (k+1)

### Centered Moments
def get_unifdist_centered_moments(n, a=0., b=1.):
    moments = []
    centered_moments = []
    for k in range(1, n+1):
        mk = unifdist_moments(k, a=a, b=b)
        moments.append(mk)
    m1 = moments[0]
    centered_moments.append(m1)
    if n < 2:
        return centered_moments
    for p in range(2, n+1):
        cm = 0.
        for k in range(p+1):
            coef = binom(p, k)
            #print(k, coef, moments[p-k-1], p-k-1)
            cm += coef * ((-m1)**k) * (1. if k == p else moments[p-k-1])
        centered_moments.append(cm)
    return centered_moments

###
def get_normdist_centered_moments(n, mu=0., std=1.):
    centered_moments = []
    for p in range(1, n+1):
        centered_moments.append(
           0. if (p % 2) == 1 else ((std**p) * factorial2(p-1))
        )
    return centered_moments

#### Reset gradients
def zero_grad(variables):
    for v in variables:
        if v.grad is not None:
           v.grad.zero_()

#### Inverse sigmoid
def inverse_sigmoid(y, eps=1e-8):
    return th.log(y + eps) - th.log(1. - y + eps)

#### Inverse hyperbolic tangent
def inverse_tanh(y, eps=1e-8):
    return 0.5 * th.log( (1. + y + eps) / (1. - y + eps) )

#### MD updates
def mirror_descent_update(variables, lr=1e-3, fn=th.sigmoid, inv_fn=inverse_sigmoid):
    for v in variables:
        g = v.grad
        if g is not None:
           #v.data = th.sigmoid(inverse_sigmoid(v.data) - lr * g)
           v.data = fn(inv_fn(v.data) - lr * g)

#### GD updates
def gradient_descent_update(variables, lr=1e-3):
    for v in variables:
        g = v.grad
        if g is not None:
           v.data = v.data - lr * g

##############################################
def train_ut(dim, M=10, seed=10, tol=1e-8, uniform=True): # M the number of moments to be matched
    th.manual_seed(seed)
    np.random.seed(seed)
    ##############################################
    n_points = 10#2*dim + 1
    if uniform:
        points = [
            th.tensor([0.089, 0.898]),
            th.tensor([0.905, 0.705]),
            th.tensor([0.687, 0.081]),
            th.tensor([0.306, 0.918])
        ]
        sig_points = th.stack([th.rand(dim) for _ in range(n_points-1)] + [th.zeros(dim) + 0.5])
        #sig_points = th.stack([e for e in points] + [th.zeros(dim) + 0.5])
        assert (sig_points > 0.).all()
        assert (sig_points < 1.).all()
        ##############################################
        centered_moments = get_unifdist_centered_moments(n=M, a=0., b=1.)
        assert centered_moments[0] == 0.5
        descent_update_fn = mirror_descent_update
        dist_type = "uniform"
        fn = th.sigmoid
        inv_fn = inverse_sigmoid
    else:
        sig_points = th.stack([th.randn(dim) for _ in range(n_points-1)] + [th.zeros(dim)])
        ##############################################
        centered_moments = get_normdist_centered_moments(n=M, mu=0., std=1.)
        assert centered_moments[0] == 0.
        descent_update_fn = gradient_descent_update
        dist_type = "gaussian"
        fn = th.tanh
        inv_fn = inverse_tanh
    ##############################################
    sig_points.requires_grad_()
    sig_weights = th.stack([th.tensor([1. / n_points]) for _ in range(n_points)])
    #sig_weights.requires_grad_()
    print("Before training: ", sig_points.detach().squeeze(), sig_weights.detach().squeeze())
    print("Before training: ", sig_points.detach().squeeze().shape, sig_weights.detach().squeeze().shape)
    print("-------")
    print(dist_type)

    ##############################################
    target_moments = th.stack([th.ones(dim)] + [cmk + th.zeros(dim) for cmk in centered_moments]).unsqueeze(1)

    ##############################################
    best_loss = np.inf
    best_sigp = sig_points.detach().squeeze()
    best_sigw = sig_weights.detach().squeeze()

    ##############################################
    N_STEPS = 500000
    learning_rate = 1e-3 if uniform else 1e-4
    prev_best_loss = 0.
    counter = 0
    for n_step in range(1, N_STEPS+1):
        vandermonde_tensor = th.stack([th.ones(sig_points.shape)] + [sig_points] + [(sig_points - centered_moments[0]).pow(k) for k in range(2, M+1)]
                             + [])
        pred_moments = (sig_weights.T.unsqueeze(0) @ vandermonde_tensor)
        assert th.isnan(vandermonde_tensor).sum() == 0, "{}; {}; {}".format(n_step, [(sig_points - centered_moments[0])], sig_points)
        assert th.isnan(pred_moments).sum() == 0, "{}; {}; {}".format(pred_moments, sig_weights, prev_best_loss)
        loss = tensor_lpqr_norm(pred_moments - target_moments, p=2, q=1, r=1)
        assert th.isnan(loss).sum() == 0, "{}; {}".format(loss, prev_best_loss)

        zero_grad([sig_points, sig_weights])
        loss.backward()
        #descent_update_fn([sig_points, sig_weights], lr=learning_rate)
        #sig_weights.data.div_(sig_weights.detach().sum() + 1e-8)
        descent_update_fn([sig_points], lr=learning_rate)

        loss = loss.item()
        print("Step {}: Loss value = {}  \r".format(n_step, loss), end="")
        if loss < best_loss:
            best_sigp = sig_points.detach().squeeze()
            best_sigw = sig_weights.detach().squeeze()
            best_loss = loss
        if n_step % 1000 == 0:
            print("Step {}: Loss value = {}; Best loss: {}".format(n_step, loss, best_loss))
            if prev_best_loss - best_loss == 0:
                counter += 1
            else:
                counter = 0
            prev_best_loss = best_loss
        if loss <= tol:
            print("Step {}: Loss value = {}; Best loss: {}".format(n_step, loss, best_loss))
            break
        if n_step % 100000 == 0 and not uniform:
            print("\n-------------------")
            print("++ Reduce learning rate: {} ==> {}".format(learning_rate, learning_rate / 10.))
            print("\n-------------------")
            learning_rate /= 10.
        ######
        if counter >= 10:
            break

    print("-------")
    print("After training: ", best_sigp, best_sigw)
    if uniform:
        assert (best_sigp > 0.).all()
        assert (best_sigp < 1.).all()
        assert (best_sigw > 0.).all()

    pd.DataFrame({"Points": list(best_sigp.numpy()), "Weights": list(best_sigw.numpy())}).to_csv("./ut_{}_10sigmapoints_dim{}_seed{}.csv".format(dist_type, dim, seed), index=False)

    if uniform:
        print("-------")
        for k in range(M+1):
            print("==> k = {}".format(k))
            print("+ True val.: ", unifdist_moments(k=k))
            if dim == 1:
                print("+ Approx.: ", (best_sigp.pow(k) * best_sigw).sum(0))
            else:
                print("+ Approx.: ", (best_sigp.pow(k) * best_sigw.unsqueeze(1)).sum(0))

    print("-------")
    for i, cmk in enumerate(centered_moments[1:]):
        print("==> i = {}".format(i+2))
        print("+ True val.: ", cmk)
        if dim == 1:
            print("+ Approx.: ", ((best_sigp - centered_moments[0]).pow(i+2) * best_sigw).sum(0))
        else:
            print("+ Approx.: ", ((best_sigp - centered_moments[0]).pow(i+2) * best_sigw.unsqueeze(1)).sum(0))

    print("-------")
    if dim == 3:
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')
        xs, ys, zs = best_sigp.numpy().T
        ax.scatter(xs, ys, zs)
        plt.savefig("./ut_uniform_sigmapoints_dim{}_seed{}.pdf".format(dim, seed))
    elif dim == 2:
        xs, ys = best_sigp.numpy().T
        plt.scatter(xs, ys)
        plt.savefig("./ut_uniform_sigmapoints_dim{}_seed{}.pdf".format(dim, seed))


#####################################################################################
if __name__ == "__main__":
    args = parser.parse_args()
    dim = args.dim
    train_ut(dim=dim, M=max(10, dim), seed=args.seed*1234, uniform=not args.normal)
    #train_ut(dim=dim, M=4, uniform=not args.normal)
