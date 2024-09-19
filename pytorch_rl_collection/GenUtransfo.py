import numpy as np
from scipy import linalg

EPSILON = np.finfo(float).eps
class GenUnscented:
    def __init__(self, sigma=0.1, mean=None, cov_matrix=None, diag_skewness=None, diag_kurtosis=None,
                       dim=1, slack_param=0.9, low_bound=None, up_bound=None):
        self.sigma_sq = sigma**2
        if mean is None:
            mean = np.zeros(dim)
            self.dim = dim
        else:
            self.dim = int(np.prod(mean.shape))
        if cov_matrix is None:
            cov_matrix = np.eye(self.dim) * self.sigma_sq
            if diag_kurtosis is None:
                diag_kurtosis = np.ones(self.dim) * 3. * (sigma**4)
        else:
            if diag_kurtosis is None:
                diag_kurtosis = 3. * (np.diag(cov_matrix)**2)
        if diag_skewness is None:
            diag_skewness = np.zeros(self.dim)
        #####
        #print("cov_matrix: ", cov_matrix)
        cov_matrix_sqrt = linalg.sqrtm(cov_matrix)
        #print("cov_matrix_sqrt: ", cov_matrix_sqrt)
        assert (cov_matrix_sqrt.imag == 0.).all(), "cov_matrix_sqrt: {}".format(cov_matrix_sqrt)
        # Calculate free parameter vector u
        self.u = 0.5 * ( - diag_skewness / np.diag(cov_matrix_sqrt**3) + np.sqrt( np.abs(4. * diag_kurtosis / np.diag(cov_matrix_sqrt**4)
                    - 3. * (diag_skewness / np.diag(cov_matrix_sqrt**3))**2 )) )
        #print("u: ", self.u)
        # Calculate parameter vector
        self.v = self.calculate_v(self.u, cov_matrix_sqrt, diag_skewness)
        #print("v: ", self.v)
        # Calculate the sigma points (I omit the mean, since it is zero) and the corresponding weights
        self.sigma_points, self.weights = self.calculate_sigma_points(mean, cov_matrix_sqrt, self.u, self.v)
        ######
        self.sigma_points, self.weights = self._check_bounds(mean=mean, cov_matrix_sqrt=cov_matrix_sqrt, low_bound=low_bound, up_bound=up_bound, slack_param=slack_param)
        ######

    def calculate_v(self, u, cov_matrix_sqrt, diag_skewness):
        return u + diag_skewness / np.diag(cov_matrix_sqrt**3)

    def _check_bounds(self, mean=None, cov_matrix_sqrt=None, diag_skewness=None, low_bound=None, up_bound=None, slack_param=0.9):
        sigma_points = self.sigma_points
        weights = self.weights
        if low_bound is not None or up_bound is not None:
            if mean is None:
                mean = np.zeros(self.dim)
            if cov_matrix_sqrt is None:
                cov_matrix_sqrt = np.eye(self.dim) * np.sqrt(self.sigma_sq)
            if diag_skewness is None:
                diag_skewness = np.zeros(self.dim)
            ####
            u = self.u
            v = self.v
            ########
            #print("u: ", u, ", v: ", v)
            orig_v = np.copy(v)
        if low_bound is not None:
            for i, x in enumerate(sigma_points):
                #print(x, (x < low_bound).any())
                if (x < low_bound).any():
                    if i == 0:
                        continue
                    #print(i, i-self.dim, cov_matrix_sqrt.shape)
                    if i <= (self.dim):
                        u[(i-1)] = slack_param * np.min( np.abs( (mean - low_bound) / (cov_matrix_sqrt[:, (i-1)] + EPSILON) ) )
                    else:
                        v[(i-1)-self.dim] = slack_param * np.min( np.abs( (low_bound - mean) / (cov_matrix_sqrt[:, (i-1)-self.dim] + EPSILON) ) )
            ###
            if (orig_v == v).all(): # repeat v calculation if v was not redefined
                # RE-Calculate parameter vector
                v = self.calculate_v(u, cov_matrix_sqrt, diag_skewness)
            # RE-Calculate the sigma points and the corresponding weights
            sigma_points, _ = self.calculate_sigma_points(mean, cov_matrix_sqrt, u, v)
        ######
        if up_bound is not None:
            orig_v = np.copy(v)
            for i, x in enumerate(sigma_points):
                if i == 0:
                    continue
                if (x > up_bound).any():
                    if i <= (self.dim):
                        u[(i-1)] = slack_param * min( np.abs( (mean - up_bound) / (cov_matrix_sqrt[:, (i-1)] + EPSILON) ) )
                    else:
                        v[(i-1)-self.dim] = slack_param * min( np.abs( (up_bound - mean) / (cov_matrix_sqrt[:, (i-1)-self.dim] + EPSILON) ) )
            ###
            #print("u: ", u, ", v: ", v, ", orig_v: ", orig_v)
            if (orig_v == v).all(): # repeat v calculation if v was not redefined
                # RE-Calculate parameter vector
                v = self.calculate_v(u, cov_matrix_sqrt, diag_skewness)
            # RE-Calculate the sigma points and the corresponding weights
            sigma_points, weights = self.calculate_sigma_points(mean, cov_matrix_sqrt, u, v)
        ######
        return sigma_points, weights

    def calculate_sigma_points(self, mean, cov_matrix_sqrt, u, v):
        part1, part2 = [], []
        wpp = 1. / ((u+v) * v) # (1 / v) / (u+v)
        #print("wpp: ", wpp)
        wp = wpp * (v / u)
        sigma_points = [(mean, 1. - np.sum(wp + wpp))]
        for j in range(1, self.dim+1):
            Aj = cov_matrix_sqrt[:, j-1]
            #print(mean.shape, Aj.shape)
            sj = mean - u[j-1] * Aj
            part1.append((sj, np.sum(wp[j-1])))
            sLj = mean + v[j-1] * Aj
            part2.append((sLj, np.sum(wpp[j-1])))
        sigma_points = sigma_points + part1 + part2
        #print(sigma_points)
        sigma, weights = map(np.stack, zip(*sigma_points))
        return sigma, weights#_points

    def get_sigma_points_and_weights(self):
        #s, w = map(np.stack, zip(*self.sigma_points))
        return self.sigma_points, self.weights#s, w

#####################
def get_unimoments(a, b, order=0):
    n = order
    return (b**(n+1) - a**(n+1)) / ((n+1)*(b - a))

################################################################################
if __name__ == "__main__":
    dim = 7
    a, b = np.zeros(dim), np.ones(dim)#-np.ones(dim), np.ones(dim)#np.array([0., 0.]), np.array([1., 0.01])
    print(a.shape[0])
    mean = 0.5 * (a + b)
    var = (1. / 12.) * (b - a)**2
    skew = np.zeros(a.shape)
    kurt = np.ones(a.shape) * (9. / 5.)
    ugen = GenUnscented(sigma=np.sqrt(var), mean=mean, cov_matrix=np.diag(var), diag_skewness=skew, diag_kurtosis=kurt,
                             dim=a.shape[0], slack_param=1.0, low_bound=a, up_bound=b)
    s, w = ugen.get_sigma_points_and_weights()
    print(sum(w))
    w = w.reshape(-1, 1)
    print(s, s.shape)
    print(w, w.shape)
    print("------")
    print(mean)
    print((s * w).sum(axis=0))
    print(var)
    print(((s - mean)**2 * w).sum(axis=0))
    print("------")
    """M = np.concatenate((np.ones((1, s.shape[0])), (s**2).T, (s**3).T), axis=0)
    b = np.concatenate((np.ones(1), get_unimoments(a, b, 2), get_unimoments(a, b, 3)), axis=0)
    print(M.shape, b.shape)
    from scipy.optimize import nnls
    w = nnls(M, b)[0].reshape(-1, 1)
    #w = np.linalg.lstsq(M, b, rcond=None)[0].reshape(-1, 1)
    print(w, w.shape, sum(w))
    print(mean)
    print((s * w).sum(axis=0))
    print(var)
    print(((s - mean)**2 * w).sum(axis=0))"""

"""def get_roots(a, b, c, d):
    delta0 = b**2 - 3. * a * c
    delta1 = 2.*(b**3) - 9.*a*b*c + 27*(a**2)*d
    C = (0.5 * (delta1 + np.sqrt(delta1**2 - 4.*(delta0**3) + 0j)))
    z = 0.5 * (-1. + np.sqrt(-3.+0j))
    print(C, z)
    for k in range(3):
        xk = (-1. / (3.*a)) * (b + (z**k)*C + delta0 / ((z**k)*C))
        print(xk)"""
