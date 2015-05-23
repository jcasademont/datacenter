import numpy as np
from numpy.linalg import inv, cholesky, solve

import time

def timefunc(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f.__name__, 'took', end - start, 'time')
        return result
    return f_timer

class GaussianModel():

    def __init__(self):
        self.mean_ = None
        self.precision_ = None
        self.covariance_ = None

    def check(self):
        if self.precision_ is None:
            raise ValueError("The precision matrix is not set")
        elif self.mean_ is None:
            raise ValueError("The mean vector is not set")

    def predict(self, X, names):
        self.check()

        Q = self.precision_
        mu = self.mean_

        indices = np.array([np.where(self.variables_names == n)[0][0] for n in names], dtype=int)

        _indices = np.array(list(filter(lambda x: x not in indices,
                                np.arange(Q.shape[0]))), dtype=int)

        new_indices = np.append(indices, _indices)

        _Q = (Q[new_indices, :])[:, new_indices]

        lim_a = np.size(indices)
        Qaa = _Q[:lim_a, :lim_a]
        Qab = _Q[:lim_a, lim_a:]

        # if self.covariance_ is not None:
        #     S = self.covariance_
        #     _S = (S[new_indices, :])[:, new_indices]
        #     iQaa = _S[:lim_a, :lim_a]
        # else:
        iQaa = inv(Qaa)

        mean_a = mu[indices]
        mean_b = mu[_indices]

        preds = mean_a - (np.dot(iQaa,
                np.dot(Qab, (X[:, _indices] - mean_b).T))).T

        return preds

    def sample(self, size=1):
        self.check()

        Q = self.precision_
        mu = self.mean_

        n = Q.shape[0]

        L = cholesky(Q).T

        samples = np.empty((size, n))

        for i in range(size):
            z = np.random.multivariate_normal([0] * n, np.eye(n))
            v = solve(L, z)
            samples[i, :] = v + mu

        return samples

    def variances(self, names):
        self.check()

        Q = self.precision_
        mu = self.mean_

        indices = [np.where(self.variables_names == n)[0][0] for n in names]

        _indices = list(filter(lambda x: x not in indices,
                                np.arange(Q.shape[0])))

        new_indices = np.append(indices, _indices)

        _Q = (Q[new_indices, :])[:, new_indices]

        lim_a = np.size(indices)
        Qaa = _Q[:lim_a, :lim_a]
        Qab = _Q[:lim_a, lim_a:]

        if self.covariance_ is not None:
            S = self.covariance_
            _S = (S[new_indices, :])[:, new_indices]
            iQaa = _S[:lim_a, :lim_a]
        else:
            iQaa = inv(Qaa)

        return np.diag(iQaa)
