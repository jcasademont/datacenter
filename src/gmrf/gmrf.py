import warnings
import numpy as np
from scipy.special import erfinv
from numpy.linalg import inv, cholesky, solve
from sklearn.covariance import GraphLassoCV, GraphLasso

class GMRF():

    def __init__(self, method="bic", variables_names=[], alpha=None, verbose=False):
        self.alpha_ = alpha
        self.method_ = method
        self.bic_scores = []
        self.precision_ = None
        self.mean_ = None
        self.verbose = verbose
        self.variables_names = np.array(variables_names)

    def check(self):
        if self.precision_ is None:
            raise ValueError("The precision matrix is not set")
        elif self.mean_ is None:
            raise ValueError("The mean vector is not set")

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)

        if self.alpha_:
            gl = GraphLasso(self.alpha_, max_iter=100000)

            gl.fit(X)
            self.precision_ = gl.precision_

        elif self.method_ == 'cv':
            gl = GraphLassoCV(verbose=self.verbose)
            gl.fit(X)
            self.alpha_ = gl.alpha_
            self.precision_ = gl.precision_

        elif self.method_ == 'bic':
            min_score = np.inf
            min_precision = None
            alphas = np.arange(0.0, 5.0, 0.1)

            for a in alphas:
                if self.verbose:
                    print("[GMRF] Alpha = {}".format(a))

                gl = GraphLasso(a, max_iter=100000)

                gl.fit(X)
                self.precision_ = gl.precision_
                score, converged = self.bic(X, gamma=0.0)

                if converged:
                    self.bic_scores.append(score)

                    if score <= min_score:
                        min_score = score
                        self.alpha_ = a
                        min_precision = self.precision_

            self.precision_ = min_precision

        else:
            raise NotImplementedError(self.method_ +
                    " is not a valid method, use 'cv' or 'bic'")

    def _logpdf(self, x, mean, Q):
        k = x.shape[0]

        u = x - mean

        R = cholesky(Q)

        clogdet = np.sum(np.log(np.diag(R)))

        alpha = -k * np.log(2 * np.pi)
        l = 0.5 * (alpha - np.dot(np.dot(u.T, Q), u)) + clogdet

        return l

    def log_likelihood(self, X):
        self.check()

        Q = self.precision_

        ll = 0.

        mean = np.mean(X, axis=0)

        failures = 0
        for i in range(X.shape[0]):
            x = X[i, :]
            l = self._logpdf(x, mean, Q)

            if l <= 0:
                ll += l
            else:
                failures += 1

        ratio_failure = failures / X.shape[0]
        if ratio_failure != 0.0:
            warnings.warn("Ratio of failure = {}".format(ratio_failure))

        if ll > 0:
            raise ValueError("Log likelihood ( = {} ) \
                              greater than zero".format(ll))

        return ll, ratio_failure < 0.01

    def bic(self, X, gamma=0):
        self.check()

        Q = self.precision_

        n = Q.shape[0]
        d = np.diag(Q)
        nb_params =  n + n * (n - 1) / 2 \
                        - (len(Q[Q == 0]) - len(d[d == 0])) / 2

        ll, converged = self.log_likelihood(X)

        return -2 * ll + nb_params * np.log(X.shape[0]) \
            + 4 * nb_params * gamma * np.log(X.shape[1]), converged

    def predict(self, X, names):
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

        iQaa = inv(Qaa)

        mean_a = mu[indices]
        mean_b = mu[_indices]

        preds = np.zeros((X.shape[0], np.size(indices)))

        for i in range(preds.shape[0]):
            pred = mean_a - (np.dot(iQaa,
                    np.dot(Qab, (X[i, _indices] - mean_b).T))).reshape(mean_a.shape)
            preds[i, :] = pred

        return preds

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

        iQaa = inv(Qaa)

        return np.diag(iQaa)

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
