import warnings
import numpy as np
from numpy.linalg import inv, cholesky
from sklearn.covariance import GraphLassoCV, GraphLasso

class GMRF():

    def __init__(self, method="cv", alpha=None, verbose=False):
        self.alpha_ = alpha
        self.method_ = method
        self.bic_scores = []
        self.precision_ = None
        self.verbose = verbose

    def check(self):
        if self.precision_ is None:
            raise ValueError("The precision matrix is not set")

    def fit(self, X):
        if self.alpha_:
            gl = GraphLasso(self.alpha_, max_iter=100000)

            gl.fit(X)
            self.precision_ = gl.precision_

        elif self.method_ is 'cv':
            gl = GraphLassoCV(verbose=self.verbose)
            gl.fit(X)
            self.alpha_ = gl.alpha_
            self.precision_ = gl.precision_

        elif self.method_ is 'bic':
            min_score = np.inf
            min_precision = None
            alphas = np.arange(0.0, 5.0, 0.1)

            for a in alphas:
                if self.verbose:
                    print("[GMRF] Alpha = {}".format(a))

                gl = GraphLasso(a, max_iter=100000)

                gl.fit(X)
                self.precision_ = gl.precision_
                score, converged = self.bic(X, gamma=0.5)

                if converged:
                    self.bic_scores.append(score)

                    if score <= min_score:
                        min_score = score
                        self.alpha_ = a
                        min_precision = self.precision_

            self.precision_ = min_precision

        else:
            raise NotImplementedError(method +
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

    def predict(self, X, indices):
        self.check()

        Q = self.precision_

        _indices = list(filter(lambda x: x not in indices,
                                np.arange(Q.shape[0])))

        new_indices = np.append(indices, _indices)

        _Q = (Q[new_indices, :])[:, new_indices]

        lim_a = np.size(indices)
        Qaa = _Q[:lim_a, :lim_a]
        Qab = _Q[:lim_a, lim_a:]

        iQaa = inv(Qaa)

        mean_a = np.mean(X[:, indices], axis=0)
        mean_b = np.mean(X[:, _indices], axis=0)

        pred = mean_a - (np.dot(iQaa,
                np.dot(Qab, (X[:, _indices] - mean_b).T))).reshape(mean_a.shape)

        return pred

    def score(self, X, indices):
        self.check()

        Q = self.precision_

        _indices = list(filter(lambda x: x not in indices,
                                np.arange(Q.shape[0])))

        new_indices = np.append(indices, _indices)

        _Q = (Q[new_indices, :])[:, new_indices]

        lim_a = np.size(indices)
        Qaa = _Q[:lim_a, :lim_a]
        Qab = _Q[:lim_a, lim_a:]

        iQaa = inv(Qaa)

        mean_a = np.mean(X[:, indices], axis=0)
        mean_b = np.mean(X[:, _indices], axis=0)

        # errors = np.zeros(lim_a)
        errors = np.zeros((X.shape[0], lim_a))
        for i in range(X.shape[0]):
            pred = mean_a - np.dot(iQaa,
                            np.dot(Qab, X[i, _indices] - mean_b))
            # errors = errors + np.absolute(pred - X[i, indices])
            errors[i, :] = np.absolute(pred - X[i, indices])

        # errors = errors / X.shape[0]

        return errors

    def sample(self):
        pass
