import math
import numpy as np
from numpy.linalg import inv, det, slogdet
from scipy.stats import multivariate_normal

def logpdf(x, mean, Q):
    k = x.shape[0]
    u = x - mean
    (sign, logdet) = slogdet(Q)
    alpha = -k / 2 * np.log(2 * np.pi) + 0.5 * logdet
    return alpha -0.5 * np.dot(u.T, np.dot(Q, u))

def bic(X, Q, mean):
    n = Q.shape[0]
    d = np.diag(Q)
    nb_params =  n + n * (n - 1) / 2 \
                    - (len(Q[Q == 0]) - len(d[d == 0])) / 2

    ll = log_likelihood(X, Q, mean)

    return ll - 0.5 * nb_params * np.log(X.shape[0])

def log_likelihood(X, Q, mean):
    p = 0.

    for i in range(X.shape[0]):
        x = X[i, :]
        l = logpdf(x, mean, Q)

        p += l

    assert p <= 0, "Log likelihood greater than zero"

    return p
