import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal

def bic(X, Q):
    # TODO Fix number of paramters
    k = Q.shape[0] * Q.shape[1] - len(Q[Q == 0])
    return log_likelihood(X, Q) - 0.5 * k * np.log(X.shape[0])

def log_likelihood(X, Q):
    p = 0.
    S = inv(Q)
    for i in range(X.shape[0]):
        x = X[i, :]
        p += multivariate_normal.logpdf(x, [0] * len(x), S)

    return p
