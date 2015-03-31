import math
import numpy as np
from numpy.linalg import inv, det, slogdet
from scipy.stats import multivariate_normal

def logpdf(x, mean, Q):
    k = x.shape[0]
    u = x - mean
    (sign, logdet) = slogdet(Q)
    alpha = -k / 2 * np.log(2 * np.pi) + 0.5 * logdet
    #alpha = 0
    return alpha -0.5 * np.dot(u.T, np.dot(Q, u))

def bic(X, Q, mean):
    n = Q.shape[0]
    d = np.diag(Q)
    nb_params =  n + n * (n - 1) / 2 \
                    - (len(Q[Q == 0]) - len(d[d == 0])) / 2

    ll = log_likelihood(X, Q, mean)

    assert ll <= 0, "Log likelihood = %f" % ll

    return ll - 0.5 * nb_params * np.log(X.shape[0])

def log_likelihood(X, Q, mean):
    p = 0.
    j = np.seterr(under='warn')
    for i in range(X.shape[0]):
        x = X[i, :]
        l = logpdf(x, mean, Q)

        assert l <= 0, "Log pdf = {0}, log_det = {1}, log_pi = {2}, dot = {3}, scipy = {4}".format(l, slogdet(Q), np.log(2 * np.pi), np.dot(np.dot((x - mean).T, Q), (x - mean)), multivariate_normal.logpdf(x, mean, inv(Q)))

        p += l

    return p
