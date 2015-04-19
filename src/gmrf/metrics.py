import math
import numpy as np
import scipy.sparse as sp
import scipy.io
from numpy.linalg import inv, det, slogdet, cholesky, matrix_rank
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def lognormpdf(x,mu,S):
    """ Calculate gaussian probability density of x, when x ~ N(mu,sigma) """
    nx = len(S)
    norm_coeff = nx*math.log(2*math.pi)+np.linalg.slogdet(S)[1]

    err = x-mu
    print(err)
    # if (sp.issparse(S)):
    #     numerator = spln.spsolve(S, err).T.dot(err)
    # else:
    numerator = np.linalg.solve(S, err).T.dot(err)

    return -0.5*(norm_coeff+numerator)

def logpdf(x, mean, Q):
    k = x.shape[0]

    u = x - mean

    R = cholesky(Q)

    clogdet = np.sum(np.log(np.diag(R)))

    alpha = -k * np.log(2 * np.pi)
    l = 0.5 * (alpha - np.dot(np.dot(u.T, Q), u)) + clogdet

    # if l > 0:
    #     scipy.io.savemat('x.mat', mdict={'x': x})
    #     scipy.io.savemat('mean.mat', mdict={'mean': mean})
    #     scipy.io.savemat('Q.mat', mdict={'Q': Q})
    #     raise ValueError("Log pdf is greater than zero: {}".format(l))


    return l

def bic(X, Q, gamma=0):
    n = Q.shape[0]
    d = np.diag(Q)
    nb_params =  n + n * (n - 1) / 2 \
                    - (len(Q[Q == 0]) - len(d[d == 0])) / 2

    ll, converged = log_likelihood(X, Q)

    return -2 * ll + nb_params * np.log(X.shape[0]) \
            + 4 * nb_params * gamma * np.log(X.shape[1]), converged

def log_likelihood(X, Q):
    ll = 0.

    cov = np.cov(X.T, bias=1)
    mean = np.mean(X, axis=0)

    (sign, logdet) = slogdet(Q)

    if sign <= 0:
        raise ValueError("Determinant is negative")

    # k = Q.shape[0]
    # ll = - np.sum(cov * Q) + logdet
    # ll -= k * np.log(2 * np.pi)
    # ll /= 2

    S = inv(Q)

    failures = 0
    for i in range(X.shape[0]):
        x = X[i, :]
        l = logpdf(x, mean, Q)

        if l <= 0:
            ll += l
        else:
            failures += 1

    ratio_failure = failures / X.shape[0]
    if ratio_failure != 0.0:
        print("Ratio of failure = {}".format(ratio_failure))

    if ll > 0:
        raise ValueError("Log likelihood ( = {} ) greater than zero"
                            .format(ll))

    return ll, ratio_failure < 0.01
