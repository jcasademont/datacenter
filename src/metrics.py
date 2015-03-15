import math
import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal

def bic(X, Q, mean):
    n = Q.shape[0]
    nb_params = math.factorial(n) / (2 * math.factorial(n - 2)) + n
    return log_likelihood(X, Q, mean) - 0.5 * nb_params * np.log(X.shape[0])

def log_likelihood(X, Q, mean):
    p = 0.
    S = inv(Q)
    for i in range(X.shape[0]):
        x = X[i, :]
        p += multivariate_normal.logpdf(x, mean, S)

    return p
