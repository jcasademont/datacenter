# import pymc as pm
import numpy as np
import metrics as m
from numpy.linalg import inv
from sklearn.covariance import GraphLassoCV, GraphLasso


def bayesian(X):
    gamma = pm.Gamma('gamma', 10e-6, 10e-6)
    gamma.value = 1.0
    alpha = pm.InverseGamma('alpha', 1., gamma / 2)

    N = 0
    n = 1
    for i in range(X.shape[1]):
        N += n
        n += 1

    Q = inv(np.cov(X))
    print(Q)
    jv = Q[np.triu_indices(X.shape[1])]

    j = np.empty(N, dtype=object)
    for i in range(N):
        j[i] = pm.Normal("j" + str(i), 0, 1. / alpha, value=jv[i])

    @pm.deterministic
    def J(j=j):
        J = np.zeros((X.shape[1], X.shape[1]))
        inds_u = np.triu_indices(X.shape[1])
        inds_l = np.tril_indices(X.shape[1])
        J[inds_u] = j
        J[inds_l] = j
        return inv(np.mat(J))

    xs = np.empty(X.shape[0], dtype=object)
    for i in range(X.shape[0]):
        xs[i] = pm.MvNormal("x" + str(i), np.zeros(X.shape[1]), J, value=X[i, :].T, observed=True)

    model = pm.Model([gamma, alpha, pm.Container(j), J, pm.Container(xs)])
    mcmc = pm.MCMC(model)
    mcmc.sample(10000, 5000)


def empirical(S, labels, graph):
    Q = inv(S)
    for i in range(Q.shape[0]):
        node = labels[i]
        nei = np.array([])

        for k in range(len(graph[node])):
            nei = np.append(nei, np.where(labels == graph[node][k])[0])

        for j in range(Q.shape[1]):
            if j not in nei and j != i:
                Q[i, j] = 0
                Q[j, i] = 0

    return Q


def graphlasso(X, method="cv", assume_centered=False):

    if method is 'cv':
        gl = GraphLassoCV()
        gl.fit(X)
        alpha = gl.alpha_
        Q = gl.precision_

    elif method is 'bic':
        max_score = -np.inf
        alphas = np.arange(0.0, 5.0, 0.1)
        mean = np.mean(X, axis=0)

        for a in alphas:
            gl = GraphLasso(a, max_iter=10000,
                            assume_centered=assume_centered)
            gl.fit(X)
            score = m.bic(X, gl.precision_, mean)

            if score > max_score:
                max_score = score
                alpha = a
                Q = gl.precision_

    else:
        raise NotImplementedError(method +
                " is not a valid method, use 'cv' or 'bic'")

    return Q, alpha
