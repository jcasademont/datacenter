import numpy as np
import gmrf.metrics as m
from numpy.linalg import inv
from sklearn.covariance import GraphLassoCV, GraphLasso

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


def graphlasso(X, method="cv", assume_centered=False, log=False, alpha=None):
    scores = []

    if alpha:
        gl = GraphLasso(alpha, max_iter=100000,
                        assume_centered=assume_centered)
        gl.fit(X)
        Q = gl.precision_
        _alpha = alpha

    elif method is 'cv':
        gl = GraphLassoCV()
        gl.fit(X)
        _alpha = gl.alpha_
        Q = gl.precision_

    elif method is 'bic':
        min_score = np.inf
        alphas = np.arange(0.0, 1.0, 0.1)

        for a in alphas:
            print(" * Alpha = {}".format(a))
            gl = GraphLasso(a, max_iter=100000,
                            assume_centered=assume_centered)
            gl.fit(X)
            score, converged = m.bic(X, gl.precision_, 0.5)

            if converged:
                scores.append(score)

                if score <= min_score:
                    min_score = score
                    _alpha = a
                    Q = gl.precision_

    else:
        raise NotImplementedError(method +
                " is not a valid method, use 'cv' or 'bic'")

    if log:
        return Q, _alpha, scores
    else:
        return Q, _alpha
