import numpy as np
import metrics as m
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
