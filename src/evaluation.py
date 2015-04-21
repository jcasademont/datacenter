import numpy as np
from numpy.linalg import inv

def score(X, indices, Q):
    _indices = list(filter(lambda x: x not in indices, np.arange(Q.shape[0])))

    new_indices = np.append(indices, _indices)
    _Q = (Q[new_indices, :])[:, new_indices]

    lim_a = np.size(indices)
    Qaa = _Q[:lim_a, :lim_a]
    Qab = _Q[:lim_a, lim_a:]

    iQaa = inv(Qaa)

    mean_a = np.mean(X[:, indices], axis=0)
    mean_b = np.mean(X[:, _indices], axis=0)

    errors = np.zeros(lim_a)
    # errors = np.empty((lim_a, X.shape[0]))
    for i in range(X.shape[0]):
        pred = mean_a - np.dot(iQaa, np.dot(Qab, X[i, _indices] - mean_b))
        errors = errors + np.absolute(pred - X[i, indices])
        # errors[:, i] = np.absolute(pred - X[i, indices])

    errors = errors / X.shape[0]

    return errors
