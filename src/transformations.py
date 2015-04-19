import numpy as np
from scipy.stats import norm

class CDF_Estimator:

    def __init__(self, X):
        self.mean = np.mean(X)
        self.std = np.std(X)
        self.sort_array = np.sort(X)
        self.values = np.unique(self.sort_array)
        self.cdf = self._cdf(self.sort_array)

    def _cdf(self, X):
        cdf = np.zeros(len(self.values))

        j = 0
        for v in X:

            if v <= self.values[j]:
                cdf[j] += 1
            else:
                j += 1
                cdf[j] = cdf[j - 1] + 1

        return cdf / len(X)

    def get(self, t, delta=None):
        idx = np.where(self.values == t)[0][0]
        cdf = self.cdf[idx]

        if delta:
            if cdf < delta:
                cdf = delta
            elif cdf > 1 - delta:
                cdf = 1 - delta

        return self.mean + self.std * norm.ppf(cdf)

def to_normal(X):
    Y = np.array(X, copy=True)
    n = Y.shape[0]

    estimators = np.empty(shape=Y.shape[1], dtype=object)
    for i in range(Y.shape[1]):
        estimators[i] = CDF_Estimator(Y[:, i])

    delta = 1 / (4 * n ** (0.25) * np.sqrt(np.pi * np.log(n)))
    for (i, j), x in np.ndenumerate(Y):
        Y[i, j] = estimators[j].get(x, delta=delta)

    return Y
