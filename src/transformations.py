import numpy as np
from scipy.stats import norm

class CDF_Estimator:

    def __init__(self, X):
        self.n = len(X)
        self.mean = np.mean(X)
        self.std = np.std(X)

        X = (X - self.mean) / self.std
        self.sort_array = np.sort(X)
        self.values = np.unique(self.sort_array)
        self.cdf = self._cdf(self.sort_array)

    def _cdf(self, sorted_X):
        cdf = np.zeros(len(self.values))

        j = 0
        for v in sorted_X:

            if v <= self.values[j]:
                cdf[j] += 1
            else:
                j += 1
                cdf[j] = cdf[j - 1] + 1

        return cdf / self.n

    def get(self, t, delta=None):
        t = (t - self.mean) / self.std
        idx = np.where(self.values == t)[0][0]
        cdf = self.cdf[idx]

        # if delta:
        #     if cdf < delta:
        #         cdf = delta
        #     elif cdf > 1 - delta:
        #         cdf = 1 - delta
        cdf = (self.n * cdf) / (self.n + 1)

        return self.mean + self.std * norm.ppf(cdf)

def to_normal(X):
    Y = np.array(X, copy=True)
    n = Y.shape[0]

    estimators = np.empty(shape=Y.shape[1], dtype=object)
    delta = 1 / (4 * n ** (0.25) * np.sqrt(np.pi * np.log(n)))
    for i in range(Y.shape[1]):
        estimators[i] = CDF_Estimator(Y[:, i])

    for (i, j), x in np.ndenumerate(Y):
        Y[i, j] = estimators[j].get(x, delta=delta)

    return Y
