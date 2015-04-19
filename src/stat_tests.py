from scipy.stats.mstats import normaltest

def test_normality(X):
    a = []
    r = []
    for i in range(X.shape[1]):
        (k2, p) = normaltest(X[:, i])
        if p >= 0.05:
            a.append((i, p))
        else:
            r.append((i, p))
    return a, r
