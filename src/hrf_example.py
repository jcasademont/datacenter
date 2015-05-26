import numpy as np
from gaussian.hrf import HRF
from gaussian.gbn import GBN
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt

a = np.random.normal(5, 0.5, 3000)
b = a * 3 + 9 + np.random.normal(0, 1, 3000)
c = (-5) * b + 11 + np.random.normal(0, 0.5, 3000)
X = np.array([a, b, c]).T

print((np.mean(a), np.std(a)))
print((np.mean(b), np.std(b)))
print((np.mean(c), np.std(c)))

hrf = HRF(k=4, k_star=4, variables_names=np.array(['a', 'b', 'c']))
gbn = GBN(np.array(['a', 'b', 'c']), edges=[('a', 'b'), ('b', 'c')])
gbn2 = GBN(np.array(['a', 'b', 'c']), edges=[('c', 'b'), ('b', 'a')])

X_train, X_test = train_test_split(X, test_size=0.25)

hrf.fit(X_train)
gbn.fit_params(X_train)
gbn.compute_mean_cov_matrix()
gbn2.fit_params(X_train)
gbn2.compute_mean_cov_matrix()

for bn in hrf.bns:
    print("=======")
    print(bn.edges)
    print(bn.nodes)

print("----------------")
print(gbn.edges)
print(gbn.nodes)
print("----------------")
print(gbn2.edges)
print(gbn2.nodes)

Y = X_test[:, [1, 0]]
Y2 = X_test[:, [1, 0]]
mean_test = np.mean(Y, axis=0)

hrf_preds = hrf.predict(X_test, ['b', 'a'])
preds = gbn.predict(X_test, ['b', 'a'])
preds2 = gbn2.predict(X_test, ['b', 'a'])
scores = np.absolute(Y - preds)
scores2 = np.absolute(Y2 - preds2)
hrf_scores = np.absolute(Y - hrf_preds)

print("*******************")
print((np.mean(scores, axis=0), np.std(scores, axis=0)))
print((np.mean(scores2, axis=0), np.std(scores2, axis=0)))
print((np.mean(hrf_scores, axis=0), np.std(hrf_scores, axis=0)))

plt.figure()
plt.plot(scores)

plt.show()
