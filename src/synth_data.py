import numpy as np
import plot as pl
import gmrf.models as mo
import gmrf.graphs as gr
import matplotlib.pyplot as plt
from numpy.linalg import inv
import transformations as tr
import stat_tests as st
from scipy.stats import norm, beta

np.random.seed(48565)


def gen_gauss():
    X = np.array([])
    for i in range(3000):
        a = np.random.normal(5.2, 2.6)
        b = a + np.random.normal(3.0, 0.7)
        c = a + np.random.normal(-5, 2.)
        d = c + np.random.normal(3., 1.)

        # print([a, b, c, d])
        X = np.append(X, [a, b, c, d])

    return X.reshape(3000, 4)

def gen_beta():
    X = np.array([])
    for i in range(3000):
        a = np.random.normal(5.2, 2.6)
        b = a + np.random.beta(3.0, 0.7)
        c = a + np.random.normal(-5, 2.)
        d = c + np.random.normal(3., 1.)

        # print([a, b, c, d])
        X = np.append(X, [a, b, c, d])

    return X.reshape(3000, 4)

def generate_data(mu, sigma):
    return np.random.normal(mu, sigma, 3000)


def log_likelihood(X):
    r = []
    for i in range(X.shape[1]):
        y = X[:, i]
        ll = norm.logpdf(y, loc=np.mean(y), scale=np.std(y)).sum()
        r.append(ll)

    return r


def main():
    # A = generate_data(1.2, 0.5)
    # B = generate_data(2.4, 0.5)
    # C = generate_data(0.4, 1.5)
    # D = generate_data(0.4, 1.5)

    # X = np.array([A, B, C, D]).T
    X = gen_gauss()
    a, r = st.test_normality(X)
    print("Accepted: {}, Refused: {}".format(a, len(r)))

    Y = gen_beta()

    Yt = tr.to_normal(Y)
    print(Yt.shape)
    a, r = st.test_normality(Y)
    print("Accepted: {}, Refused: {}".format(a, len(r)))
    a, r = st.test_normality(Yt)
    print("Accepted: {}, Refused: {}".format(a, len(r)))

    print(log_likelihood(Y))
    print(log_likelihood(Yt))
    plt.subplot(2, 2, 1)
    plt.hist(X[:,0], alpha=0.5)
    plt.hist(Y[:,0], alpha=0.5)
    plt.hist(Yt[:,0], alpha=0.5)
    plt.subplot(2, 2, 2)
    plt.hist(X[:,1], alpha=0.5)
    plt.hist(Y[:,1], alpha=0.5)
    plt.hist(Yt[:,1], alpha=0.5)
    plt.subplot(2, 2, 3)
    plt.hist(X[:,2], alpha=0.5)
    plt.hist(Y[:,2], alpha=0.5)
    plt.hist(Yt[:,2], alpha=0.5)
    plt.subplot(2, 2, 4)
    plt.hist(X[:,3], alpha=0.5)
    plt.hist(Y[:,3], alpha=0.5)
    plt.hist(Yt[:,3], alpha=0.5)

    # Q, a, s = mo.graphlasso(X, method="bic", log=True)
    # print(a)
    # Q_b, a_b, s_b = mo.graphlasso(Y, method="bic", log=True)
    # print(a_b)

    # fig2, ax2 = plt.subplots(1, 1)
    # pl.precision_matrix(Q, ["A", "B", "C", "D"], fig=fig2, ax=ax2, text=True)

    # fig4, ax4 = plt.subplots(1, 1)
    # pl.precision_matrix(Q_b, ["A", "B", "C", "D"], fig=fig4, ax=ax4, text=True)

    # fig3, ax3 = plt.subplots(1, 1)
    # ax3.plot(s)
    # ax3.plot(s_b)

    plt.show()

if __name__ == "__main__":
    main()
