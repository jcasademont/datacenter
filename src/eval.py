import utils
import layouts
import plot as pl
import numpy as np
import evaluation as ev
import gmrf.models as mo
import transformations as tr
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold

import multiprocessing as mp

def scoring(X, alpha, indices, train, test):
    Q, a = mo.graphlasso(X[train], method="bic", alpha=alpha)
    print(" - Selected alpha = {}".format(a))
    scores = ev.score(X[test], indices, Q)

    return scores

def main():
    K = list(layouts.datacenter_layout.keys())

    df = utils.prep_dataframe(keep=K, remove_mean=True)
    df_shifted = utils.create_shifted_features(df)

    df = df.join(df_shifted, how="outer")

    df = df.dropna()

    print(" Tranform data")
    Z = tr.to_normal(df.values)
    # Q = np.load("non_gaussian_prec.npy")
    # Qg = np.load("gaussian_prec.npy")

    # indices = np.array([0, 1])
    indices = np.append(np.arange(37), [42])

    Q, a, s = mo.graphlasso(df.values, method="bic", log=True)
    Qg, ag, sg = mo.graphlasso(Z, method="bic", log=True)

    plt.figure()
    plt.plot(s)
    plt.plot(sg)

    kf = KFold(df.shape[0], n_folds=5, shuffle=False)

    pool = mp.Pool(processes=8)

    print(" Non Gaussian model")
    kf_non_gauss_scores = [pool.apply_async(scoring,
                            args=(df.values, a, indices, train, test))
                            for train, test in kf]

    kf_non_gauss_scores = [p.get() for p in kf_non_gauss_scores]

    print(" Gaussian model")
    kf_gauss_scores = [pool.apply_async(scoring,
                            args=(Z, ag, indices, train, test))
                            for train, test in kf]

    kf_gauss_scores = [p.get() for p in kf_gauss_scores]

    kf_non_gauss_score = np.sum(kf_non_gauss_scores, axis=0) / len(kf)
    kf_gauss_score = np.sum(kf_gauss_scores, axis=0) / len(kf)

    print(" CV, non gaussian score = {}".format(kf_non_gauss_score))
    print(" CV, gaussian score = {}".format(kf_gauss_score))

    plt.show()


if __name__ == "__main__":
    main()
