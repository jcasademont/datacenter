import utils
import layouts
import plot as pl
import numpy as np
import transformations as tr
import matplotlib.pyplot as plt

from gmrf.gmrf import GMRF
from sklearn.cross_validation import KFold

import multiprocessing as mp

def scoring(X, alpha, indices, train, test):
    gmrf = GMRF(method="bic", alpha=alpha)
    gmrf.fit(X[train])
    scores = gmrf.score(X[test], indices)

    return scores

def main():
    K = list(layouts.datacenter_layout.keys())

    df = utils.prep_dataframe(keep=K, remove_mean=False)
    df_shifted = utils.create_shifted_features(df)

    df = df.join(df_shifted, how="outer")

    df = df.dropna()

    print("Tranform data")
    Z = tr.to_normal(df.values)

    # Indices of racks and IT power consumption
    indices = np.append(np.arange(38), [42])

    gmrf = GMRF(method="bic")
    gmrf_gaussian = GMRF(method="bic")

    gmrf.fit(df.values)
    gmrf_gaussian.fit(Z)

    a = gmrf.alpha_
    ag = gmrf_gaussian.alpha_

    plt.figure()
    plt.plot(gmrf.bic_scores)
    plt.plot(gmrf_gaussian.bic_scores)

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

    print("CV, non gaussian score = {}".format(kf_non_gauss_score))
    print("CV, gaussian score = {}".format(kf_gauss_score))

    np.save("non_gauss_cv_mae_score", kf_non_gauss_score)
    np.save("gauss_cv_mae_score", kf_gauss_score)

    plt.show()


if __name__ == "__main__":
    main()
