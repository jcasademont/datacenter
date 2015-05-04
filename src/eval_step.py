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
    # gmrf = GMRF(method="cv", alpha=alpha)
    gmrf = GMRF(alpha=alpha)
    gmrf.fit(X[train])
    preds = gmrf.predict(X[test], indices)

    scores = np.absolute(X[test, indices] - preds)

    return scores[:568, :]

def main():
    K = list(layouts.datacenter_layout.keys())

    df = utils.prep_dataframe(keep=K, remove_mean=False)
    df_shifted = utils.create_shifted_features(df)

    df = df.join(df_shifted, how="outer")

    df = df.dropna()

    # print("Tranform data")
    # Z = tr.to_normal(df.values)

    # Indices of racks and IT power consumption
    indices = np.append(np.arange(38), [42])

    # a = 0
    # ag = 0
    # gmrf = GMRF(method="bic")
    # gmrf_gaussian = GMRF(method="bic")

    # gmrf.fit(df.values)
    # gmrf_gaussian.fit(Z)

    # np.save("non_gauss_Q", gmrf.precision_)
    # np.save("gauss_Q", gmrf_gaussian.precision_)

    # a = gmrf.alpha_
    # ag = gmrf_gaussian.alpha_
    a = 0.1
    ag = 0.1

    # plt.figure()
    # plt.plot(gmrf.bic_scores)
    # plt.plot(gmrf_gaussian.bic_scores)

    kf = KFold(df.shape[0], n_folds=5, shuffle=False)

    pool = mp.Pool(processes=8)

    print("Non Gaussian model")
    kf_non_gauss_scores = [pool.apply_async(scoring,
                            args=(df.values, a, indices, train, test))
                            for train, test in kf]

    kf_non_gauss_scores = [p.get() for p in kf_non_gauss_scores]

    # print("Gaussian model")
    # kf_gauss_scores = [pool.apply_async(scoring,
    #                         args=(Z, ag, indices, train, test))
    #                         for train, test in kf]

    # kf_gauss_scores = [p.get() for p in kf_gauss_scores]

    kf_non_gauss_score = np.sum(kf_non_gauss_scores, axis=0) / len(kf)
    # kf_gauss_score = np.sum(kf_gauss_scores, axis=0) / len(kf)

    print("CV, non gaussian score = {}".format(kf_non_gauss_score))
    # print("CV, gaussian score = {}".format(kf_gauss_score))

    np.save("full_non_gauss_cv_mae_score", kf_non_gauss_score)
    # np.save("full_gauss_cv_mae_score", kf_gauss_score)

    labels = df.columns.values
    labels = list(filter(lambda x: 'ahu' not in x, labels))

    plt.figure()
    plt.boxplot(kf_non_gauss_score)
    plt.xticks(np.arange(1, 40), labels, rotation=90)

    # plt.figure()
    # plt.boxplot(kf_gauss_score)
    # plt.xticks(np.arange(1, 40), labels, rotation=90)

    plt.show()


if __name__ == "__main__":
    main()
