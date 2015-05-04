import utils
import layouts
import plot as pl
import numpy as np
import transformations as tr
import matplotlib.pyplot as plt

from gmrf.gmrf import GMRF
from sklearn.cross_validation import KFold

import multiprocessing as mp

def scoring(X, gmrf, indices, l1_indices, train, test):
    gmrf.fit(X[train])

    X_test = X[test]
    scores = np.zeros((X_test.shape[0], np.size(indices)))

    for i in range(X_test.shape[0]):

        pred = X_test[i, l1_indices]

        for j in range(i, X_test.shape[0]):
            x = X_test[j, :].reshape(1, X_test.shape[1]).copy()
            x[0 , l1_indices] = pred

            pred = gmrf.predict(x, indices).ravel()
            scores[j - i, :] += np.absolute(pred - X_test[j, indices])

    for i in range(X_test.shape[0]):
        scores[i, :] = scores[i, :] / (X_test.shape[0] - i)

    return scores[:568, :]

def main():
    K = list(layouts.datacenter_layout.keys())

    df = utils.prep_dataframe(keep=K, remove_mean=True)
    df_shifted = utils.create_shifted_features(df)

    df = df.join(df_shifted, how="outer")

    df = df.dropna()

    print("Tranform data")
    Z = tr.to_normal(df.values)

    # Indices of racks and IT power consumption
    # indices = np.append(np.arange(38), [42])
    # l1_indices = np.append(np.arange(43, 81), [85])
    indices = np.arange(43)
    l1_indices = np.arange(43, 86)

    # gmrf = GMRF(method="bic")
    # gmrf_gaussian = GMRF(method="bic")

    # gmrf.fit(df.values)
    # gmrf_gaussian.fit(Z)

    # a = gmrf.alpha_
    # ag = gmrf_gaussian.alpha_
    a = 0.1
    ag = 0.1

    gmrf = GMRF(alpha=a)

    kf = KFold(df.shape[0], n_folds=5, shuffle=False)

    pool = mp.Pool(processes=8)

    print(" Non Gaussian model")
    kf_non_gauss_scores = [pool.apply_async(scoring,
                            args=(df.values, gmrf, indices, l1_indices, train, test))
                            for train, test in kf]

    kf_non_gauss_scores = [p.get() for p in kf_non_gauss_scores]

    print(" Gaussian model")
    kf_gauss_scores = [pool.apply_async(scoring,
                            args=(Z, gmrf, indices, l1_indices, train, test))
                            for train, test in kf]

    kf_gauss_scores = [p.get() for p in kf_gauss_scores]

    kf_non_gauss_score = np.sum(kf_non_gauss_scores, axis=0) / len(kf)
    kf_gauss_score = np.sum(kf_gauss_scores, axis=0) / len(kf)

    print("CV, non gaussian score = {}".format(kf_non_gauss_score))
    print("CV, gaussian score = {}".format(kf_gauss_score))

    np.save("n_step_non_gauss_cv_mae_score", kf_non_gauss_score)
    np.save("n_step_gauss_cv_mae_score", kf_gauss_score)

    plt.figure()
    plt.plot(kf_non_gauss_score)

    plt.figure()
    plt.plot(kf_gauss_score)

    plt.show()


if __name__ == "__main__":
    main()
