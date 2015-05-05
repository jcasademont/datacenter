import os
import argparse
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold

import utils
import layouts
import plot as pl
import transformations as tr

from gmrf.gmrf import GMRF

def scoring(X, gmrf, indices, l1_indices, train, test, nb_steps=None, id=-1):
    if not nb_steps:
        nb_steps = X[test].shape[0]

    gmrf.fit(X[train])

    X_test = X[test]
    mean_test = np.mean(X_test[:, indices], axis=0)

    if nb_steps == 1:
        preds = gmrf.predict(X_test, indices)
        scores = np.absolute(X_test[:, indices] - preds)

        ssRes = np.sum(np.power(scores, 2), axis=0)
        ssTot = np.sum(np.power(X_test[:, indices] - mean_test, 2),
                       axis=0)

        r2 = (1 - ssRes / ssTot).reshape(1, np.size(indices))

    else:
        scores = np.zeros((nb_steps, np.size(indices)))

        ssRes = np.zeros((nb_steps, np.size(indices)))
        ssTot = np.zeros((nb_steps, np.size(indices)))

        for i in range(X_test.shape[0]):

            pred = X_test[i, l1_indices]

            for n in range(min(nb_steps, X_test.shape[0] - i)):
                x = X_test[i + n, :].reshape(1, X_test.shape[1]).copy()
                x[0 , l1_indices] = pred

                pred = gmrf.predict(x, indices).ravel()

                error = pred - X_test[i + n, indices]

                ssTot[n, :] += np.power(X_test[i + n, indices]
                                        - mean_test, 2)
                ssRes[n, :] += np.power(error, 2)
                scores[n, :] += np.absolute(error)

        r2 = 1 - ssRes / ssTot
        for i in range(nb_steps):
            scores[i, :] = scores[i, :] / (X_test.shape[0] - i)

    if id >= 0:
        print("** Worker {} done.".format(id))

    return r2[:568, :], scores[:568, :], gmrf.variances(indices)

def main(alpha, transform, temporal, layout, steps, output):
    variables = getattr(layouts, layout)

    K = list(variables.keys())

    df = utils.prep_dataframe(keep=K)

    if temporal:
        df_shifted = utils.create_shifted_features(df)
        df = df.join(df_shifted, how="outer")
        df = df.dropna()

    X = df.values

    if transform:
        print("* Tranform data")
        X = tr.to_normal(X)

    # indices = np.append(np.arange(38), [42])
    # l1_indices = np.append(np.arange(43, 81), [85])
    indices = np.arange(38)
    l1_indices = np.arange(42, 80)

    gmrf = GMRF(alpha=alpha)

    kf = KFold(df.shape[0], n_folds=5, shuffle=False)

    pool = mp.Pool(processes=8)

    print("* Scoring")
    kf_scores = [pool.apply_async(scoring,
                 args=(X, gmrf, indices, l1_indices,
                       train, test, steps, id))
                 for id, (train, test) in enumerate(kf)]

    results = [p.get() for p in kf_scores]
    results = [np.array(t) for t in zip(*results)]

    r2 = results[0]
    kf_scores = results[1]
    variances = results[2]

    r2 = np.sum(r2, axis=0) / len(kf)
    scores = np.sum(kf_scores, axis=0) / len(kf)
    var = np.sum(variances, axis=0) / len(kf)

    if output:
        results_name = os.path.join(os.path.dirname(__file__),
                                    "../results/")
        np.save(results_name + output + str(steps) + "_kf_scores", kf_scores)
        np.save(results_name + output + str(steps) + "_scores", scores)
        np.save(results_name + output + str(steps) + "_r2", r2)
        np.save(results_name + output + str(steps) + "_var", var)

    labels = df.columns.values
    labels = list(filter(lambda x: 'ahu' not in x, labels))

    if steps == 1:
        plt.figure()
        plt.boxplot(scores)
        plt.xticks(np.arange(1, 40), labels, rotation=90)
    else:
        plt.figure()
        plt.plot(scores)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate GMRF.')
    parser.add_argument('layout', metavar='LAYOUT',
                        help="Layout containing the variables.")
    parser.add_argument('alpha', type=float,
                        help="Selected alpha to evaluate.")
    parser.add_argument('-n', '--steps', type=int,
                        help="Number of steps use for evaluation.")
    parser.add_argument('-t', '--transform',
                        action='store_true', default=False,
                        help="Transform the data to Gaussian.")
    parser.add_argument('-p', '--temporal',
                        action='store_true', default=False,
                        help="Use temporal data.")
    parser.add_argument('-o', '--output',
                        help="File name to store output data.")
    args = parser.parse_args()
    main(**vars(args))
