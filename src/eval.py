import os
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold

import utils
import layouts
import plot as pl
import transformations as tr

from gaussian.gmrf import GMRF
from gaussian.hrf import HRF

def build_vector(data, cols):
    m = np.size(cols)
    x = np.zeros((1, m))

    for k in data.keys():
        i = np.where(cols == k)[0][0]
        x[0, i] = data[k]

    return x

def scoring(df, model, names, train, test, nb_steps=None, id=-1):
    X = df.values

    if not nb_steps:
        nb_steps = X[test].shape[0]

    cols = df.columns.values

    model.fit(X[train])

    X_test = X[test]
    Y = df[names].values[test]
    mean_test = np.mean(Y, axis=0)

    if nb_steps == 1:
        preds = model.predict(X_test, names)
        scores = np.absolute(Y - preds)

        ssRes = np.sum(np.power(scores, 2), axis=0)
        ssTot = np.sum(np.power(Y - mean_test, 2), axis=0)

        r2 = (1 - ssRes / ssTot).reshape(1, np.size(names))

    else:
        scores = np.zeros((nb_steps, np.size(names)))

        ssRes = np.zeros((nb_steps, np.size(names)))
        ssTot = np.zeros((nb_steps, np.size(names)))

        for i in range(X_test.shape[0]):

            pred = X_test[i, :]
            data = dict(zip(cols, pred.ravel()))

            for n in range(min(nb_steps, X_test.shape[0] - i)):
                x = build_vector(data, cols)

                pred = model.predict(x, names).ravel()

                error = pred - Y[i + n, :]

                ssTot[n, :] += np.power(Y[i + n, :] - mean_test, 2)
                ssRes[n, :] += np.power(error, 2)
                scores[n, :] += np.absolute(error)

                data = dict(zip(['l1_' + n for n in names], pred.ravel()))

        r2 = 1 - ssRes / ssTot
        for i in range(nb_steps):
            scores[i, :] = scores[i, :] / (X_test.shape[0] - i)

    if id >= 0:
        print("** Worker {} done.".format(id))

    return r2[:568, :], scores[:568, :]#, model.variances(names)

def main(alpha, transform, temporal, layout, steps, output, hybrid):
    variables = getattr(layouts, layout)

    K = list(variables.keys())

    df = utils.prep_dataframe(keep=K)

    if temporal:
        df_shifted = utils.create_shifted_features(df)
        df = df.join(df_shifted, how="outer")
        df = df.dropna()

        names = list(filter(lambda x: 'l1_' not in x, df.columns.values))

    X = df.values

    if transform:
        print("* Tranform data")
        X = tr.to_normal(X)
        df = pd.DataFrame.from_records(columns=df.columns.values, data=X)

    if hybrid:
        model = HRF(5, 10, variables_names=df.columns.values)
    else:
        model = GMRF(variables_names=df.columns.values, alpha=alpha)

    kf = KFold(df.shape[0], n_folds=5, shuffle=False)

    pool = mp.Pool(processes=5)

    print("* Scoring")
    kf_scores = [pool.apply_async(scoring,
                 args=(df, model, names, train, test, steps, id))
                 for id, (train, test) in enumerate(kf)]

    results = [p.get() for p in kf_scores]
    results = [np.array(t) for t in zip(*results)]

    r2 = results[0]
    kf_scores = results[1]
    #variances = results[2]

    r2 = np.sum(r2, axis=0) / len(kf)
    scores = np.sum(kf_scores, axis=0) / len(kf)
    #var = np.sum(variances, axis=0) / len(kf)

    if output:
        results_name = os.path.join(os.path.dirname(__file__),
                                    "../results/")
        np.save(results_name + output + str(steps) + "_kf_scores", kf_scores)
        np.save(results_name + output + str(steps) + "_scores", scores)
        np.save(results_name + output + str(steps) + "_r2", r2)
       # np.save(results_name + output + str(steps) + "_var", var)

    labels = df.columns.values
    labels = list(filter(lambda x: 'ahu' not in x, labels))

    if steps == 1:
        plt.figure()
        plt.boxplot(scores)
        plt.xticks(np.arange(1, 40), labels, rotation=90)
    else:
        plt.figure()
        plt.plot(scores)
        plt.figure()
        plt.plot(r2)

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
    parser.add_argument('-b', '--hybrid',
                        action='store_true', default=False,
                        help="Use hybrid model.")
    parser.add_argument('-o', '--output',
                        help="File name to store output data.")
    args = parser.parse_args()
    main(**vars(args))
