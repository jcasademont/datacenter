import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
import layouts
import plot as pl
import transformations as tr

from gaussian.hrf import HRF
from gaussian.gmrf import GMRF

def main(method, transform, temporal, layout, hybrid, threshold, output):
    variables = getattr(layouts, layout)

    K = list(variables.keys())

    df = utils.prep_dataframe(keep=K)

    if temporal:
        df_shifted = utils.create_shifted_features(df)
        df = df.join(df_shifted, how="outer")
        df = df.dropna()

    if transform:
        print("* Tranform data")
        X = tr.to_normal(df.values)
        df = pd.DataFrame(X, index=df.index.values, columns=df.columns.values)

    X = df.values

    if hybrid:
        model = HRF(variables_names=df.columns.values)
    else:
        model = GMRF(method=method[0])

    model.fit(X)

    if not hybrid:
        print("* Selected alpha = {}".format(model.alpha_))
    else:
        print("* Selected k = {}, k star = {}".format(model.k, model.k_star))

    if threshold:
        Q = model.precision_.copy()
        ts = np.arange(0., 1., 0.001)

        bics = np.empty(len(ts))
        connectivity = np.empty(len(ts))

        n = Q.shape[0]
        gmrf_test = GMRF()
        gmrf_test.mean_ = np.mean(X, axis=0)
        for i, t in enumerate(ts):
            Q[Q < t] = 0
            gmrf_test.precision_ = Q
            bics[i], _ = gmrf_test.bic(X)
            connectivity[i] = 1 - np.size(np.where(Q == 0)[0]) / (n * n)

        fig, (ax, ax1) = plt.subplots(2, 1)
        ax.plot(connectivity)
        ax1.plot(bics)

    if output:
        results_name = os.path.join(os.path.dirname(__file__), "../results/")
        if hybrid:
            BNs = np.empty(len(model.variables_names), dtype=object)
            for i in range(len(BNs)):
                BNs[i] = (model.bns[i].variables_names, model.bns[i].nodes, model.bns[i].edges)
            np.save(results_name + output + "_bns", BNs)
        else:
            np.save(results_name + output + "_prec", model.precision_)
            np.save(results_name + output + "_mean", model.mean_)
            np.save(results_name + output + "_bic_scores", model.bic_scores)

    if not hybrid:
        plt.figure()
        plt.plot(model.bic_scores)

        fig, ax = plt.subplots(1, 1)
        pl.bin_precision_matrix(model.precision_, df.columns.values, ax)

        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create GMRF model.')
    parser.add_argument('layout', metavar='LAYOUT',
                        help="Layout containing the variables.")
    parser.add_argument('-m', '--method',
                        nargs=1, choices=['bic', 'cv'],
                        default=['bic'],
                        help="Method to use.")
    parser.add_argument('-t', '--transform',
                        action='store_true', default=False,
                        help="Transform the data to Gaussian.")
    parser.add_argument('-p', '--temporal',
                        action='store_true', default=False,
                        help="Use temporal data.")
    parser.add_argument('-b', '--hybrid',
                        action='store_true', default=False,
                        help="Use hybrid model.")
    parser.add_argument('-r', '--threshold',
                        action='store_true', default=False,
                        help="Use the threshold method.")
    parser.add_argument('-o', '--output',
                        help="File name to store output data.")
    args = parser.parse_args()
    main(**vars(args))
