import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import utils
import layouts
import plot as pl
import transformations as tr

from gmrf.gmrf import GMRF

def main(method, transform, temporal, layout, threshold, output):
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

    gmrf = GMRF(method=method[0])

    gmrf.fit(X)

    print("* Selected alpha = {}".format(gmrf.alpha_))

    if threshold:
        Q = gmrf.precision_.copy()
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
        results_name = os.path.join(os.path.dirname(__file__),
                                    "../results/")
        np.save(results_name + output, gmrf.precision_)
        np.save(results_name + output + "_bic_scores", gmrf.bic_scores)

    plt.figure()
    plt.plot(gmrf.bic_scores)

    fig, ax = plt.subplots(1, 1)
    pl.bin_precision_matrix(gmrf.precision_, df.columns.values, ax)

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
    parser.add_argument('-r', '--threshold',
                        action='store_true', default=False,
                        help="Use the threshold method.")
    parser.add_argument('-o', '--output',
                        help="File name to store output data.")
    args = parser.parse_args()
    main(**vars(args))
