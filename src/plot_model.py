import argparse
import numpy as np
import layouts
import plot as pl
import utils
import matplotlib.pyplot as plt

from gaussian.gbn import GBN

def main(layout, gmrf, hybrid):
    variables = getattr(layouts, layout)

    K = list(variables.keys())

    df = utils.prep_dataframe(keep=K)
    df_shifted = utils.create_shifted_features(df)
    df = df.join(df_shifted, how="outer")
    df = df.dropna()

    if gmrf and len(gmrf) > 0:
        Q = np.load(gmrf[0])
        bics = np.load(gmrf[1])

    if len(hybrid) > 0:
        BNs = np.load(hybrid[0])

    if gmrf and Q and bics:
        plt.figure()
        plt.plot(np.arange(0.1, 5, 0.1), bics)
        fig, ax = plt.subplots(1, 1)
        pl.bin_precision_matrix(Q, df.columns.values, ax)

    if BNs is not None:
        for i in range(len(BNs)):
            gbn = GBN(variables_names=BNs[i][0], nodes=BNs[i][1],
                      edges=BNs[i][2])
            gbn.compute_mean_cov_matrix()
            fig, ax = plt.subplots(1, 1)
            ax.set_title(df.columns.values[i])
            pl.bin_precision_matrix(gbn.precision_, gbn.variables_names, ax)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Q and BICs.')
    parser.add_argument('layout', metavar='LAYOUT',
                        help="Layout.")
    parser.add_argument('-g', '--gmrf', nargs='+',
                        help="Files for GMRF (Prec and BICs).")
    parser.add_argument('-b', '--hybrid', nargs='+',
                        help="File for HRF (BNs).")
    args = parser.parse_args()
    main(**vars(args))
