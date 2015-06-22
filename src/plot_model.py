import argparse
import numpy as np
import layouts
import plot as pl
import utils
import matplotlib.pyplot as plt
from matplotlib import rc

from gaussian.gbn import GBN

def main(layout, gmrf, hybrid):
    variables = getattr(layouts, layout)

    K = list(variables.keys())

    df = utils.prep_dataframe(keep=K)
    df_shifted = utils.create_shifted_features(df)
    df = df.join(df_shifted, how="outer")
    df = df.dropna()

    if gmrf is not None and len(gmrf) > 0:
        Q = np.load(gmrf[0])
        bics = np.load(gmrf[1])

    if hybrid is not None and len(hybrid) > 0:
        BNs = np.load(hybrid[0])

    if gmrf is not None and Q is not None and bics is not None:
        plt.figure()
        plt.plot(np.arange(0.1, 5, 0.1), bics, 'o-')
        plt.ylabel("BIC score")
        plt.xlabel("$\\alpha$ parameter")
        fig, ax = plt.subplots(1, 1)
        pl.bin_precision_matrix(Q, df.columns.values, ax, add_color=True)

    if hybrid is not None and BNs is not None:
        for i in range(len(BNs)):
            fig, ax = plt.subplots(1, 1)
            fig.subplots_adjust(right=0.75)
            fig.subplots_adjust(left=0.75)
            pl.plot_bn(BNs, df.columns.values, df.columns.values[i], ax)
            ax.set_title(df.columns.values[i])

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
