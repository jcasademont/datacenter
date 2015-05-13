import argparse
import numpy as np
import layouts
import plot as pl
import utils
import matplotlib.pyplot as plt

def main(layout, file_q, file_bics):
    variables = getattr(layouts, layout)

    K = list(variables.keys())

    df = utils.prep_dataframe(keep=K)
    df_shifted = utils.create_shifted_features(df)
    df = df.join(df_shifted, how="outer")
    df = df.dropna()

    Q = np.load(file_q)
    s = np.load(file_bics)

    plt.figure()
    plt.plot(np.arange(0.1, 5, 0.1), s)

    fig, ax = plt.subplots(1, 1)
    pl.bin_precision_matrix(Q, df.columns.values, ax)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Q and BICs.')
    parser.add_argument('layout', metavar='LAYOUT',
                        help="Layout.")
    parser.add_argument('file_q', metavar='FILE_Q',
                        help="File containing the prec matrix.")
    parser.add_argument('file_bics', metavar='FILE_BICS',
                        help="File containing bics score.")
    args = parser.parse_args()
    main(**vars(args))
