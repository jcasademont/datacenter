import numpy as np
import layouts
import plot as pl
import utils
import matplotlib.pyplot as plt

def main():

    Q = np.load("non_gauss_precision.npy")
    s = np.load("non_gauss_bic_score.npy")
    sg = np.load("gauss_bic_score.npy")

    K = list(layouts.datacenter_layout.keys())
    df = utils.prep_dataframe(keep=K)

    df_shifted = utils.create_shifted_features(df)

    df = df.join(df_shifted, how="outer")

    df = df.dropna()

    plt.figure()
    plt.plot(np.arange(0.1, 5, 0.1), s, label="Non gaussian")
    plt.plot(np.arange(0.1, 5, 0.1), sg, label="Gaussian")
    plt.legend()

    # plt.figure()
    # plt.boxplot(G)
    fig, ax = plt.subplots(1, 1)
    pl.bin_precision_matrix(Q, df.columns.values, ax)

    plt.show()


if __name__ == "__main__":
    main()
