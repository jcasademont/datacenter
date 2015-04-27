import utils
import layouts
import plot as pl
import numpy as np
import transformations as tr
import matplotlib.pyplot as plt

from gmrf.gmrf import GMRF

def main():
    K = list(layouts.datacenter_layout.keys())

    df = utils.prep_dataframe(keep=K)
    df_shifted = utils.create_shifted_features(df)

    df = df.join(df_shifted, how="outer")

    df = df.dropna()

    print("Tranform data")
    Z = tr.to_normal(df.values)

    gmrf = GMRF(method="bic")
    gmrf_gaussian = GMRF(method="bic")

    gmrf.fit(df.values)
    gmrf_gaussian.fit(Z)

    print(gmrf.alpha_)
    print(gmrf_gaussian.alpha_)

    np.save("non_gauss_bic_score", gmrf.bic_scores)
    np.save("gauss_bic_score", gmrf_gaussian.bic_scores)
    np.save("non_gauss_precision", gmrf.precision_)
    np.save("gauss_precision", gmrf_gaussian.precision_)

    plt.figure()
    plt.plot(np.arange(0.1, 5.0, 0.1), gmrf.bic_scores)
    plt.plot(np.arange(0.1, 5.0, 0.1), gmrf_gaussian.bic_scores)

    fig, ax = plt.subplots(1, 1)
    pl.bin_precision_matrix(gmrf.precision_, df.columns.values, ax)

    plt.show()

if __name__ == "__main__":
    main()
