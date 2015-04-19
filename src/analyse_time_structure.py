import utils
import layouts
import plot as pl
import numpy as np
import gmrf.models as mo
import matplotlib.pyplot as plt

def main():
    K = list(layouts.datacenter_layout.keys())

    df = utils.prep_dataframe(keep=K)
    df_shifted = utils.create_shifted_features(df)

    df = df.join(df_shifted, how="outer")

    df = df.dropna()

    Q, alpha, scores = mo.graphlasso(df.values, method='bic', log=True)

    print(scores)
    fig, ax = plt.subplots(1, 1)
    pl.bin_precision_matrix(Q, df.columns.values, ax)

    fig, ax2 = plt.subplots(1, 1)
    ax2.plot(scores)

    plt.show()

if __name__ == "__main__":
    main()
