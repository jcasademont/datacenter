import numpy as np
import layouts
import utils
import matplotlib.pyplot as plt

def main():

    NG = np.load("full_non_gauss_cv_mae_score.npy")
    G = np.load("full_gauss_cv_mae_score.npy")

    K = list(layouts.datacenter_layout.keys())
    df = utils.prep_dataframe(keep=K)

    # plt.figure()
    # plt.plot(NG)

    # plt.figure()
    # plt.plot(G)

    labels = df.columns.values
    labels = list(filter(lambda x: 'ahu' not in x, labels))

    plt.figure()
    plt.boxplot(NG)
    plt.xticks(np.arange(1, 40), labels, rotation=90)

    # plt.figure()
    # plt.boxplot(G)

    plt.show()


if __name__ == "__main__":
    main()
