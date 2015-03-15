import utils
import models
import layouts
import graphs
import plot as pl
import numpy as np
import metrics as m
import matplotlib.pyplot as plt


def analyse(X, alphas, assume_centered=False):
    scores = []
    mean = X.mean(axis=0)
    #mean = [0] * X.shape[1]
    print(mean)
    for a in alphas:
        Q, alpha = models.graphlasso(X, a, assume_centered=assume_centered)
        scores.append(m.bic(X, Q, mean))

    return scores


def main():
    K = list(layouts.datacenter_layout.keys())

    alphas = np.arange(0.0, 5.0, 0.1)

    df = utils.prep_dataframe(keep=K, normalise=False)
    df_lagged = utils.create_lagged_features(df)
    print(df_lagged.head(10))
    scores = analyse(df_lagged.values, alphas)

    df = utils.prep_dataframe(keep=K, normalise=False)
    df_lagged = utils.create_lagged_features(df)
    df_lagged = df_lagged.drop("l1_room_it_power_(kw)", axis=1)
    df_lagged['room_it_power_(kw)'] = df['room_it_power_(kw)']
    scores2 = analyse(df_lagged.values, alphas)

    df = utils.prep_dataframe(keep=K, normalise=False)
    df_lagged = utils.create_lagged_features(df)
    df_lagged = df_lagged.drop("l1_room_it_power_(kw)", axis=1)
    cols = df_lagged.columns.values
    df_lagged = df_lagged.drop(filter(lambda x: "ahu" not in x, cols), axis=1)
    df_lagged['room_it_power_(kw)'] = df['room_it_power_(kw)']
    df_lagged['ahu_1_outlet'] = df['ahu_1_outlet']
    df_lagged['ahu_2_outlet'] = df['ahu_2_outlet']
    df_lagged['ahu_3_outlet'] = df['ahu_3_outlet']
    df_lagged['ahu_4_outlet'] = df['ahu_4_outlet']
    scores3 = analyse(df_lagged.values, alphas, assume_centered=False)

    fig, (ax, ax1, ax2) = plt.subplots(3, 1)
    pl.struct_scores(alphas, scores, ax)
    ax.set_title("BIC score - using deltas of all variable")
    pl.struct_scores(alphas, scores2, ax1)
    ax1.set_title("BIC score - using deltas of all variable but IT power consumption")
    pl.struct_scores(alphas, scores3, ax2)
    ax2.set_title("BIC score - using only deltas of the racks variables")

    plt.show()


if __name__ == "__main__":
    main()
