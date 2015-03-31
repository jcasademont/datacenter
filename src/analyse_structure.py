import utils
import layouts
import plot as pl
import numpy as np
import gmrf.metrics as m
import gmrf.graphs as gr
import gmrf.models as mo
import matplotlib.pyplot as plt


def analyse(X, alphas):
    scores = []
    mean = X.mean(axis=0)
    # mean = [0] * X.shape[1]
    for a in alphas:
        Q, alpha = mo.graphlasso(X, a)
        scores.append(m.bic(X, Q, mean))

    return scores


def main():
    K = list(layouts.datacenter_layout.keys())
    D = ['1_1', '2_1', '8_1', '9_1']

    alphas = np.arange(0.0, 5.0, 0.1)

    df = utils.prep_dataframe(keep=K, drop=D + ['outlet', 'room'], normalise=True)
    print(df.columns.values)
    scores_no_w_no_outlet = analyse(df.values, alphas)

    df = utils.prep_dataframe(keep=K, drop=D + ['room'], normalise=True)
    print(df.columns.values)
    scores_no_w = analyse(df.values, alphas)

    df = utils.prep_dataframe(keep=K, drop=D, normalise=True)
    print(df.columns.values)
    scores_w = analyse(df.values, alphas)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    pl.struct_scores(alphas, scores_w, ax1)
    ax1.set_title("With workload")
    pl.struct_scores(alphas, scores_no_w, ax2)
    ax2.set_title("Without workload")
    pl.struct_scores(alphas, scores_no_w_no_outlet, ax3)
    ax3.set_title("Without workload nor outlet")

    fig2, ax4 = plt.subplots(1, 1)
    Q, alpha = mo.graphlasso(df.values, alphas[np.argmax(scores_w)])
    G = gr.fromQ(Q, df.columns.values)
    pl.graph(G, layouts.datacenter_layout, ax4)

    fig3, ax5 = plt.subplots(1, 1)
    pl.bin_precision_matrix(Q, df.columns.values)

    plt.show()


if __name__ == "__main__":
    main()
