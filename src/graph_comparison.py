import utils
import models
import layouts
import graphs
import plot as pl
import numpy as np
import metrics as m
import matplotlib.pyplot as plt
from numpy.linalg import inv


def main():
    K = list(layouts.datacenter_layout.keys())
    D = ['1_1', '2_1', '8_1', '9_1']
    df = utils.prep_dataframe(keep=K, drop=D, remove_mean=False)
    print(df.columns.values)

    Q, alpha = models.graphlasso(df.values, 0.2)
    score_gl = m.bic(df.values, Q, df.values.mean(axis=0))

    Q = models.empirical(np.cov(df.values.T), df.columns.values, graphs.grid_graph)
    score_emp = m.bic(df.values, Q, df.values.mean(axis=0))
    G = graphs.fromQ(Q, df.columns.values)

    score_full = m.bic(df.values, inv(np.cov(df.values.T)), df.values.mean(axis=0))

    fig, ax = plt.subplots(1, 1)
    ax.bar(np.arange(3), np.abs([score_gl, score_emp, score_full]), 0.35)
    xTickMarks = ['Glasso', 'Grid graph', 'Full graph']
    ax.set_xticks(np.arange(3) + 0.35/2)
    ax.set_ylabel("-BIC")
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=10)

    fig2, ax2 = plt.subplots(1, 1)
    pl.graph(G, layouts.datacenter_layout, ax2)

    plt.show()

if __name__ == "__main__":
    main()
