import utils
import layouts
import plot as pl
import numpy as np
import gmrf.metrics as m
import gmrf.graphs as gr
import gmrf.models as mo
import matplotlib.pyplot as plt

def main():
    K = list(layouts.datacenter_layout.keys())

    # df = utils.prep_dataframe(keep=K, drop=['outlet', 'room'])
    # Q_rack, alpha_rack = mo.graphlasso(df.values, method="bic")

    # df = utils.prep_dataframe(keep=K, drop=['room'])
    # Q_ahu, alpha_ahu = mo.graphlasso(df.values, method="bic")

    df = utils.prep_dataframe(keep=K, remove_mean=True)
    Q_all, alpha_all, scores = mo.graphlasso(df.values, method="bic", log=True)

    print(alpha_all)

    fig, ax = plt.subplots(1, 1)
    G = gr.fromQ(Q_all, df.columns.values)
    pl.graph(G, layouts.datacenter_layout, ax)

    fig2, ax2 = plt.subplots(1, 1)
    pl.bin_precision_matrix(Q_all, df.columns.values, ax2)

    fig3, ax3 = plt.subplots(1, 1)
    ax3.plot(scores)

    plt.show()

if __name__ == "__main__":
    main()
