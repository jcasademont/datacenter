import graphs
import layouts
import plot as pl
import metrics as m

import pandas as pd
import matplotlib.pyplot as plt


def computeLasso():
    #K = ['e9', 'h12', 'e12', 'h9', 'k8',
    #     'k9', 'k11', 'k12', 'n1', 'n2', 'outlet']

    K = layout.keys()
    D = ['1_1', '2_1', '8_1', '9_1', 'outlet']
    df_X = prep_data(keep=K, drop=D)
    df_X = df_X - df_X.mean()

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    pl.graph_lasso(gl, fig, ax2)
    pl.precision_matrix(gl.precision_, df_X.columns.values, fig=fig, ax=ax1)
    pl.graph(G, layout, ax3)

    plt.show()

if __name__ == "__main__":
    computeLasso()
