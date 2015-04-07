import utils
import layouts
import plot as pl
import numpy as np
import gmrf.models as mo
import gmrf.graphs as gr
import gmrf.metrics as m
import matplotlib.pyplot as plt

def main():
    K = list(layouts.datacenter_layout.keys())

    df = utils.prep_dataframe(keep=K)
    df_lagged = utils.create_lagged_features(df)
    Q_all, alpha_all = mo.graphlasso(df_lagged.values, method="bic")

    df = utils.prep_dataframe(keep=K)
    df_lagged = utils.create_lagged_features(df)
    df_lagged['room_it_power_(kw)'] = df['room_it_power_(kw)']
    Q_ahu, alpha_ahu = mo.graphlasso(df_lagged.values, method="bic")

    df = utils.prep_dataframe(keep=K)
    df_lagged = utils.create_lagged_features(df)
    cols = df_lagged.columns.values
    df_lagged['room_it_power_(kw)'] = df['room_it_power_(kw)']
    df_lagged['ahu_1_outlet'] = df['ahu_1_outlet']
    df_lagged['ahu_2_outlet'] = df['ahu_2_outlet']
    df_lagged['ahu_3_outlet'] = df['ahu_3_outlet']
    df_lagged['ahu_4_outlet'] = df['ahu_4_outlet']
    Q_rack, alpha_rack = mo.graphlasso(df_lagged.values, method="bic")

    fig, ax = plt.subplots(1, 1)
    G = gr.fromQ(Q_rack, df_lagged.columns.values)
    pl.graph(G, layouts.datacenter_layout, ax)

    fig2, ax2 = plt.subplots(1, 1)
    pl.bin_precision_matrix(Q_rack, df_lagged.columns.values)

    plt.show()

if __name__ == "__main__":
    main()
