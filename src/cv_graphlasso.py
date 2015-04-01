import utils
import layouts
import plot as pl
import gmrf.graphs as gr
import gmrf.models as mo
import matplotlib.pyplot as plt


def estimate_graph(df, title):
    """Estimate and plot GMRF graph"""
    Q, alpha = mo.graphlasso(df.values, cv=True)
    G = gr.fromQ(Q, df.columns.values)

    fig, ax = plt.subplots(1, 1)
    pl.graph(G, layouts.datacenter_layout, ax)
    print(title + ": alpha = {}".format(alpha))
    ax.set_title(title)

    return G


def main():
    """Use cross-validaton to estimate GMRF using different datasets
    such as absolute data or lagged data ..."""

    K = list(layouts.datacenter_layout.keys())

    # Absolute values for racks, workload, ahus outlet
    df = utils.prep_dataframe(keep=K)
    G = estimate_graph(df, "Abs values")
    gr.saveGraph(G)

    # Absolute values for racks, no workload, no ahus
    df_no_w_no_ahs = utils.prep_dataframe(keep=K, drop=['outlet', 'room'])
    estimate_graph(df_no_w_no_ahs, "Abs values, no w/ahus")

    # Lagged values for racks, workload, ahus outlet
    df_lagged = utils.create_lagged_features(df)
    estimate_graph(df_lagged, "Lagged values")

    # Lagged values for racks and ahus outlet, absolute workload
    df_lagged = utils.create_lagged_features(df)
    df_lagged['room_it_power_(kw)'] = df['room_it_power_(kw)']
    estimate_graph(df_lagged, "Lagged values, abs workload")

    # Lagged values for racks and absolute workload, ahus
    df_lagged = utils.create_lagged_features(df)
    df_lagged['room_it_power_(kw)'] = df['room_it_power_(kw)']
    df_lagged['ahu_1_outlet'] = df['ahu_1_outlet']
    df_lagged['ahu_2_outlet'] = df['ahu_2_outlet']
    df_lagged['ahu_3_outlet'] = df['ahu_3_outlet']
    df_lagged['ahu_4_outlet'] = df['ahu_4_outlet']
    estimate_graph(df_lagged, "Lagged values, abs workload, abs ahus")

    plt.show()

if  __name__ == "__main__":
    main()
