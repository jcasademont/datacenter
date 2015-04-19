import utils
import layouts
import plot as pl
import numpy as np
import gmrf.metrics as m
import gmrf.graphs as gr
import gmrf.models as mo
import matplotlib.pyplot as plt
import pandas as pd

import stat_tests as st
import transformations as tr
from scipy.stats import norm

def log_likelihood(X):
    r = []
    for i in range(X.shape[1]):
        y = X[:, i]
        ll = norm.logpdf(y, loc=np.mean(y), scale=np.std(y)).sum()
        r.append(ll)

    return r

def main():
    K = list(layouts.datacenter_layout.keys())

    df = utils.prep_dataframe(keep=K)
    df_shifted = utils.create_shifted_features(df)

    df = df.join(df_shifted, how="outer")

    df = df.dropna()

    Z = tr.to_normal(df.values)

    a, r = st.test_normality(df.values)
    print("Accepted: {}, Refused: {}".format(a, len(r)))
    # if( len(a) > 0):
    #     plt.hist(df.values[:, a], alpha=0.5)
    a, r = st.test_normality(Z)
    print("Accepted: {}, Refused: {}".format(a, len(r)))
    # if( len(a) > 0):
    #     plt.hist(df.values[:, a], alpha=0.5)
    ll = log_likelihood(df.values)
    llg = log_likelihood(Z)

    df_Z = pd.DataFrame.from_records(Z, index=df.index, columns=df.columns)

    df.hist()
    df_Z.hist()

    print("Non gaussian graph lasso")
    Q, a, s = mo.graphlasso(df.values, method="bic", log=True)
    print("Gaussian graph lasso")
    Qg, ag, sg = mo.graphlasso(Z, method="bic", log=True)

    print(a)
    print(ag)

    # Q[np.absolute(Q) < 0.1] = 0
    # Qg[np.absolute(Qg) < 0.1] = 0

    # fig0, (ax0, ax1) = plt.subplots(1, 2)
    # G = gr.fromQ(Q, df.columns.values)
    # pl.graph(G, layouts.datacenter_layout, ax0)
    # ax0.set_title("Non Gaussian")

    # G = gr.fromQ(Qg, df.columns.values)
    # pl.graph(G, layouts.datacenter_layout, ax1)
    # ax1.set_title("Gaussian")

    fig2, ax2 = plt.subplots(1, 1)
    pl.bin_precision_matrix(Q, df.columns.values, ax=ax2)

    fig4, ax4 = plt.subplots(1, 1)
    pl.bin_precision_matrix(Qg, df.columns.values, ax=ax4)

    fig3, ax3 = plt.subplots(1, 1)
    ax3.plot(s, label='Non gaussian')
    ax3.plot(sg, label='Gaussian')
    ax3.legend(loc='upper left')

    # plt.subplot(1, 2, 1)
    # plt.plot(np.sort(Q.ravel()))
    # plt.title("Non gaussian")
    # plt.subplot(1, 2, 2)
    # plt.plot(np.sort(Qg.ravel()))
    # plt.title("Gaussian")

    df_Q = pd.DataFrame(Q, index=df.columns.values, columns=df.columns.values)
    df_Qg = pd.DataFrame(Qg, index=df.columns.values, columns=df.columns.values)
    df_Q.to_html(open('Q_non_gauss.html', 'w'))
    df_Qg.to_html(open('Q_gauss.html', 'w'))

    plt.show()

if __name__ == "__main__":
    main()
