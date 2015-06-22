import os
import sys
import plot as pl
import utils
import layouts
import numpy as np
import matplotlib as mpl
import networkx as nx
from gaussian.gbn import GBN
import transformations as tr
from scipy.linalg import inv
#mpl.use('pgf')

def figsize(scale, fig_height=None, ratio=None):
    fig_width_pt = 426.79                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches

    if fig_height == None:
        if ratio:
            fig_height = ratio * fig_width
        else:
            golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
            fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 8,               # LaTeX default is 10pt font.
    "font.size": 8,
    "legend.fontsize": 10,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt

# I make my own newfig and savefig functions
def newfig(width, height=None, ratio=None, subplots=1, sharex=False):
    plt.clf()
    if height:
        fig, axes = plt.subplots(subplots, sharex=sharex, figsize=figsize(width, fig_height=height))
    else:
        fig, axes = plt.subplots(subplots, sharex=sharex, figsize=figsize(width, ratio=ratio))

    return fig, axes

def savefig(filename):
    filename_pdf = "../doc/images/{}.pdf".format(filename)
    # filename_pgf = "../doc/images/{}.pgf".format(filename)
    plt.savefig(filename_pdf)
    # plt.savefig(filename_pgf)

def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    return ax

# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

K = list(layouts.datacenter_layout.keys())

df_static = utils.prep_dataframe(keep=K)
df_shifted = utils.create_shifted_features(df_static)
df = df_static.join(df_shifted, how="outer")
df = df.dropna()

kf_scores = np.load("../results/gmrfNone_kf_scores.npy")
r2 = np.load("../results/gmrfNone_r2.npy")
kf_scores_hybrid = np.load("../results/hybridNone_kf_scores.npy")
r2_hybrid = np.load("../results/hybridNone_r2.npy")

bns = np.load("../results/hybrid_bns.npy")

# GMRF BIC plot
def gmrf_bic_plot():
    print("BIC plot")
    fig, ax = newfig(1.)
    pl._clean_axes(ax)
    x = np.arange(0.01, 0.5, 0.01)

    bics = np.load("../results/gmrf_bic_scores.npy")
    bics_tr = np.load("../results/gmrf_tr_bic_scores.npy")
    data = [bics, bics_tr]
    lines = ["Original data", "Tranformed data"]
    for rank, column in enumerate(lines):
        ax.plot(x, data[rank], lw=2.5, color=tableau20[rank * 6], label=lines[rank])
        # y_pos = data[rank][-1] - 0.5
        # ax.text(5.1, y_pos, column, fontsize=10, color=tableau20[rank])
    # ax.set_ylabel("BIC score")
    # ax.set_xlabel("$\\alpha$ parameter")
    plt.legend(loc=0, frameon=False)
    plt.tight_layout()
    savefig('gmrf_bic')

# Static BIC plot
def static_bic():
    print("BIC static")
    fig, ax = newfig(1.)
    pl._clean_axes(ax)

    x = np.arange(0.01, 0.5, 0.01)

    bics = np.load("../results/gmrf_static_bic_scores.npy")
    data = [bics]
    lines = ["Original data"]
    for rank, column in enumerate(lines):
        ax.plot(x, data[rank], lw=2.5, color=tableau20[rank * 6], label=lines[rank])
        # y_pos = data[rank][-1] - 0.5
        # ax.text(5.1, y_pos, column, fontsize=10, color=tableau20[rank])
    # ax.set_ylabel("BIC score")
    # ax.set_xlabel("$\\alpha$ parameter")
    # plt.legend(loc=0, frameon=False)
    plt.tight_layout()
    savefig('static_bic')

# Static connectivity matrix
def adj_static():
    print("Adj static")
    Q = np.load("../results/gmrf_static_prec.npy")
    fig, ax = newfig(1.0, ratio=1)
    pl.bin_precision_matrix(Q, df_static.columns.values, ax, add_color=True)
    savefig('adj_static')

# GMRF connectivity matrix
def gmrf_adj_matrix(name=None, highlight=None):
    print("Adj matrix")
    Q = np.load("../results/gmrf_prec.npy")
    fig, ax = newfig(1.0, ratio=1)
    pl.bin_precision_matrix(Q, df.columns.values, ax, add_color=True, interleave=True, highlight=highlight)
    if name:
        savefig(name)
    else:
        savefig('gmrf_con')

# GMRF Eval
def gmrf_res(var):
    print("GMRF Result {}".format(var.upper()))
    fig, (ax, ax1) = newfig(1.0, subplots=2, sharex=True)
    pl.plot_score(kf_scores, df.columns.values, var, ax, label="GMRF", ylabel="Mean Absolute Error")

    pl.plot_r2(r2, df.columns.values, var, ax1,  xlim=(0, 50), ylim=(-1.1, 1.1), ylabel="$R^2$", xlabel="Time steps (15 mins.)")
    savefig('gmrf_res_{}'.format(var))

# Comparaison EO6 GMRF / Hybrid
def comp(var, res1=True, res2=True, r21=True, r22=True, name=None):
    print("Comparaison {}".format(var.upper()))
    fig, (ax, ax1) = newfig(1.0, subplots=2, sharex=True)
    if res1:
        pl.plot_score(kf_scores, df.columns.values, var, ax, label="GMRF", ylabel="Mean Absolute Error")
    if res2:
        pl.plot_score(kf_scores_hybrid, df.columns.values, var, ax, label="HRF", color='orange', xlim=(0, 50), ylabel="Mean Absolute Error")
    ax.legend(loc=0, mode="expand", ncol=2, frameon=False)

    if r21:
        pl.plot_r2(r2, df.columns.values, var, ax1,  xlim=(0, 50), ylim=(-1.1, 1.1), ylabel="$R^2$", xlabel="Time steps (15 mins.)")
    if r22:
        pl.plot_r2(r2_hybrid, df.columns.values, var, ax1, color='orange', xlim=(0, 50), ylim=(-1.1, 1.1), ylabel="$R^2$", xlabel="Time steps (15 mins.)")

    if name:
        savefig(name)
    else:
        savefig('comp_{}'.format(var))


def mrf_ex():
    print("MRF Example")
    fig, ax = newfig(1.0)

    G = nx.Graph()
    for (p, q) in [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D')]:
        G.add_edge(p, q)

    nx.draw_spring(G, node_color='w', ax=ax, with_labels=True, font_size=10)
    savefig('mrf_ex')

def bn_ex():
    print("BN Example")
    fig, ax = newfig(1.0)

    G = nx.DiGraph()
    for (p, q) in [('A', 'C'), ('B', 'C'), ('C', 'D')]:
        G.add_edge(p, q)

    nx.draw_spring(G, node_color='w', ax=ax, with_labels=True, font_size=10)
    savefig('bn_ex')

def qqplot(name=None, axins=True):
    print("GMRF QQplot")
    fig, ax = newfig(1.0)
    Q = np.load("../results/gmrf_prec.npy")
    pl.QQplot(df.values, Q, pos=[50, 120, 10, 120], ax=ax, axins=axins)
    if name:
        savefig(name)
    else:
        savefig('gmrf_qqplot')

def nonparam_qqplot(name=None, axins=True):
    print("Nonparam QQplot")
    fig, ax = newfig(1.0)
    Q = np.load("../results/gmrf_prec.npy")
    X = tr.to_normal(df.values)
    pl.QQplot(X, inv(np.cov(X.T)), pos=[50, 120, 10, 120], ax=ax, color='red', axins=axins)
    if name:
        savefig(name)
    else:
        savefig('nonparam_qqplot')

def hrf_qqplot(var, name=None, axins=True):
    print("HRF QQ plot {}".format(var))
    fig, ax = newfig(1.0)
    idx = np.where(np.array(df.columns.values) == var)[0][0]
    bn = bns[idx]
    gbn = GBN(variables_names=bn[0], nodes=bn[1], edges=bn[2])
    gbn.compute_mean_cov_matrix()
    pl.QQplot(df[gbn.variables_names].values, gbn.precision_, pos=[0, 20, 0, 20], color='orange', ax=ax, axins=axins)
    if name:
        savefig(name)
    else:
        savefig('hrf_qqplot_{}'.format(var))

def nutshell():
    print("Nutshell")
    fig, ax = newfig(1.)
    names = list(filter(lambda x: 'l1_' not in x, df.columns.values))

    errors = np.mean(kf_scores, axis=0)
    hyb_errors = np.mean(kf_scores_hybrid, axis=0)

    limit = []
    pl.barplot(names, [('GMRF', errors), ('HRF', hyb_errors)], [1], limit=limit, ax=ax)
    savefig('nutshell')

def hrf_model():
    print("HRF model")
    fig, ax = newfig(1., ratio=1)
    edges = []

    for bn in bns:
        for e in bn[2]:
            edges.append(e)

    pl.hrf_model(df.columns.values, edges, ax=ax, add_color=True, interleave=True)
    savefig('hrf_model')

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # nutshell()
        hrf_model()
        plt.show()
    elif sys.argv[1] == 'presentation':
        comp('eo6')
        comp('q8')
        comp('eo6', res1=False, res2=True, r21=False, r22=True, name='comp_eo6_1')
        comp('q8', res1=False, res2=True, r21=False, r22=True, name='comp_q8_1')
        gmrf_adj_matrix()
        gmrf_adj_matrix(name="gmrf_adj_qn", highlight=('^(l1_)?q', '^(l1_)?n'))
        hrf_qqplot('eo6', name='simple_hrf_qqplot_eo6', axins=False)
        hrf_qqplot('q8', name='simple_hrf_qqplot_q8', axins=False)
        qqplot(name='simple_gmrf_qqplot', axins=False)
    else:
        basename = os.path.basename(sys.argv[1])
        name = os.path.splitext(basename)[0]
        if name[:4] == 'comp':
            param = name[5:]
            comp(param)
        elif name == 'gmrf_bic':
            gmrf_bic_plot()
        elif name == 'gmrf_con':
            gmrf_adj_matrix()
        elif name == 'mrf_ex':
            mrf_ex()
        elif name == 'bn_ex':
            bn_ex()
        elif name == 'adj_static':
            adj_static()
        elif name == 'static_bic':
            static_bic()
        elif name[:8] == 'gmrf_res':
            param = name[9:]
            gmrf_res(param)
        elif name == 'gmrf_qqplot':
            qqplot()
        elif name[:10] == 'hrf_qqplot':
            param = name[11:]
            hrf_qqplot(param)
        elif name == 'nutshell':
            nutshell()
        elif name == "hrf_model":
            hrf_model()
        elif name == 'nonparam_qqplot':
            nonparam_qqplot()
