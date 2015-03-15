import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def struct_scores(x, y, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.plot(x, y)
    ax.axvline(x[np.argmax(y)], color='.5')

def graph_lasso(gl, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.plot(gl.cv_alphas_, np.mean(gl.grid_scores, axis=1), 'o-')
    ax.axvline(gl.alpha_, color='.5')
    ax.set_title('Model selection')
    ax.set_ylabel('Cross-validation score')
    ax.set_xlabel('alpha')


def precision_matrix(Q, labels, fig=None, ax=None, text=False):
    if ax is None:
        ax = plt.gca()

    cax = ax.pcolor(Q, cmap='Blues')

    if text:
        for y in range(Q.shape[0]):
            for x in range(Q.shape[1]):
                plt.text(x + 0.5, y + 0.5, '%.1f' % Q[y, x],
                         horizontalalignment='center',
                         verticalalignment='center')

    fig.colorbar(cax)

    ax.set_xticks(np.arange(Q.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(Q.shape[1])+0.5, minor=False)
    ax.set_xticklabels(labels, rotation=-90, minor=False)
    ax.set_yticklabels(labels, minor=False)
    ax.axis('tight')


def bin_precision_matrix(Q, labels, ax=None):
    if ax is None:
        ax = plt.gca()

    Q[Q != 0] = 1
    np.fill_diagonal(Q, 0)
    print(Q.sum())

    cax = ax.pcolor(Q, cmap='Greys')

    ax.set_xticks(np.arange(Q.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(Q.shape[1])+0.5, minor=False)
    ax.set_xticklabels(labels, rotation=-90, minor=False)
    ax.set_yticklabels(labels, minor=False)
    ax.axis('tight')


def graph(G, layout, ax=None):
    if ax is None:
        ax = plt.gca()

    nx.draw(G, layout, node_color='w', with_labels=True, alpha=1.,
            font_color='b', ax=ax)
