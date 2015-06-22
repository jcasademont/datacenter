import re
import math
import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LogNorm
import matplotlib.mlab as mlab

from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes

import scipy.stats as stats

from scipy.linalg import inv

# These are the "Tableau 20" colors as RGB.
tableau = [(31, 119, 180), (174, 199, 232), (255, 127, 14),
           (255, 187, 120), (44, 160, 44), (152, 223, 138),
           (214, 39, 40), (255, 152, 150), (148, 103, 189),
           (197, 176, 213), (140, 86, 75), (196, 156, 148),
           (227, 119, 194), (247, 182, 210), (127, 127, 127),
           (199, 199, 199), (188, 189, 34), (219, 219, 141),
           (23, 190, 207), (158, 218, 229), (65, 68, 81)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau)):
    r, g, b = tableau[i]
    tableau[i] = (r / 255., g / 255., b / 255.)

colorscheme = {'blue': tableau[0], 'light_blue': tableau[1], 'orange': tableau[2], 'light_orange': tableau[3], 'green': tableau[5], 'light_green': tableau[4], 'red': tableau[6], 'light_red': tableau[7], 'purple': tableau[8], 'light_purple': tableau[9], 'cyan': tableau[18], 'light_cyan': tableau[19], 'gray': tableau[20]}

cdict = {
    'red'  :  (
               (0., 31./255, 31./255),
               (0.3, 158./255, 158./255),
               (0.5, 152./255, 152./255),
               (0.6, 255./255, 255./255),
               (0.7, 255./255, 255./255),
               (0.85, 214./255, 214./255),
               (1., 177./255, 177./255)
              ),
    'green':  (
               (0., 119./255, 119./255),
               (0.3, 223./255, 223./255),
               (0.5, 218./255, 218./255),
               (0.6, 221./255, 221./255),
               (0.7, 127./255, 127./255),
               (0.85, 39./255, 39./255),
               (1., 3./255, 3./255)
              ),
    'blue' :  (
               (0., 180./255, 180./255),
               (0.3, 229./255, 229./255),
               (0.5, 138./255, 138./255),
               (0.6, 113./255, 113./255),
               (0.7, 14./255, 14./255),
               (0.85, 40./255, 40./255),
               (1., 24./255, 24./255),
              )
}

tableau_cmap = colors.LinearSegmentedColormap('tableau_colormap', cdict, 1024)

def transform_label(label):
    if 'l1' in label:
        new_label = (label[3:]).upper()
        new_label = new_label.replace("_", "")
        if 'AHU' in new_label:
            new_label = new_label[:4]
        new_label = '$' + new_label + '_{t-1}$'
    else:
        new_label = label.upper()
        new_label = new_label.replace("_", "")
        if 'AHU' in new_label:
            new_label = new_label[:4]
        new_label = '$' + new_label + '_t$'

    return new_label

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

def plot_bigauss(mu, cov, ax=None, data=None):
    if ax is None:
        ax = plt.gca()

    if data is not None:
        data = list(zip(*data))
        x = data[0]
        y = data[1]
        ax.scatter(x, y, c='k', alpha=0.1)

    delta = 0.025
    diag = np.sqrt(np.diag(cov))
    ci = 2.58 * np.sqrt(diag)
    x = np.arange(mu[0] - ci[0], mu[0] + ci[0], delta)
    y = np.arange(mu[1] - ci[1], mu[1] + ci[1], delta)

    X, Y = np.meshgrid(x, y)
    Z = mlab.bivariate_normal(X, Y, diag[0], diag[1], mu[0], mu[1], cov[0][1])

    CS = ax.contour(X, Y, Z, alpha=.7, cmap=tableau_cmap, linewidths=2)
    # CS.collections[0].remove()

def scatter_matrix(var, names, mu, Q, S, data=None, given=None, fig=None):
    if fig is None:
        fig = plt.figure()

    var_idx = np.where(names == var)[0][0]

    if given is None:
        given = names

    nb_dep = 0
    for i, n in enumerate(names):
        if Q[var_idx, i] != 0 and n in given and n != var:
            nb_dep += 1

    nb_x = min(len(given) + 1, 3)
    nb_y = math.ceil((nb_dep + 1) / nb_x)
    ax = fig.add_subplot(nb_y, nb_x, 1)
    # _clean_axes(ax)
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)

    sig = np.sqrt(S[var_idx, var_idx])
    ci = 2.58 * sig

    if data is not None:
        ax.hist(data[:, var_idx], normed=True, bins=20, color=colorscheme['gray'])

    x = np.linspace(mu[var_idx] - ci, mu[var_idx] + ci, 1000)
    y = scipy.stats.norm.pdf(x, loc=mu[var_idx], scale=sig)

    ax.plot(x, y, lw=2.5, c=colorscheme['red'])
    ax.set_title(transform_label(var))

    p = 2
    for i, n in enumerate(names):
        if Q[var_idx, i] != 0 and n in given and n != var:
            ax = fig.add_subplot(nb_y, nb_x, p)

            mean = mu[[var_idx, i]]
            cov = (S[:, [var_idx, i]])[[var_idx, i]]

            plot_bigauss(mean, cov, ax=ax, data=data[:, [var_idx, i]])
            ax.set_title(transform_label(n))
            _clean_axes(ax)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            p += 1

def QQplot(obs, Q, pos, color=None, ax=None, axins=True):
        if not color:
            color='blue'

        if ax is None:
            ax = plt.gca()

        mean = np.mean(obs, axis=0)
        # obs = np.random.multivariate_normal(mean, S, 3000)

        md2 = np.diag(np.dot(np.dot(obs - mean, Q), (obs -mean).T))
        sorted_md2 = np.sort(md2)
        v = (np.arange(1, obs.shape[0] + 1) - 0.375) / (obs.shape[0] + 0.25)
        quantiles = scipy.stats.chi2.ppf(v, df=obs.shape[1])

        # axins = inset_axes(ax, width="60%", height=1., loc=2)
        if axins:
            axins = inset_axes(ax, width="60%", height=1., loc=2)
            axins.axis(pos)

            axins.get_xaxis().tick_bottom()
            axins.get_yaxis().tick_left()
            # Remove the tick marks; they are unnecessary with the tick lines we just plotted.
            axins.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="off", left="off", right="off", labelleft="off")
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
            axins.xaxis.set_major_locator(MaxNLocator(nbins=1, prune='lower'))
            axins.scatter(quantiles, sorted_md2, color=colorscheme[color], alpha=0.3)
            axins.plot(quantiles, quantiles, color=colorscheme['green'], lw=2.5)


        ax.scatter(quantiles, sorted_md2, color=colorscheme[color], alpha=0.3)
        ax.plot(quantiles, quantiles, color=colorscheme['green'], lw=2.5)
        _clean_axes(ax)

def precision_matrix(Q, labels, fig=None, ax=None, text=False):
    if ax is None:
        ax = plt.gca()

    Q = np.absolute(Q)
    m = np.max(Q)

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

def bin_precision_matrix(Q, labels, ax=None, add_color=False, interleave=False, highlight=None):
    if ax is None:
        ax = plt.gca()

    R = np.array(Q, copy=True)

    R[R != 0] = 1
    if interleave:
        mid = int(R.shape[0] / 2)
        new_indices = []

        for i in range(mid):
            new_indices.append(i)
            new_indices.append(i + mid)

        R = (R[new_indices, :])[:, new_indices]
        labels = labels[new_indices]

    if add_color:
        clusters = ['^(l1_)?ahu', '^(l1_)?e', '^(l1_)?h', '^(l1_)?k', '^(l1_)?n', '^(l1_)?q']

        cs = ['white', 'black', 'grey']
        for c in ['blue', 'purple', 'red', 'cyan', 'orange', 'green']:
            cs.append(colorscheme[c])
            cs.append(colorscheme['light_' + c])
        values = np.arange(len(cs))

        for (i, j), v in np.ndenumerate(R):
            if v == 0:
                continue

            if highlight and R[i, j] == 1 and not ((re.search(highlight[0], labels[i]) and re.search(highlight[1], labels[j])) or (re.search(highlight[0], labels[j]) and re.search(highlight[1], labels[i]))):
                R[i, j] = R[j, i] = 2

            for k, c in enumerate(clusters):
                if re.search(c, labels[i]) and re.search(c, labels[j]):
                    if highlight and not (re.search(highlight[0], labels[i]) and re.search(highlight[0], labels[j])):
                        R[i, j] = R[j, i] = values[3 + 2 * k + 1]
                    else:
                        R[i, j] = R[j, i] = values[3 + 2 * k]
                    break

        cmap = colors.ListedColormap(cs)
        bounds = np.arange(len(cs))
        norm = colors.BoundaryNorm(bounds, cmap.N)
    else:
        cmap='Greys'
        norm=None

    labels = [transform_label(label) for label in labels]

    ax2 = ax.twinx()
    ax3 = ax.twiny()

    ax.pcolor(R, cmap=cmap, norm=norm, edgecolor='white')
    ax2.pcolor(R, cmap=cmap, norm=norm, edgecolor='white')
    ax3.pcolor(R, cmap=cmap, norm=norm, edgecolor='white')

    even_labels = [l for i, l in enumerate(labels) if i % 2 == 0]
    odd_labels = [l for i, l in  enumerate(labels) if i % 2 == 1]

    ax.set_xticks(np.arange(0, R.shape[0], 2)+0.25, minor=False)
    ax.set_yticks(np.arange(0, R.shape[1], 2)+0.25, minor=False)
    ax.set_xticklabels(even_labels, rotation=-90, minor=False)
    ax.set_yticklabels(even_labels, minor=False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax2.set_yticks(np.arange(1, R.shape[1], 2)+0.25, minor=False)
    ax2.set_yticklabels(odd_labels, minor=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    ax3.set_xticks(np.arange(1, R.shape[0], 2)+0.25, minor=False)
    ax3.set_xticklabels(odd_labels, rotation=90, minor=False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["bottom"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_visible(False)

    for tic in ax3.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False

    ax.axis('tight')
    ax2.axis('tight')
    ax3.axis('tight')

def _clean_axes(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.
    ax.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    if grid:
        ax.yaxis.grid(True, lw=0.5, color="black", alpha=0.3, linestyle="--")

def plot_score(kf_scores, labels, plot_label, ax=None, color=None, label=None, xlim=None, ylabel=None, xlabel=None):
    if ax is None:
        ax = plt.gca()

    if color is None:
        color = 'blue'

    idx = np.where(np.array(labels) == plot_label)[0][0]

    mean = np.mean(kf_scores[:, :, idx], axis=0)
    std = np.std(kf_scores[:, :, idx], axis=0)

    _clean_axes(ax, grid=False)

    if xlim:
        ax.set_xlim(xlim)

    ax.fill_between(np.arange(len(mean)), mean - std, mean + std, color=colorscheme['light_' + color], alpha=0.3)
    ax.plot(mean - std, '--', color=colorscheme['light_' + color], lw=1.5)
    ax.plot(mean + std, '--', color=colorscheme['light_' + color], lw=1.5)
    ax.plot(mean, color=colorscheme[color], lw=2.5, label=label)

    if ylabel:
        ax.set_ylabel(ylabel)

    if xlabel:
        ax.set_xlabel(xlabel)

    for i, v in enumerate(ax.yaxis.get_ticklocs()):
        if i % 2 == 1:
            ax.plot([v] * kf_scores.shape[1], lw=0.5, color="black", alpha=0.3, linestyle="--")

def plot_r2(r2, labels, plot_label, ax=None, label=None, xlim=None, color=None, ylim=None, ylabel=None, xlabel=None):
    if ax is None:
        ax = plt.gca()

    if color is None:
        color = 'blue'

    idx = np.where(np.array(labels) == plot_label)[0][0]
    r2 = r2[:, idx]

    if xlim:
        ax.set_xlim(xlim)

    if ylim:
        ax.set_ylim(ylim)

    if ylabel:
        ax.set_ylabel(ylabel)

    if xlabel:
        ax.set_xlabel(xlabel)

    ax.plot(r2, '.:', color=colorscheme[color], label=label)
    _clean_axes(ax, grid=False)

    for i, v in enumerate(ax.yaxis.get_ticklocs()):
        if i % 2 == 1:
            ax.plot([v] * len(r2), lw=0.5, color="black", alpha=0.3, linestyle="--")

def plot_bn(bns, labels, plot_label, ax=None):
    if ax is None:
        ax = plt.gca()

    idx = np.where(np.array(labels) == plot_label)[0][0]

    G = nx.DiGraph()
    for (p, q) in bns[idx][2].keys():
        G.add_edge(transform_label(p), transform_label(q))

    nx.draw_circular(G, node_color='w', with_labels=True, alpha=0.5,
            font_color=colorscheme['red'], ax=ax, node_size=0, font_size=16, node_shape="s")

def graph(G, layout, ax=None):
    if ax is None:
        ax = plt.gca()

    nx.draw(G, layout, node_color='w', with_labels=True, alpha=0.5,
            font_color='b', ax=ax)

def barplot(names, dpoints, steps, limit=None, ax=None):
    if ax is None:
        ax = plt.gca()

    space = 0.5
    algos = [a for a, e in dpoints]
    errors = [e for a, e, in dpoints]

    n = len(steps)
    width = (1 - space) / n

    print(errors[0][0])
    print(errors[1][0])
    print(errors[1][0] - errors[0][0])
    indeces = np.arange(1, len(names) + 1)
    for i, step in enumerate(steps):
        pos = [j - (1 - space) /2. + i * width for j in range(1, len(names)+1)]
        gmrf_vals = errors[0][step - 1, :]
        hrf_vals = errors[1][step - 1, :]

        ax.bar(pos, gmrf_vals, width=width, color=colorscheme['blue'])
        ax.bar(pos, hrf_vals, width=width, bottom=gmrf_vals, color=colorscheme['orange'])

    _clean_axes(ax)
    ax.set_xticks(indeces)
    ax.set_xticklabels([transform_label(l) for l in names], rotation=90, minor=False)

def hrf_model(labels, edges, interleave=False, add_color=False, ax=None):
    if ax is None:
        ax = plt.gca()

    R = np.zeros((np.size(labels), np.size(labels)))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if (a, b) in edges:
                R[i, j] = 1

    if interleave:
        mid = int(R.shape[0] / 2)
        new_indices = []

        for i in range(mid):
            new_indices.append(i)
            new_indices.append(i + mid)

        R = (R[new_indices, :])[:, new_indices]
        labels = labels[new_indices]

    if add_color:
        clusters = ['^(l1_)?ahu', '^(l1_)?e', '^(l1_)?h', '^(l1_)?k', '^(l1_)?n', '^(l1_)?q']
        values = np.arange(2, len(clusters) + 2)
        cs = ['white', 'black', colorscheme['blue'], colorscheme['purple'], colorscheme['red'], colorscheme['cyan'], colorscheme['orange'], colorscheme['green']]
        for (i, j), v in np.ndenumerate(R):
            if v != 0:
                for k, c in enumerate(clusters):
                    if re.search(c, labels[i]) and re.search(c, labels[j]):
                        R[i, j] = values[k]
                        break

        cmap = colors.ListedColormap(cs)
        bounds = np.arange(len(clusters) + 3)
        norm = colors.BoundaryNorm(bounds, cmap.N)
    else:
        cmap='Greys'
        norm=None

    labels = [transform_label(label) for label in labels]

    ax2 = ax.twinx()
    ax3 = ax.twiny()

    ax.pcolor(R, cmap=cmap, norm=norm, edgecolor='white')
    ax2.pcolor(R, cmap=cmap, norm=norm, edgecolor='white')
    ax3.pcolor(R, cmap=cmap, norm=norm, edgecolor='white')

    even_labels = [l for i, l in enumerate(labels) if i % 2 == 0]
    odd_labels = [l for i, l in  enumerate(labels) if i % 2 == 1]

    ax.set_xticks(np.arange(0, R.shape[0], 2)+0.25, minor=False)
    ax.set_yticks(np.arange(0, R.shape[1], 2)+0.25, minor=False)
    ax.set_xticklabels(even_labels, rotation=-90, minor=False)
    ax.set_yticklabels(even_labels, minor=False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax2.set_yticks(np.arange(1, R.shape[1], 2)+0.25, minor=False)
    ax2.set_yticklabels(odd_labels, minor=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    ax3.set_xticks(np.arange(1, R.shape[0], 2)+0.25, minor=False)
    ax3.set_xticklabels(odd_labels, rotation=90, minor=False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["bottom"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_visible(False)

    for tic in ax3.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False

    ax.axis('tight')
    ax2.axis('tight')
    ax3.axis('tight')
