import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

def explore(a, b, ax, xlabel, ylabel):
    plt.scatter(a, b)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    mi = metrics.normalized_mutual_info_score(a, b)
    co = stats.pearsonr(a, b)

    text = "MI = {0:.6}".format(mi)
    text = text + ", Cor = {0:.6}".format(co[0])

    plt.text(0.95, 0.01, text,
             verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            fontsize=15)


racks = pd.read_pickle('data/rack_temps.pickle')
ahu = pd.read_pickle('data/ahu.pickle')

r = ['E9', 'H12', 'H6', 'K9', 'N14', 'Q1']

ref = racks.H9
fig = plt.figure(1)
for i in range(6):
    ax = fig.add_subplot(3, 2, i + 1)
    explore(ref, racks[r[i]], ax, 'H9', r[i])

fig = plt.figure(2)
for i in range(4):
    ax = fig.add_subplot(2, 2, i + 1)
    explore(ref[:2843], ahu['ahu_' + str(i + 1) + '_outlet'], ax, 'H9', 'ahu_' + str(i + 1) + '_outlet')

fig = plt.figure(3)
for i in range(12):
    ax = fig.add_subplot(6, 2, i + 1)
    explore(ref[:-(i+1)], ref.shift(-(i+1)).dropna(), ax, "H9", "H9 - " + str(i + 1))

plt.show()
