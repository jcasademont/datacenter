import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

racks = pd.read_pickle('data/rack_temps.pickle')
ahu = pd.read_pickle('data/ahu.pickle')

r = ['E9', 'H12', 'E12', 'H9', 'K8', 'K9', 'K11', 'K12', 'N1', 'N2']
a = ['ahu_2_outlet', 'ahu_3_outlet']
v = r + a

df = pd.DataFrame()

for i in range(len(r)):
    df[r[i]] = racks[r[i]]

for i in range(len(a)):
    df[a[i]] = ahu[a[i]]

df = df.dropna()

cov = df.cov()
Q = inv(cov.values)
mu = df.mean(axis=0).values

fig, ax = plt.subplots()
cols = [n for n in v]
plt.pcolor(Q, cmap='seismic', vmin=-60, vmax=60)

for y in range(Q.shape[0]):
        for x in range(Q.shape[1]):
                    plt.text(x + 0.5, y + 0.5, '%.4f' % Q[y, x],
                                              horizontalalignment='center',
                                              verticalalignment='center',
                                              )

plt.colorbar()
ax.set_xticks(np.arange(Q.shape[0])+0.5, minor=False)
ax.set_yticks(np.arange(Q.shape[1])+0.5, minor=False)
ax.set_xticklabels(cols, rotation=-90, minor=False)
ax.set_yticklabels(cols, minor=False)
ax.axis('tight')

Q[abs(Q) < 3] = 0
G = nx.Graph()
for i in range(len(v)):
    G.add_node(v[i])
for (i, j), x in np.ndenumerate(Q):
    if(x != 0 and i != j):
        G.add_edge(v[i], v[j], w=x)

edges, weights = zip(*nx.get_edge_attributes(G, 'w').items())

fig, ax = plt.subplots()
plt.figure(2)
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color='b', node_size=2000, edgelist=edges, edge_color=weights, width=10.0,
        edge_cmap=plt.cm.seismic, with_labels=True)

plt.show()
