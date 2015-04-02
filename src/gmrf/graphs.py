import os
import json
import numpy as np
import networkx as nx
import layouts
from networkx.readwrite import json_graph


grid_graph = \
    {'eo6': ['e9', 'h3'], 'e9': ['eo6', 'e12', 'h9'],
     'e12': ['e9', 'e15', 'h12'], 'e15': ['e20', 'e12', 'h15'],
     'e20': ['h25', 'e15', 'h22', 'h19'],
     'h3': ['eo6', 'h9', 'k1', 'k4'], 'h9': ['h3', 'h12', 'k9', 'e9'],
     'h12': ['h9', 'h15', 'e12', 'k12'],
     'h15': ['h12', 'h19', 'e15', 'k15'],
     'h19': ['e20', 'h22', 'k19', 'h15'], 'h22': ['e20', 'h19', 'h25'],
     'h25': ['e20', 'h22'], 'k1': ['h3', 'n2'],
     'k4': ['h3', 'k8', 'n5', 'n2'], 'k8': ['k9', 'n5', 'n10'],
     'k9': ['k8', 'h9', 'n10', 'k11'], 'k11': ['k9', 'k12', 'n10'],
     'k12': ['k11', 'h12', 'n10', 'k15'], 'k15': ['k12', 'h15', 'k19'],
     'k19': ['h19', 'k22', 'k15', 'n19'], 'k22': ['k19', 'k25'],
     'k25': ['k22'],
     'n1': [], 'n2': ['k1', 'k4'], 'n5': ['k4', 'k8', 'n10'],
     'n10': ['k8', 'k9', 'k11', 'k12', 'n5', 'n14'],
     'n14': ['n10', 'n19'], 'n19': ['k19', 'n14', 'n22'],
     'n22': ['n19', 'n25', 'q20', 'q23', 'q25'], 'n25': ['k19', 'n22'],
     'q1': [], 'q4a': [], 'q8': [], 'q11': [], 'q13': [],
     'q20': ['n22'], 'q23': ['n22'], 'q25': ['n22'], 'ahu_1_outlet': ['eo6', 'e9', 'e12', 'e15', 'h3', 'h9', 'h12', 'h15', 'k1', 'k4', 'k8', 'k9', 'k11', 'k12', 'k15'], 'ahu_2_outlet': ['e12', 'e15', 'e20', 'h12', 'h15', 'h19', 'h22', 'h25', 'k12', 'k15', 'k19', 'k22', 'k25'], 'ahu_3_outlet': ['n1', 'n2', 'n5', 'n10', 'n14', 'q1', 'q4a', 'q8', 'q11', 'q13'], 'ahu_4_outlet': ['q11', 'q13', 'q20', 'q23', 'q25'], 'room_it_power_(kw)': ['eo6', 'e9', 'e12', 'e15', 'e20', 'h3', 'h9', 'h12', 'h15', 'h19', 'h22', 'h25', 'k1', 'k4', 'k8', 'k9', 'k11', 'k12', 'k15', 'k19', 'k22', 'k25', 'n1', 'n2', 'n5', 'n10', 'n14', 'n19', 'n22', 'n25', 'q1', 'q4a', 'q8', 'q11', 'q13', 'q20', 'q23', 'q25']}

def _get_group(name):

    if("_" not in name):
        return 1
    if("ahu" in name):
        return 2
    else:
        return 3

def _get_node_attrs(name):
    attrs = dict()
    attrs['name'] = name
    attrs['group'] = _get_group(name)
    attrs['x'] = (layouts.datacenter_layout[name][0] + 5) / layouts.max_x
    attrs['y'] = (layouts.max_y - layouts.datacenter_layout[name][1] - 5) / layouts.max_y
    attrs['fixed'] = True

    return attrs

def _get_edge_attrs(v):
    attrs = dict()
    attrs['value'] = v

    return attrs

def fromQ(Q, labels):
    G = nx.Graph()

    for i in range(len(labels)):
        G.add_node(labels[i], _get_node_attrs(labels[i]))

    for (i, j), x in np.ndenumerate(Q):
        if(x != 0 and i != j):
            G.add_edge(labels[i], labels[j], _get_edge_attrs(x))

    return G

def saveGraph(G, name="graph.json"):
    d = json_graph.node_link_data(G)
    json.dump(d, open(os.path.join(os.path.dirname(__file__), '../../viz/') + name, 'w'))
