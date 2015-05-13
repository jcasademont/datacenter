import copy
import itertools
import numpy as np

from scipy.stats import norm

class GBN():
    def __init__(self, variables_names, nodes=None, edges=dict()):
        if nodes:
            self.nodes = nodes
        else:
            self.nodes = dict()
            for vn in variables_names:
                self.nodes[vn] = (0, 1)

        if isinstance(edges, dict):
            self.edges = edges
        else:
            self.edges = dict()
            for edge in edges:
                self.edges[edge] = 0

        self.variables_names = np.array(variables_names)

        self.indices = dict()
        for i, n in enumerate(self.variables_names):
            self.indices[n] = i

    def _parents(self, name):
        parents_set = []
        for (p, c) in self.edges:
            if c == name:
                parents_set.append(p)

        return parents_set

    def _children(self, name):
        children_set = []
        for (p, c) in self.edges:
            if p == name:
                children_set.append(c)

        return children_set

    def log_conditional_prob(self, x, name, parents_names):
        v = x[self.indices[name]]

        mean = self.nodes[name][0]
        parents_coefs = [self.edges[(p, name)] for p in parents_names]
        parents_values = [x[self.indices[p]] for p in parents_names]
        mean += np.sum([b * u for b, u in zip(parents_coefs, parents_values)])

        std = self.nodes[name][1]

        return norm.logpdf(v, loc=mean, scale=std)

    def log_likelihood(self, X):
        ll = 0
        for x in X:
            for n in self.variables_names:
                ll += self.log_conditional_prob(x, n, self._parents(n))

        return ll

    def mdl(self, X):
        nb_params = len(self.nodes) + len(self.edges)

        return self.log_likelihood(X) - nb_params / 2 * np.log(X.shape[0])

    def create_loop(self, edge):
        new_parent, new_child = edge

        ancesters = set()
        candidate = [new_parent]
        while len(candidate) != 0:
            current = candidate.pop()
            current_parents = self._parents(current)
            if new_child in current_parents:
                return True
            else:
                candidate.extend(current_parents)

        return False

    def get_neighbours(self):
        neighbours = []

        list_edges = list(self.edges.keys())

        for edge in list_edges:
            less_edges = copy.copy(list_edges)
            less_edges.remove(edge)
            gbn_minus_edge = GBN(self.variables_names, edges=less_edges)
            neighbours.append(gbn_minus_edge)

            reverse_edges = copy.copy(list_edges)
            reverse_edges.remove(edge)
            reverse_edges.append((edge[1], edge[0]))
            gbn_reverse_edge = GBN(self.variables_names, edges=reverse_edges)
            neighbours.append(gbn_reverse_edge)

        for a, b in itertools.permutations(self.nodes.keys(), r=2):
            new_edges = copy.copy(list_edges)
            if not (self.create_loop((a, b)) or (a, b) in list_edges):
                new_edges.append((a, b))
                newGBN = GBN(self.variables_names, edges=new_edges)
                neighbours.append(newGBN)

        return neighbours

    def fit(self, X):
        self.edges = {}
        while True:
            s = self.mdl(X)
            neighbours = self.get_neighbours()
            scores = []
            for nei in neighbours:
                nei.fit_params()
                scores.append(nei.mdl(X))

            s_prime = np.max(scores)

            if s_prime <= s:
                break;

            best_neighbour = neighbours[np.argmax(scores)]
            self.edges = best_neighbour.edges
            self.nodes = best_neighbour.nodes
