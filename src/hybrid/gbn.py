import copy
import itertools
import numpy as np

from numpy.linalg import inv
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

        self.mu = None
        self.cov = None

    def _ancestors(self, name):
        ancestors = []
        candidates = self._parents(name)
        while len(candidates) > 0:
            c = candidates.pop()
            ancestors.append(c)
            candidates.extend(self._parents(c))

        return ancestors

    def _parents(self, name):
        parents_set = []
        for (p, c) in self.edges:
            if c == name:
                parents_set.append(self.indices[p])

        return parents_set

    def _children(self, name):
        children_set = []
        for (p, c) in self.edges:
            if p == name:
                children_set.append(self.indices[c])

        return children_set

    def markov_blanket(self, name):
        if name not in self.variables_names:
            return []

        idx = self.indices[name]
        children = self._children(name)
        parents = self._parents(name)
        children_parents = [self._parents(c) for c in children]

        mb = children + parents
        for cps in children_parents:
            for cp in cps:
                if cp != idx:
                    mb.append(cp)

        return self.variables_names[mb]

    def log_conditional_prob(self, x, name, parents_names):
        v = x[self.indices[name]]

        mean = self.nodes[name][0]
        parents_coefs = [self.edges[(p, name)] for p in parents_names]
        parents_values = [x[self.indices[p]] for p in parents_names]
        mean += np.sum([b * u for b, u in zip(parents_coefs, parents_values)])

        std = self.nodes[name][1]

        if std != 0:
            lp = norm.logpdf(v, loc=mean, scale=std)
            return lp
        else:
            return 0

    def log_likelihood(self, X):
        ll = 0
        for x in X:
            for n in self.variables_names:
                ll += self.log_conditional_prob(x, n,
                        self.variables_names[self._parents(n)])
        return ll

    def mdl(self, X):
        nb_params = len(self.nodes) + len(self.edges)

        return self.log_likelihood(X) - nb_params / 2 * np.log(X.shape[0])

    def create_loop(self, edge):
        new_parent, new_child = edge

        new_child_idx = self.indices[new_child]
        new_parent_idx = self.indices[new_parent]
        candidate = set([new_parent_idx])
        while len(candidate) != 0:
            current = candidate.pop()
            current_parents = self._parents(self.variables_names[current])
            if new_child_idx in current_parents:
                return True
            else:
                candidate.update(current_parents)

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
            if not gbn_reverse_edge.create_loop((edge[1], edge[0])):
                neighbours.append(gbn_reverse_edge)

        for a, b in itertools.permutations(self.nodes.keys(), r=2):
            new_edges = copy.copy(list_edges)
            if (a, b) not in list_edges:
                new_edges.append((a, b))
                newGBN = GBN(self.variables_names, edges=new_edges)
                if not newGBN.create_loop((a, b)):
                    neighbours.append(newGBN)

        return neighbours

    def compute_mean_cov_matrix(self):
        self.mu = np.zeros(len(self.variables_names))
        sorted_by_ancestors = [i[0] for i in sorted(enumerate(self.variables_names), key=lambda x: len(self._ancestors(x[1])))]

        for v in sorted_by_ancestors:
            self.mu[v] = self.nodes[self.variables_names[v]][0] \
                  + np.sum([self.edges[(self.variables_names[p], self.variables_names[v])] * self.mu[p] for p in self._parents(self.variables_names[v])])

        self.cov = np.zeros((len(self.variables_names), len(self.variables_names)))

        I = np.eye(len(sorted_by_ancestors))
        for i in sorted_by_ancestors:
            for j in range(len(sorted_by_ancestors)):
                self.cov[i, j] = np.sum([self.edges[(self.variables_names[k], self.variables_names[i])] * self.cov[j, k] for k in self._parents(self.variables_names[i])]) + I[i, j] * self.nodes[self.variables_names[i]][1]
                self.cov[j, i] = self.cov[i, j]

    def fit_params(self, X):
        S = np.cov(X.T, bias=1)

        for node in list(self.nodes.keys()):
            idx_node = self.indices[node]
            parents = self._parents(node)

            a = np.zeros((len(parents) + 1, len(parents) + 1))
            a[0, 0] = 1
            a[0, 1:] = np.mean(X[:, parents], axis=0)
            for i, u in enumerate(parents):
                a[i + 1, 0] = np.mean(X[:, u], axis=0)
                a[i + 1, 1:] = [np.mean(np.multiply(X[:, u], X[:, j])) for j in parents]

            b = np.empty(len(parents) + 1)
            b[0] = np.mean(X[:, idx_node])
            b[1:] = [np.mean(np.multiply(X[:, idx_node], X[:, j])) for j in parents]

            betas = np.linalg.solve(a, b)

            var = S[idx_node, idx_node]
            for i, u in enumerate(parents):
                for j, v in enumerate(parents):
                    var -= betas[i + 1] * betas[j + 1] * S[u, v]

            std = 0
            if var > 0:
                std = np.sqrt(var)

            self.nodes[node] = (betas[0], std)

            for i, p in enumerate(parents):
                self.edges[(self.variables_names[p], node)] = betas[i + 1]

    def fit(self, X):
        self.edges = {}
        while True:
            s = self.mdl(X)
            s_prime = -np.inf
            neighbours = self.get_neighbours()
            scores = []
            for nei in neighbours:
                nei.fit_params(X)
                scores.append(nei.mdl(X))

            if len(scores) > 0:
                s_prime = np.max(scores)

            if s_prime <= s:
                break;

            idx = np.where(np.array(scores) == s_prime)[0]
            i = np.random.choice(idx)
            best_neighbour = neighbours[i]
            self.edges = best_neighbour.edges
            self.nodes = best_neighbour.nodes

    def proba(self, name, data, given):
        if self.mu == None or self.cov == None:
            self.compute_mean_cov_matrix()

        ll = 0
        Q = inv(self.cov)
        idx = self.indices[name]
        evidence_indices = np.array([self.indices[n] for n in given if n in self.variables_names], dtype=int)

        indices = list(filter(lambda x: x not in evidence_indices,
                                np.arange(Q.shape[0])))

        new_indices = np.array(np.append(indices, evidence_indices), dtype=int)

        pos = np.where(np.array(indices) == idx)[0][0]
        _Q = (Q[new_indices, :])[:, new_indices]

        lim_a = np.size(indices)
        Qaa = _Q[:lim_a, :lim_a]
        Qab = _Q[:lim_a, lim_a:]

        iQaa = inv(Qaa)

        mean_a = self.mu[indices]
        mean_b = self.mu[evidence_indices]

        std = np.sqrt(iQaa[pos, pos])

        for x in data:
            mean = mean_a - (np.dot(iQaa,
                    np.dot(Qab, (x[evidence_indices] - mean_b).T))).reshape(mean_a.shape)
            ll += norm.logpdf(x[idx], mean[pos], std)

        return ll

    def predict(self, names, evidences):
        if self.mu == None or self.cov == None:
            self.compute_mean_cov_matrix()

        Q = inv(self.cov)

        evidence_indices = [np.where(self.variables_names == e)[0][0] for e in evidences.keys() if e in self.variables_names]

        indices = list(filter(lambda x: x not in evidence_indices, np.arange(Q.shape[0])))

        result_indices = [self.indices[n] for n in names]

        new_indices = np.append(indices, evidence_indices)

        _Q = (Q[new_indices, :])[:, new_indices]

        lim_a = np.size(indices)
        Qaa = _Q[:lim_a, :lim_a]
        Qab = _Q[:lim_a, lim_a:]

        iQaa = inv(Qaa)

        mean_a = self.mu[indices]
        mean_b = self.mu[evidence_indices]

        evidences_values = np.array([evidences[n] for n in self.variables_names[evidence_indices]])
        pred = mean_a - np.dot(iQaa, np.dot(Qab, evidences_values - mean_b))

        result = np.zeros(len(new_indices))
        result[evidence_indices] = evidences_values
        result[indices] = pred
        return result[result_indices]

    def variance(self, name):
        return self.nodes[name][1] ** 2
