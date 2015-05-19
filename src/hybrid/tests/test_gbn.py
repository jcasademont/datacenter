import unittest
import numpy as np
from numpy.testing import assert_allclose

from ..gbn import GBN

class TestGBN(unittest.TestCase):

    def test_log_conditional_prob(self):
        """ Test log cond. proba for a -> b """
        x = [2, 7]
        gbn = GBN(['a', 'b'])

        gbn.nodes = {'a': (5, .5), 'b': (7, 1.)}
        gbn.edges = {('a', 'b'): 3}

        lcp = gbn.log_conditional_prob(x, 'a', [])
        self.assertAlmostEquals(lcp, -18.2257913526)
        lcp = gbn.log_conditional_prob(x, 'b', ['a'])
        self.assertAlmostEquals(lcp, -18.9189385332)

    def test_mdl(self):
        """ Test mdl for a -> b """
        X = np.array([[1, 2], [7, 15]])
        gbn = GBN(['a', 'b'])

        gbn.nodes = {'a': (5, .5), 'b': (7, 1.)}
        gbn.edges = {('a', 'b'): 3}

        mdl = gbn.mdl(X)
        self.assertAlmostEquals(mdl, -159.829180543)

    def test_neighbours(self):
        """ Test get_neighbours function """
        gbn = GBN(['a', 'b', 'c'], edges=[('a', 'b'), ('b', 'c')])
        neighbours = gbn.get_neighbours()
        neighbours = [set(gbn.edges.keys()) for gbn in neighbours]
        neighbours_expected = \
                          [set([('a', 'b')]),
                           set([('b', 'c')]),
                           set([('a', 'b'), ('c', 'b')]),
                           set([('b', 'a'), ('b', 'c')]),
                           set([('a', 'b'), ('b', 'c'), ('a', 'c')])]

        for neighbour in neighbours:
            self.assertIn(neighbour, neighbours_expected)
        self.assertEquals(len(neighbours), len(neighbours_expected))

    def test_fit_chain(self):
        """ Test fit params for chain network """
        gbn = GBN(['a', 'b', 'c'], edges=[('a', 'b'), ('b', 'c')])
        X = np.array([[1, 2, 9], [7, 14, 21]])
        gbn.fit_params(X)

        self.assertEquals(gbn.nodes['a'][0], 4)
        self.assertEquals(gbn.nodes['b'][0], 0.0)
        self.assertEquals(gbn.edges[('a', 'b')], 2.0)
        self.assertEquals(gbn.nodes['c'][0], 7)
        self.assertEquals(gbn.edges[('b', 'c')], 1.0)

        self.assertEquals(gbn.nodes['a'][1], 3)
        self.assertEquals(gbn.nodes['b'][1], 0.)
        self.assertEquals(gbn.nodes['c'][1], 0.)

    def test_fit_param(self):
        """ Test fit params for network with two parents """
        gbn = GBN(['a', 'b', 'c'], edges=[('a', 'b'), ('c', 'b')])
        X = np.array([[1, 8, 2], [7, 42, 10], [5, 18, 1]])
        gbn.fit_params(X)

        self.assertAlmostEquals(gbn.nodes['a'][0], 13/3)
        self.assertAlmostEquals(gbn.nodes['b'][0], 1.0)
        self.assertAlmostEquals(gbn.edges[('a', 'b')], 3.0)
        self.assertAlmostEquals(gbn.nodes['c'][0], 13/3)
        self.assertAlmostEquals(gbn.edges[('c', 'b')], 2.0)

    def test_to_multivaraite_gaussian(self):
        """ Test tranformation from GBN chain to multivariate gaussian """
        gbn = GBN(['a', 'b', 'c'])
        gbn.nodes = {'a': (1, 4), 'b': (-5, 4), 'c': (4, 3)}
        gbn.edges = {('a', 'b'): 0.5, ('b', 'c'): -1}

        gbn.compute_mean_cov_matrix()

        assert_allclose(np.array(gbn.mu), np.array([1, -4.5, 8.5]))
        assert_allclose(np.array(gbn.cov), np.array([[4, 2, -2],
                                                     [2, 5, -5],
                                                     [-2, -5, 8]]))

    def test_fit_params_chain_more_data(self):
        """ Test fit params on chain network on 1000 points """
        gbn = GBN(['a', 'b', 'c'], edges=[('a', 'b'), ('b', 'c')])
        a = np.random.normal(5, 2, 1000)
        b = a * 3 + 9 + np.random.normal(0, 0.2, 1000)
        c = (-5) * b + 11 + np.random.normal(0, 0.5, 1000)
        X = np.array([a, b, c]).T

        gbn.fit_params(X)
        self.assertAlmostEquals(gbn.nodes['a'][0], 5, delta=0.2)
        self.assertAlmostEquals(gbn.nodes['b'][0], 9, delta=0.2)
        self.assertAlmostEquals(gbn.edges[('a', 'b')], 3.0, delta=0.2)
        self.assertAlmostEquals(gbn.nodes['c'][0], 11, delta=0.2)
        self.assertAlmostEquals(gbn.edges[('b', 'c')], -5, delta=0.2)

    def test_fit_params_net_more_data(self):
        """ Test fit params on network on 1000 points """
        gbn = GBN(['a', 'b', 'c', 'd'], edges=[('a', 'b'), ('c', 'b'), ('b', 'd')])
        a = np.random.normal(5, 2, 1000)
        c = np.random.normal(3, 0.5, 1000)
        b = 5 * c + a * 3 + 9 + np.random.normal(0, 0.2, 1000)
        d = b * 2 + 1 + np.random.normal(0, 0.1, 1000)
        X = np.array([a, b, c, d]).T

        gbn.fit_params(X)
        self.assertAlmostEquals(gbn.nodes['a'][0], 5, delta=0.5)
        self.assertAlmostEquals(gbn.nodes['b'][0], 9, delta=0.5)
        self.assertAlmostEquals(gbn.edges[('a', 'b')], 3.0, delta=0.5)
        self.assertAlmostEquals(gbn.edges[('c', 'b')], 5.0, delta=0.5)
        self.assertAlmostEquals(gbn.nodes['c'][0], 3, delta=0.5)
        self.assertAlmostEquals(gbn.nodes['d'][0], 1, delta=0.5)
        self.assertAlmostEquals(gbn.edges[('b', 'd')], 2, delta=0.5)

    # def test_fit_chain(self):
    #     """ Test fit on chain network """
    #     gbn = GBN(['a', 'b', 'c'])
    #     a = np.random.normal(5, 1, 1000)
    #     b = a * 3 + 9 + np.random.normal(0, 0.5, 1000)
    #     c = (-5) * b + 4 + np.random.normal(0, 0.1, 1000)
    #     X = np.array([a, b, c]).T

    #     gbn.fit(X)
    #     print(gbn.edges)
    #     self.assertAlmostEquals(gbn.nodes['a'][0], 5, delta=0.5)
    #     self.assertAlmostEquals(gbn.nodes['b'][0], 9, delta=0.5)
    #     self.assertAlmostEquals(gbn.edges[('a', 'b')], 3.0, delta=0.5)
    #     self.assertAlmostEquals(gbn.nodes['c'][0], 4, delta=0.5)
    #     self.assertAlmostEquals(gbn.edges[('b', 'c')], -5, delta=0.5)

    # def test_fit_net(self):
    #     """ Test fit on larger network """
    #     gbn = GBN(['a', 'b', 'c', 'd'])
    #     a = np.random.normal(5, 1, 1000)
    #     c = np.random.normal(2, 1, 1000)
    #     b = 8 * c + a * 3 + 9 + np.random.normal(0, 0.5, 1000)
    #     d = -1 * b + 6 + np.random.normal(0, 0.3, 1000)
    #     X = np.array([a, b, c, d]).T

    #     gbn.fit(X)
    #     print(gbn.edges)
    #     self.assertAlmostEquals(gbn.nodes['a'][0], 5, delta=0.5)
    #     self.assertAlmostEquals(gbn.nodes['b'][0], 9, delta=0.5)
    #     self.assertAlmostEquals(gbn.edges[('a', 'b')], 3.0, delta=0.5)
    #     self.assertAlmostEquals(gbn.edges[('c', 'b')], 8.0, delta=0.5)
    #     self.assertAlmostEquals(gbn.nodes['c'][0], 2, delta=0.5)
    #     self.assertAlmostEquals(gbn.nodes['d'][0], 6, delta=0.5)
    #     self.assertAlmostEquals(gbn.edges[('b', 'd')], -1, delta=0.5)
