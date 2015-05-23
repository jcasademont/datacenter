import unittest
import numpy as np
from numpy.testing import assert_allclose

from ..gbn import GBN

class TestGBN(unittest.TestCase):

    def setUp(self):
        self.chain = GBN(['a', 'b', 'c'], edges=[('a', 'b'), ('b', 'c')])

    def test_log_conditional_prob(self):
        """ Test log cond. proba for a -> b """
        x = np.array([[2, 7]])
        gbn = GBN(['a', 'b'])

        gbn.nodes = {'a': (5, .5), 'b': (7, 1.)}
        gbn.edges = {('a', 'b'): 3}

        lcp = gbn.log_conditional_prob(x, 'a', [])
        self.assertAlmostEqual(lcp, -18.2257913526)
        lcp = gbn.log_conditional_prob(x, 'b', ['a'])
        self.assertAlmostEqual(lcp, -18.9189385332)

    def test_mdl(self):
        """ Test mdl for a -> b """
        X = np.array([[1, 2], [7, 15]])
        gbn = GBN(['a', 'b'])

        gbn.nodes = {'a': (5, .5), 'b': (7, 1.)}
        gbn.edges = {('a', 'b'): 3}

        mdl = gbn.mdl(X)
        self.assertAlmostEqual(mdl, -159.829180543)

    def test_neighbours(self):
        """ Test get_neighbours function """
        neighbours = self.chain.get_neighbours()
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
        gbn = self.chain
        X = np.array([[1, 2, 9], [7, 14, 21]])
        gbn.fit_params(X)

        self.assertEqual(gbn.nodes['a'][0], 4)
        self.assertEqual(gbn.nodes['b'][0], 0.0)
        self.assertEqual(gbn.edges[('a', 'b')], 2.0)
        self.assertEqual(gbn.nodes['c'][0], 7)
        self.assertEqual(gbn.edges[('b', 'c')], 1.0)

        self.assertEqual(gbn.nodes['a'][1], 3)
        self.assertEqual(gbn.nodes['b'][1], 0.)
        self.assertEqual(gbn.nodes['c'][1], 0.)

    def test_fit_param(self):
        """ Test fit params for network with two parents """
        gbn = GBN(['a', 'b', 'c'], edges=[('a', 'b'), ('c', 'b')])
        X = np.array([[1, 8, 2], [7, 42, 10], [5, 18, 1]])
        gbn.fit_params(X)

        self.assertAlmostEqual(gbn.nodes['a'][0], 13/3)
        self.assertAlmostEqual(gbn.nodes['b'][0], 1.0)
        self.assertAlmostEqual(gbn.edges[('a', 'b')], 3.0)
        self.assertAlmostEqual(gbn.nodes['c'][0], 13/3)
        self.assertAlmostEqual(gbn.edges[('c', 'b')], 2.0)

    def test_ancestors(self):
        """ Test _ancestors function """
        gbn = self.chain
        self.assertEqual(gbn._ancestors('a'), [])
        self.assertEqual(gbn._ancestors('b'), [0])
        self.assertEqual(gbn._ancestors('c'), [1, 0])

    def test_to_multivaraite_gaussian(self):
        """ Test tranformation from GBN chain to multivariate gaussian """
        gbn = GBN(['b', 'a', 'c'])
        gbn.nodes = {'a': (1, 2), 'b': (-5, 2), 'c': (4, np.sqrt(3))}
        gbn.edges = {('a', 'b'): 0.5, ('b', 'c'): -1}

        gbn.compute_mean_cov_matrix()

        assert_allclose(np.array(gbn.mean_), np.array([-4.5, 1, 8.5]))
        assert_allclose(np.array(gbn.covariance_),
                        np.array([[5, 2, -5],
                                  [2, 4, -2],
                                  [-5, -2, 8]]))

    def test_fit_params_chain_more_data(self):
        """ Test fit params on chain network on 1000 points """
        gbn = self.chain
        a = np.random.normal(5, 2, 1000)
        b = a * 3 + 9 + np.random.normal(0, 0.2, 1000)
        c = (-5) * b + 11 + np.random.normal(0, 0.5, 1000)
        X = np.array([a, b, c]).T

        gbn.fit_params(X)
        self.assertAlmostEqual(gbn.nodes['a'][0], 5, delta=0.2)
        self.assertAlmostEqual(gbn.nodes['b'][0], 9, delta=0.2)
        self.assertAlmostEqual(gbn.edges[('a', 'b')], 3.0, delta=0.2)
        self.assertAlmostEqual(gbn.nodes['c'][0], 11, delta=0.2)
        self.assertAlmostEqual(gbn.edges[('b', 'c')], -5, delta=0.2)

    def test_fit_params_net_more_data(self):
        """ Test fit params on network on 1000 points """
        gbn = GBN(['a', 'b', 'c', 'd'], edges=[('a', 'b'), ('c', 'b'), ('b', 'd')])
        a = np.random.normal(5, 2, 1000)
        c = np.random.normal(3, 0.5, 1000)
        b = 5 * c + a * 3 + 9 + np.random.normal(0, 0.2, 1000)
        d = b * 2 + 1 + np.random.normal(0, 0.1, 1000)
        X = np.array([a, b, c, d]).T

        gbn.fit_params(X)
        self.assertAlmostEqual(gbn.nodes['a'][0], 5, delta=0.5)
        self.assertAlmostEqual(gbn.nodes['b'][0], 9, delta=0.5)
        self.assertAlmostEqual(gbn.edges[('a', 'b')], 3.0, delta=0.5)
        self.assertAlmostEqual(gbn.edges[('c', 'b')], 5.0, delta=0.5)
        self.assertAlmostEqual(gbn.nodes['c'][0], 3, delta=0.5)
        self.assertAlmostEqual(gbn.nodes['d'][0], 1, delta=0.5)
        self.assertAlmostEqual(gbn.edges[('b', 'd')], 2, delta=0.5)

    def test_prediction_on_chain(self):
        """ Test predict on chain """
        gbn = self.chain
        gbn.nodes = {'a': (5, 1), 'b': (2, 1), 'c': (-1, 1)}
        gbn.edges = {('a', 'b'): 3, ('b', 'c'): 7}
        gbn.compute_mean_cov_matrix()

        X = np.array([[8, 0, 0]])

        pred = gbn.predict(X, ['b', 'c'])
        self.assertAlmostEqual(pred[0][0], 26.)

    def test_prediction_mb(self):
        """ Test that prediction given MB is same that given all nodes """
        gbn = GBN(['a', 'b', 'c', 'd', 'e'])
        gbn.nodes = {'a': (2, 1), 'b': (1, 3), 'c': (5, 0.5), 'd': (3, 1), 'e': (2, 3)}
        gbn.edges = {('a', 'b'): 3, ('c', 'b'): 1, ('b', 'd'): 8, ('d', 'e'): 6}
        gbn.compute_mean_cov_matrix()

        X = np.array([[1, 0, 6, 5, 7]])

        pred_mb = gbn.predict(X, ['b', 'e'])
        pred_full = gbn.predict(X, ['b'])
        self.assertEqual(pred_mb[0][0], pred_full[0][0])

    def test_mb(self):
        """ Test Markov blanket """
        gbn = GBN(['a', 'b', 'c', 'd', 'e'])
        gbn.nodes = {'a': (2, 1), 'b': (1, 3), 'c': (5, 0.5), 'd': (3, 1), 'e': (2, 3)}
        gbn.edges = {('a', 'b'): 3, ('c', 'b'): 1, ('b', 'd'): 8, ('e', 'd'): 6}

        mb = gbn.markov_blanket('b')
        self.assertEqual(tuple(sorted(mb)), tuple(['a', 'c', 'd', 'e']))

    def test_mb_no_node(self):
        """ Test Markov blanket if node doesn't exist """
        gbn = GBN(['a', 'b', 'c', 'd', 'e'])
        gbn.nodes = {'a': (2, 1), 'b': (1, 3), 'c': (5, 0.5), 'd': (3, 1), 'e': (2, 3)}
        gbn.edges = {('a', 'b'): 3, ('c', 'b'): 1, ('b', 'd'): 8, ('d', 'e'): 6}

        mb = gbn.markov_blanket('f')
        self.assertEqual(tuple(sorted(mb)), tuple([]))

    def test_proba(self):
        """ Test proba function """
        gbn = self.chain
        gbn.nodes = {'a': (5, 1), 'b': (2, 1), 'c': (-1, 1)}
        gbn.edges = {('a', 'b'): 3, ('b', 'c'): 7}
        gbn.compute_mean_cov_matrix()

        X = np.array([[8, 2, 1], [5, 6, 1], [4, 4, 9]])
        proba = gbn.proba('b', X, ['a'])
        proba_parents = gbn.log_conditional_prob(X, 'b', np.array(['a']))
        self.assertAlmostEqual(proba, proba_parents)


