import unittest
import numpy as np

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
