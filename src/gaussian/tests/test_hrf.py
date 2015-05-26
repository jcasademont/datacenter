import unittest
import numpy as np
from numpy.testing import assert_allclose

from ..hrf import HRF
from ..gbn import GBN

class TestHRF(unittest.TestCase):

    def setUp(self):
        self.hrf = HRF(k=1, k_star=2, variables_names=['a', 'b', 'c', 'd', 'e'])

        bna = GBN(['a', 'b'])
        bna.nodes = {'a': (3, 1), 'b': (2, 1)}
        bna.edges = {('a', 'b'): -1}

        bnb = GBN(['a', 'b', 'c'])
        bnb.nodes = {'a': (3, 1), 'b': (2, 1), 'c': (4, 1)}
        bnb.edges = {('a', 'b'): -2, ('c', 'b'): 3}
        bnb.compute_mean_cov_matrix()

        bnc = GBN(['c', 'b', 'd'])
        bnc.nodes = {'d': (3, 1), 'b': (2, 1), 'c': (-1, 1)}
        bnc.edges = {('b', 'c'): 1, ('c', 'd'): 7}

        bnd = GBN(['e', 'd'])
        bnd.nodes = {'e': (3, 1), 'd': (2, 1)}
        bnd.edges = {('e', 'd'): -2}
        bnd.compute_mean_cov_matrix()

        bne = GBN(['d', 'e'])
        bne.nodes = {'d': (1, 1), 'e': (-7, 1)}
        bne.edges = {('d', 'e'): -1}

        self.hrf.bns = [bna, bnb, bnc, bnd, bne]

    def test_predict(self):
        """ Test predict on HRF """
        X = np.array([[1, 4, 3, 2, 1], [5, 4, 3, 2, 2]])
        preds = self.hrf.predict(X, ['d', 'b'])

        assert_allclose(preds, [[0, 9], [-2, 1]])
