import unittest
import numpy as np
from ..gmrf import GMRF
from numpy.testing import assert_allclose

class TestEvaluation(unittest.TestCase):

    def test_score(self):
        """ Test score function """
        X = np.array([[1, 7], [2, 1]])
        indices = np.array([1])
        Q = np.array([[1, 2], [2, 3]])

        gmrf = GMRF()
        gmrf.precision_ = Q
        score = gmrf.score(X, indices)
        self.assertEqual(score, 0.5 * (abs(7 - 13/3) + abs(1 - 11/3)))

    def test_score2(self):
        """ Test score function 2 """
        X = np.array([[1, 7, 5], [2, 1, 0]])
        indices = np.array([1])
        Q = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 7]])

        gmrf = GMRF()
        gmrf.precision_ = Q
        score = gmrf.score(X, indices)
        self.assertEqual(score, 5.875)
