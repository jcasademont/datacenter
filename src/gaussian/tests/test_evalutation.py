import unittest
import numpy as np
from ..gmrf import GMRF
from numpy.testing import assert_allclose

class TestEvaluation(unittest.TestCase):

    def test_predict(self):
        """ Test predict function """
        X = np.array([[1, 7]])

        Q = np.array([[1, 2], [2, 3]])
        mu = np.array([0, 0])

        gmrf = GMRF(variables_names=['a', 'b'])
        gmrf.precision_ = Q
        gmrf.mean_ = mu

        pred = gmrf.predict(X, ['b'])
        assert_allclose(pred, np.array([[-2/3]]))

    def test_predict2(self):
        """ Test score function 2 """
        X = np.array([[1, 7, 5], [2, 1, 0]])

        Q = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 7]])
        mu = np.array([1, 2, 3])

        gmrf = GMRF(variables_names=['a', 'b', 'c'])
        gmrf.precision_ = Q
        gmrf.mean_ = mu

        preds = gmrf.predict(X, ['b'])
        assert_allclose(preds, np.array([[-1/2], [21/4]]))
