import unittest
import numpy as np
from ..gmrf import GMRF
from numpy.linalg import inv
from scipy.stats import norm, multivariate_normal

class TestLogPdf(unittest.TestCase):

    def test_one_dim(self):
        """ Test log pdf for one dimensional data """
        x = np.array([1])
        mean = np.array([2])
        Q  = np.array([[ 1 / 25 ]])

        gmrf = GMRF()
        self.assertAlmostEqual(gmrf._logpdf(x, mean, Q),
                               norm.logpdf(1, 2, 5))

    def test_mulit_dim(self):
        """ Test log pdf for multi dimensional data """
        x = np.array([1, 2, 1.7])
        mean = np.array([2, 1, 5])
        Q  = np.array([[1.2, 0.7, -0.4],
                       [0.7, 0.68, 0.01],
                       [-0.4, 0.01, 1]])

        gmrf = GMRF()
        self.assertAlmostEqual(gmrf._logpdf(x, mean, Q),
                               multivariate_normal.logpdf(x, mean, inv(Q)))

class TestBic(unittest.TestCase):

    def test_simple_chain(self):
        """ Test BIC for simple chain A - B - C"""
        Q = np.array([[1, -0.5, 0], [-0.5, 1.25, -0.5], [0, -0.5, 1]])

        X = np.array([[-0.5, -1.5, 0.4],
                      [3.9, -1.7, -1.1],
                      [7.8, -3.2, 1.3],
                      [2.0, -2.9, 3.2],
                      [3.4, -8, 1.3]])

        mean = np.mean(X, axis=0)

        gmrf = GMRF()
        gmrf.precision_ = Q
        bic, converged = gmrf.bic(X, gamma=0)

        self.assertTrue(converged)
        self.assertAlmostEqual(bic, -2 * np.sum(multivariate_normal.logpdf(X, mean, inv(Q))) + 5 * np.log(5))
