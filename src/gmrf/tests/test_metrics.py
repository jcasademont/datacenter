import unittest
import numpy as np
import metrics
from numpy.linalg import inv
from scipy.stats import norm, multivariate_normal

class TestLogPdf(unittest.TestCase):

    def test_one_dim(self):
        """ Test log pdf for one dimensional data """
        x = np.array([1])
        mean = np.array([2])
        Q  = np.array([[ 1 / 25 ]])
        self.assertAlmostEqual(metrics.logpdf(x, mean, Q),
                               norm.logpdf(1, 2, 5))

    def test_mulit_dim(self):
        """ Test log pdf for multi dimensional data """
        x = np.array([1, 2, 1.7])
        mean = np.array([2, 1, 5])
        Q  = np.array([[1.2, 0.7, -0.4],
                       [0.7, 0.68, 0.01],
                       [-0.4, 0.01, 1]])

        self.assertAlmostEqual(metrics.logpdf(x, mean, Q),
                               multivariate_normal.logpdf(x, mean, inv(Q)))

class TestBic(unittest.TestCase):

    def test_simple_chain(self):
        """ Test BIC for simple chain A - B - C"""
        Q = np.array([[1, -0.5, 0], [-0.5, 1.25, -0.5], [0, -0.5, 1]])
        mean = np.array([0.7, 3, 0.4])
        X = np.array([[0.2, 1.5, 0.8],
                      [4.6, 1.3, -0.7],
                      [8.5, -0.2, 1.7],
                      [2.7, 0.1, 3.6],
                      [4.1, -5, 1.7]])

        self.assertAlmostEqual(metrics.bic(X, Q, mean), -170.025628)
