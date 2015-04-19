import unittest
import numpy as np
import transformations as tr
from numpy.testing import assert_allclose

class TestCDF_Estimator(unittest.TestCase):

    def test_cdf(self):
        """ Test CDF estimation """
        array = np.array([2, 1, 1, 3, 2, 5, 1])
        est = tr.CDF_Estimator(array)
        assert_allclose(est.cdf, np.array([3/7, 5/7, 6/7, 7/7]))

    def test_mean(self):
        """ Test Mean """
        array = np.array([2, 1, 1, 3, 2, 5, 1])
        est = tr.CDF_Estimator(array)
        self.assertEqual(est.mean, 15/7)
