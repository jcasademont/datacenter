import unittest
import numpy as np

from numpy.linalg import inv
from numpy.testing import assert_allclose

from ..gmrf import GMRF

class TestSampling(unittest.TestCase):

    def test_sample(self):
        """Test GMRF sample function"""
        gmrf = GMRF()
        gmrf.precision_ = np.array([[3.0, 0.3, 0.0],
                                    [0.3, 2.0, 1.0],
                                    [0.0, 1.0, 1.8]])
        gmrf.mean_ = np.array([0, 0, 0])
        S = gmrf.sample(10000)
        assert_allclose(np.mean(S, axis=0), gmrf.mean_, atol=0.1)
        assert_allclose(inv(np.cov(S.T)), gmrf.precision_, atol=0.1)
