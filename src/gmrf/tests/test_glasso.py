import unittest
import numpy as np
from ..gmrf import GMRF
from numpy.linalg import inv

class TestGlasso(unittest.TestCase):

    def generate_data(Q):
        return np.random.multivariate_normal([0] * Q.shape[0], inv(Q), 3000)


    @classmethod
    def setUpClass(self):
        self.Q_chain = np.array([[3.0, 0.3, 0.0],
                                 [0.3, 2.0, 1.0],
                                 [0.0, 1.0, 1.8]])

        self.Q_loop = np.array([[3.0, 0.3, 1.7, 2.0],
                                [0.3, 2.0, 1.0, 2.3],
                                [1.7, 1.0, 1.8, 0.4],
                                [2.0, 2.3, 0.4, 1.3]])

        self.chain_data = self.generate_data(self.Q_chain)
        self.loop_data = self.generate_data(self.Q_loop)

    def test_simple_chain_cv(self):
        """ Test glasso with simple model A - B - C using CV """
        gmrf = GMRF(method="cv")
        gmrf.fit(self.chain_data)
        Q_pred = gmrf.precision_
        for idx, x in np.ndenumerate(self.Q_chain):
            self.assertLess(abs(Q_pred[idx] - x), 0.2)

    def test_loop_cv(self):
        """ Test glasso with simple model A - B using CV
                                          |   |
                                          C - D
                                                        """
        gmrf = GMRF(method="cv")
        gmrf.fit(self.loop_data)
        Q_pred = gmrf.precision_
        for idx, x in np.ndenumerate(self.Q_loop):
            self.assertLess(abs(Q_pred[idx] - x), 0.2)

    def test_simple_chain_bic(self):
        """ Test glasso with simple model A - B - C using BIC """
        gmrf = GMRF(method="bic")
        gmrf.fit(self.chain_data)
        Q_pred = gmrf.precision_
        for idx, x in np.ndenumerate(self.Q_chain):
            if x == 0:
                self.assertEqual(Q_pred[idx], x)
            else:
                self.assertLess(abs(Q_pred[idx] - x), 0.2)
