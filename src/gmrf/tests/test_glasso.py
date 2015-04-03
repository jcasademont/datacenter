import unittest
import numpy as np
import models as mo
from numpy.linalg import inv

class TestGlasso(unittest.TestCase):

    def generate_data(Q):
        return np.random.multivariate_normal([0] * Q.shape[0], inv(Q), 3000)


    @classmethod
    def setUpClass(self):
        self.Q_chain = np.array([[3.0, 0.3, 0.0],
                                 [0.3, 2.0, 1.0],
                                 [0.0, 1.0, 1.8]])

        self.chain_data = self.generate_data(self.Q_chain)

    def test_simple_chain_cv(self):
        """ Test glasso with simple model A - B - C using CV """
        Q_pred, _ = mo.graphlasso(self.chain_data, method='cv')
        for idx, x in np.ndenumerate(self.Q_chain):
            self.assertLess(abs(Q_pred[idx] - x), 0.2)

    def test_simple_chain_bic(self):
        """ Test glasso with simple model A - B - C using BIC """
        Q_pred, _ = mo.graphlasso(self.chain_data, method='bic')
        for idx, x in np.ndenumerate(self.Q_chain):
            self.assertLess(abs(Q_pred[idx] - x), 0.2)
