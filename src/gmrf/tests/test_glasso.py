import unittest
import numpy as np
import models as mo
from numpy.linalg import inv

class TestGlasso(unittest.TestCase):

    def generate_data(self, Q):
        return np.random.multivariate_normal([0] * Q.shape[0], inv(Q), 3000)

    def test_simple_chain(self):
        """ Test glasso with simple model A - B - C """
        Q = np.array([[3.0, 0.3, 0.0],
                      [0.3, 2.0, 1.0],
                      [0.0, 1.0, 1.8]])

        data = self.generate_data(Q)
        Q_pred, _ = mo.graphlasso(data, cv=True)
        for idx, x in np.ndenumerate(Q):
            self.assertLess(abs(Q_pred[idx] - x), 0.1)
