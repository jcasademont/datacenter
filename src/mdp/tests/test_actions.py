import unittest
import numpy as np
from numpy.testing import assert_allclose
from ..mdp import MDP

class TestAction(unittest.TestCase):

    def test_get_action(self):
        """ Test MDP get action """
        mdp = MDP(GMRFStub(), 0, None, 0, None, DiscretiserStub(),
                  np.array([0, 1]), np.array([2, 3]),
                  np.array([4, 5]), np.array([6, 7]))
        mdp.value_function = lambda x: np.sum(2 * x)
        a = mdp.get_action(np.array([10, -3]))
        assert_allclose(a, np.array([5, 5]))

class DiscretiserStub():
    def __init__(self):
        self.values = np.array([5, 10])

class GMRFStub():
    def predict(self, x, indices):
        full_indices = np.arange(np.size(x))
        _indices = np.setdiff1d(full_indices, indices)
        return np.power(x[:, indices], 2) \
                - np.sum(np.multiply(x[:, _indices],
                         np.arange(np.size(x[:, _indices]))))
