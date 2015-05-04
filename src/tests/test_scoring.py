import unittest
import numpy as np
from numpy.testing import assert_allclose
from eval import scoring

class TestScoring(unittest.TestCase):

    def test_one_step_scoring(self):
        """ Test one step scoring """
        X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [6, 9, 1, 9]])
        indices = np.array([2, 3])
        l1_indices = np.array([0, 1])

        train = np.array([0])
        test = np.array([1, 2])

        gmrf_stub = GMRFStub()
        scores = scoring(X, gmrf_stub, indices, l1_indices,
                         train, test, 1)[1]

        assert_allclose(scores, np.array([[2, 2], [5, 0]]))

    def test_n_steps_scoring(self):
        """ Test n steps scoring """
        X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [6, 9, 1, 9]])
        indices = np.array([2, 3])
        l1_indices = np.array([0, 1])

        train = np.array([0])
        test = np.array([1, 2])

        gmrf_stub = GMRFStub()
        scores = scoring(X, gmrf_stub, indices, l1_indices,
                         train, test)[1]

        assert_allclose(scores, np.array([[3.5, 1], [8, 1]]))

    def test_n_steps_scoring2(self):
        """ Test n steps scoring 2"""
        X = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [6, 9, 1, 9],
                      [14, 4, 7, 3]])

        indices = np.array([2, 3])
        l1_indices = np.array([0, 1])

        train = np.array([0])
        test = np.array([1, 2, 3])

        gmrf_stub = GMRFStub()
        scores = scoring(X, gmrf_stub, indices, l1_indices,
                         train, test)[1]

        assert_allclose(scores, np.array([[14/3, 1], [19/2, 7/2], [14, 5]]))

class GMRFStub():
    def fit(self, X):
        pass

    def predict(self, X, indices):
        _indices = list(filter(lambda x: x not in indices,
                                np.arange(X.shape[1])))
        preds = np.zeros((X.shape[0], np.size(indices)))
        for i in range(X.shape[0]):
            preds[i, :] = X[i, indices] * 2 - X[i, _indices]
        return preds
