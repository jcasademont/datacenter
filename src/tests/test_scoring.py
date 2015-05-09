import unittest
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
from eval import scoring

class TestScoring(unittest.TestCase):

    def test_one_step_scoring(self):
        """ Test one step scoring """
        X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [6, 9, 1, 9]])
        df = pd.DataFrame(X, index=np.arange(3),
                          columns=['l1_a', 'l1_b', 'a', 'b'])

        names = ['a', 'b']

        train = np.array([0])
        test = np.array([1, 2])

        gmrf_stub = GMRFStub(df.columns.values)
        scores = scoring(df, gmrf_stub, names, train, test, 1)[1]

        assert_allclose(scores, np.array([[3, 4], [11, 9]]))

    def test_n_steps_scoring(self):
        """ Test n steps scoring """
        X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [6, 9, 1, 9]])
        df = pd.DataFrame(X, index=np.arange(3),
                          columns=['l1_a', 'l1_b', 'a', 'b'])

        names = ['a', 'b']
        train = np.array([0])
        test = np.array([1, 2])

        gmrf_stub = GMRFStub(df.columns.values)
        scores = scoring(df, gmrf_stub, names, train, test)[1]

        assert_allclose(scores, np.array([[7, 6.5], [19, 15]]))

    def test_n_steps_scoring2(self):
        """ Test n steps scoring 2"""
        X = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [6, 9, 1, 9],
                      [14, 4, 7, 3]])
        df = pd.DataFrame(X, index=np.arange(4),
                          columns=['l1_a', 'l1_b', 'a', 'b'])

        names = ['a', 'b']
        train = np.array([0])
        test = np.array([1, 2, 3])

        gmrf_stub = GMRFStub(df.columns.values)
        scores = scoring(df, gmrf_stub, names, train, test)[1]

        assert_allclose(scores, np.array([[35/3, 6], [18, 24], [33, 45]]))

class GMRFStub():
    def __init__(self, vns):
        self.vns = np.array(vns)

    def fit(self, X):
        pass

    def predict(self, X, names):
        indices = [np.where(self.vns == n)[0][0] for n in names]
        _indices = list(filter(lambda x: x not in indices,
                                np.arange(X.shape[1])))
        preds = np.zeros((X.shape[0], np.size(indices)))
        for i in range(X.shape[0]):
            preds[i, :] = X[i, _indices] * 2
        return preds

    def variances(self, indices):
        pass
