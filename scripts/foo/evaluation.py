import numpy as np
from sklearn.metrics import mean_squared_error


def eval_n_iterate(clf, test, target, nb_ite):
    errors = np.ndarray((len(test.values), nb_ite))
    for j in range(len(test.values)):
        cur = test.values[j]
        for i in range(min(len(target.values) - j, nb_ite)):
            p = clf.predict(cur)
            cur = [p, cur[0]]
            e = mean_squared_error([p], [target.values[j + i]])
            errors[j][i] = e

    return errors
