import utils
import layouts
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gaussian.gmrf import GMRF
from gaussian.hrf import HRF
from gaussian.gbn import GBN
from sklearn.cross_validation import train_test_split

import multiprocessing as mp

def scoring(df, model, names, X_train, X_test, i):
    X = df.values

    indices = [np.where(df.columns.values == n)[0][0] for n in names]

    Y_test = X_test[:, indices]
    Y_train = X_train[:, indices]

    print("** Fit ({})".format(i))
    try:
        model.fit(X_train)
    except:
        print("{} Not working".format(i))
        return i, None, None

    print("** Score train ({})".format(i))
    preds = model.predict(X_train, names)
    train_score = np.mean(np.absolute(Y_train - preds))

    print("** Score test ({})".format(i))
    preds = model.predict(X_test, names)
    test_score = np.mean(np.absolute(Y_test - preds))

    # plt.scatter(i, test_score, c="b")
    # plt.scatter(i, train_score, c="r")
    # plt.draw()

    return i, train_score, test_score

def main():
    K = list(layouts.datacenter_layout.keys())

    df = utils.prep_dataframe(keep=K)
    df_shifted = utils.create_shifted_features(df)
    df = df.join(df_shifted, how="outer")
    df = df.dropna()
    names = list(filter(lambda x: 'l1_' not in x, df.columns.values))

    # a = np.random.normal(5, 2, 3000)
    # d = np.random.normal(-2, 3, 3000)
    # b = a * 3 + 9 + np.random.normal(0, 0.2, 3000)
    # c = 4 * d + (-5) * b + 11 + np.random.normal(0, 0.01, 3000)
    # X = np.array([a, b, c, d]).T
    # names = ['c', 'a']

    # df = pd.DataFrame.from_records(columns=np.array(['a', 'b', 'c', 'd']), data=X)

    X = df.values

    gmrf = GMRF(variables_names=df.columns.values, alpha=0.1)
    hrf = HRF(k=5, k_star=10, variables_names=df.columns.values)
    gbn = GBN(variables_names=df.columns.values)

    X_train, X_test = train_test_split(X, test_size=0.25)

    cv_scores = []
    train_scores = []

    pool = mp.Pool(processes=8)
    results = [pool.apply_async(scoring,
               args=(df, hrf, names, X_train[:i, :], X_test, i))
               for i in range(100, X_train.shape[0], 100)]

    output = [p.get() for p in results]
    output.sort()
    output = [np.array(t) for t in zip(*output)]

    cv_scores = output[2]
    train_scores = output[1]

    # hl, = plt.plot([], [])
    # for i in range(100, X_train.shape[0], 100):
    #     print("* Round {}".format(int(i / 100)))
    #     train_score, cv_score = scoring(df, gmrf, names, X_train[:i, :], X_test, i)

    #     cv_scores.append(cv_score)
    #     train_scores.append(train_score)

    #     hl.set_xdata(numpy.append(hl.get_xdata(), i))
    #     hl.set_ydata(numpy.append(hl.get_ydata(), train_score))
    #     plt.draw()

    #     # plt.plot(cv_scores, 'bo-')
    #     # plt.plot(train_scores, 'ro-')
    #     # plt.draw()

    # plt.ioff()
    plt.plot(range(100, X_train.shape[0], 100), cv_scores, 'bo-')
    plt.plot(range(100, X_train.shape[0], 100), train_scores, 'ro-')
    plt.show()

if __name__ == "__main__":
    main()
