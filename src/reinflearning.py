import utils
import layouts
import numpy as np
import matplotlib.pyplot as plt

from mdp.mdp import MDP
from gmrf.gmrf import GMRF
from discretiser import Discretiser
from sklearn.cross_validation import KFold

def reward(state, action, discretiser):
    r = 0

    # if np.any(state >= 32):
    #     r = -100 * np.size(np.where(state >= 32)[0])
    r = -np.sum(0.2 * np.power(state, 2))

    # for a in action:
    #     r += np.where(discretiser.values == a)[0][0] \
    #             - np.size(discretiser.values)

    return r

def feature_creator(X):
    return X

def main():
    K = list(layouts.datacenter_layout.keys())

    df = utils.prep_dataframe(keep=K)

    df_shifted = utils.create_shifted_features(df)
    df = df.join(df_shifted, how="outer")
    df = df.dropna()

    X = df.values

    # indices = np.append(np.arange(38), [42])
    # l1_indices = np.append(np.arange(43, 81), [85])
    indices = np.arange(38)
    l1_indices = np.arange(42, 80)

    gmrf = GMRF()

    lim = int(X.shape[0] * 4/5)
    train = X[:lim, :]
    test = X[lim:, :]

    gmrf.fit(train)

    discretiser = Discretiser(4, 5, 25)
    mdp = MDP(gmrf, 1000, reward, 0.8, feature_creator, discretiser,
              l1_indices, np.array([38, 39, 40, 41]),
              np.array([80, 81, 82, 83]), indices)
    mdp.learn()

    plt.figure()
    plt.hist(test[:, 38:42].ravel(), bins=5, range=(5, 30))

    plt.figure()
    plt.plot(test[:, 38:42])

    actions = np.empty((test.shape[0], 4))
    for i in range(actions.shape[0]):
        actions[i, :] = mdp.get_action(test[i, l1_indices])

    plt.figure()
    plt.hist(actions.ravel(), bins=5, range=(5, 30))

    plt.figure()
    plt.plot(actions[:, 0], label="1")
    plt.plot(actions[:, 1], label="2")
    plt.plot(actions[:, 2], label="3")
    plt.plot(actions[:, 3], label="4")
    plt.legend(loc=0)

    print(np.mean(actions, axis=0))

    plt.show()

if __name__ == "__main__":
    main()
