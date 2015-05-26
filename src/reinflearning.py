import utils
import itertools
import layouts
import argparse
import numpy as np
import matplotlib.pyplot as plt

from mdp.mdp import MDP
from gaussian.gmrf import GMRF
from gaussian.hrf import HRF
from discretiser import Discretiser
from sklearn.cross_validation import train_test_split

indice_ahu3_air_on = 91
inter_low = -4.57
coef_low = 0.32

inter_high = 3.12
coef_high = 0.08

class Controller():
    def __init__(self, nb_intervals, minimum, maximun):
        levels = [0, 1]
        step = (maximun - minimum) / nb_intervals
        power = np.arange(nb_intervals + 1) * step + minimum

        self.values = [(p, q) for p, q in itertools.product(levels, power)]
    def get_cost(self, name, value, state):
        return value[1]

    def get_value(self, name, value, state):
        if value[0] == 0:
            c = coef_low
            inter = inter_low
        else:
            c = coef_high
            inter = inter_high

        return state[0, indice_ahu3_air_on] - c * value[1] - inter

def reward(state, action, controller):
    r = 0

    # if np.any(state >= 32):
    #     r = -100 * np.size(np.where(state >= 32)[0])
    racks_values = []
    for k, v in state.items():
        if 'ahu' not in k:
            racks_values.append(v)
    r = -np.sum(0.2 * np.power(racks_values, 2))

    controls_cost = []
    for k, v in action.items():
        controls_cost.append(controller.get_cost(k, v, state))

    r -= np.sum(controls_cost)

    # for a in action:
    #     r += np.where(discretiser.values == a)[0][0] \
    #             - np.size(discretiser.values)

    return r

def feature_creator(X):
    return X

def main(layout, model, output):
    variables = getattr(layouts, layout)

    K = list(variables.keys())

    df = utils.prep_dataframe(keep=K)

    df_shifted = utils.create_shifted_features(df)
    df = df.join(df_shifted, how="outer")
    df = df.dropna()

    X = df.values

    if model[0] == 'gmrf':
        model = GMRF(variables_names=df.columns.values, alpha=0.1)
    elif model[0] == 'hybrid':
        model = HRF(k=5, k_star=10, variables_names=df.columns.values)

    train, test = train_test_split(X, test_size=0.25)

    model.fit(train)

    controls_vars = ['ahu_3_outlet']
    controller = Controller(6, 15, 30)
    mdp = MDP(model, 1000, reward, 0.8, feature_creator, controller,
              controls_vars)
    mdp.learn()

    # plt.figure()
    # plt.hist(test[:, 38:42].ravel(), bins=5, range=(5, 30))

    # plt.figure()
    # plt.plot(test[:, 38:42])

    l1_indices = [i for i, n in enumerate(model.variables_names) if 'l1_' in n and n[3:] not in controls_vars]
    actions = list()
    for i in range(test.shape[0]):
        actions.append(mdp.get_action(test[i, l1_indices]))

    actions_values_one = [a[0][1] for a in actions if a[0][0] == 0]
    actions_values_two = [a[0][1] for a in actions if a[0][0] == 1]
    print(np.sum(actions_values_one + actions_values_two))
    print(np.mean(np.append(actions_values_one, actions_values_two)))

    plt.figure()
    plt.plot(actions_values_one)

    plt.figure()
    plt.plot(actions_values_two)

    # plt.figure()
    # plt.plot(actions[:, 0], label="1")
    # plt.plot(actions[:, 1], label="2")
    # plt.plot(actions[:, 2], label="3")
    # plt.plot(actions[:, 3], label="4")
    # plt.legend(loc=0)

    # print(np.mean(actions, axis=0))

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reinforcement learning.')
    parser.add_argument('layout', metavar='LAYOUT',
                        help="Layout containing the variables.")
    parser.add_argument('-m', '--model',
                        nargs=1, choices=['gmrf', 'hybrid'],
                        default=['gmrf'],
                        help="Model to use.")
    parser.add_argument('-o', '--output',
                        help="File name to store output data.")
    args = parser.parse_args()
    main(**vars(args))
