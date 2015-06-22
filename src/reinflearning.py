import utils
import itertools
import layouts
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import transformations as tr

from mdp.mdp import MDP
from gaussian.gmrf import GMRF
from gaussian.hrf import HRF
from discretiser import Discretiser
from sklearn.cross_validation import train_test_split

indice_air_on = {'ahu_1_outlet': 87, 'ahu_2_outlet': 89,
                 'ahu_3_outlet': 91, 'ahu_4_outlet': 93}
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

        return state[0, indice_air_on[name]] - c * value[1] - inter

def reward(state, action, controller):
    r = 0
    gamma = 0.2
    roi = ['h6', 'eo6', 'e20', 'q4a', 'q13', 'n2', 'q1']

    # if np.any(state >= 32):
    #     r = -100 * np.size(np.where(state >= 32)[0])
    racks_values = []
    for k, v in state.items():
        # if 'ahu' not in k and 'air' not in k:
        if k in roi:
            racks_values.append(v)

    r_racks = np.sum(0.2 * np.power(racks_values, 2))
    # r_racks = np.sum(racks_values)

    # max_racks = np.sum(0.2 * np.power([40] * len(racks_values), 2))

    controls_cost = []
    for k, v in action.items():
        controls_cost.append(controller.get_cost(k, v, state))

    r_power = np.sum(controls_cost)

    r = r_racks + r_power

    # max_power = np.sum([30] * len(controls_cost))

    # r = ((1 - gamma) * r_power + gamma * r_racks) / ((1 - gamma) * max_power + gamma * max_racks)
    # for a in action:
    #     r += np.where(discretiser.values == a)[0][0] \
    #             - np.size(discretiser.values)

    return -r

def feature_creator(X):
    # ['h6', 'eo6', 'e20', 'q4a', 'q13', 'n2', 'q1']
    return X[:, [0, 3, 6, 31, 32, 35, 24, 47]]

def run_simulation(X_test, controls_vars, mdp, model, controller):
    l1_indices = [i for i, n in enumerate(model.variables_names)
                    if 'l1_' in n and n[3:] not in controls_vars]
    names = [n for n in model.variables_names
                if 'l1_' not in n and n not in controls_vars]
    controls_indices = [i for i, n in enumerate(model.variables_names)
                            if n[3:] in controls_vars]

    states = list()
    actions = list()

    s = X_test[0, l1_indices]
    for i in range(X_test.shape[0]):
        a = mdp.get_action(s)

        actions.append(a)
        states.append(s)

        x = np.zeros((1, len(model.variables_names)))
        x[0, l1_indices] = s

        outlet_values = []
        for i, n in enumerate(controls_vars):
            outlet_values.append(controller.get_value(n, a[i], x))

        x[0, controls_indices] = outlet_values

        s = model.predict(x, names)[0]

    return np.array(actions), np.array(states)

def main(layout, model, transform, output):
    variables = getattr(layouts, layout)

    K = list(variables.keys())

    df = utils.prep_dataframe(keep=K)

    df_shifted = utils.create_shifted_features(df)
    df = df.join(df_shifted, how="outer")
    df = df.dropna()

    # print(list(enumerate(df.columns.values)))
    # assert False

    if transform:
        print("* Tranform data")
        X = tr.to_normal(df.values)
        df = pd.DataFrame(X, index=df.index.values, columns=df.columns.values)

    X = df.values

    if model[0] == 'gmrf':
        model = GMRF(variables_names=df.columns.values, alpha=0.1)
    elif model[0] == 'hybrid':
        model = HRF(k=5, k_star=10, variables_names=df.columns.values)

    lim = int(X.shape[0] * 0.75)
    X_train = X[:lim]
    X_test = X[lim:]

    model.fit(X_train)
    print("* Model Fitted")

    # controls_vars = ['ahu_1_outlet', 'ahu_2_outlet', 'ahu_3_outlet', 'ahu_4_outlet']
    controls_vars = ['ahu_3_outlet']
    controller = Controller(6, 15, 30)
    mdp = MDP(model, 1000, reward, 0.8, feature_creator, controller,
              controls_vars, n_jobs=3)
    mdp.learn()

    # plt.figure()
    # plt.hist(test[:, 38:42].ravel(), bins=5, range=(5, 30))

    # plt.figure()
    # plt.plot(test[:, 38:42])

    actions, states = run_simulation(X_test, controls_vars, mdp, model, controller)

    print(actions)
    actions_values_one = [None] * len(controls_vars)
    actions_values_two = [None] * len(controls_vars)
    for i in range(len(controls_vars)):
        actions_values_one[i] = [(j, a[i][1]) for j, a in enumerate(actions)
                                if a[i][0] == 0]
        actions_values_two[i] = [(j, a[i][1]) for j, a in enumerate(actions)
                                if a[i][0] == 1]

        actions_values_one[i] = list(zip(*actions_values_one[i]))
        actions_values_two[i] = list(zip(*actions_values_two[i]))

    for i in range(len(controls_vars)):
        plt.figure()
        if len(actions_values_one[i]) != 0:
            plt.plot(list(actions_values_one[i][0]), list(actions_values_one[i][1]), 'b')
        if len(actions_values_two[i]) != 0:
            plt.plot(list(actions_values_two[i][0]), list(actions_values_two[i][1]), 'g')
        plt.title(controls_vars[i])

    max_states = np.amax(states, axis=1)
    mean_states = np.mean(states, axis=1)
    min_states = np.amin(states, axis=1)

    plt.figure()
    plt.plot(max_states, 'r')
    plt.plot(mean_states, 'g')
    plt.plot(min_states, 'b')

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
    parser.add_argument('-t', '--transform',
                        action='store_true', default=False,
                        help="Transform the data to Gaussian.")
    parser.add_argument('-o', '--output',
                        help="File name to store output data.")
    args = parser.parse_args()
    main(**vars(args))
