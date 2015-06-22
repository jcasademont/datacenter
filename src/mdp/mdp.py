import operator
import itertools
import numpy as np
import multiprocessing as mp
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

class MDP():
    def __init__(self, model, nb_samples, reward, gamma, feature_extractor,
                 controller, control_vars, n_jobs):

        self.model = model
        self.nb_samples = nb_samples
        self.reward = reward
        self.gamma = gamma
        self.feature_extractor = feature_extractor
        self.controller = controller
        self.linreg = LinearRegression()

        self.control_vars = control_vars

        self.n_jobs = n_jobs

        self.prev_state_indices = []
        self.next_state_indices = []
        self.prev_controls_indices = []
        self.next_controls_indices = []

        for i, n in enumerate(model.variables_names):
                if n[3:] in control_vars:
                    self.prev_controls_indices.append(i)
                elif n in control_vars:
                    self.next_controls_indices.append(i)
                elif 'l1_' in n:
                    self.prev_state_indices.append(i)
                else:
                    self.next_state_indices.append(i)

    def _get_next_state(self, state, controls, controller):
        x = np.zeros((1, np.size(self.model.variables_names)))
        x[0, self.prev_state_indices] = state

        outlet_values = []
        for i, n in enumerate(self.control_vars):
            outlet_values.append(controller.get_value(n, controls[i], x))
        x[0, self.prev_controls_indices] = outlet_values

        p = self.model.predict(x, self.model.variables_names[np.append(self.next_state_indices, self.next_controls_indices)])

        return p[:, :len(self.next_state_indices)]

    def value_function(self, s):
        x = self.feature_extractor(s)
        return self.linreg.predict(x)[0]

    def _estimate_value_function(self, s):
        q = dict()
        for c in itertools.product(self.controller.values,
                                   repeat=np.size(self.control_vars)):

            p = self._get_next_state(s, c, self.controller)

            state = dict(zip(self.model.variables_names[self.prev_state_indices], s))
            action = dict(zip(self.control_vars, c))

            q[c] = self.reward(state, action, self.controller) \
                 + self.gamma * self.value_function(p)

        return max(q.values())

    def learn(self):
        samples = self.model.sample(self.nb_samples)[:, self.prev_state_indices]

        converged = False
        nb_iter = 0

        X = self.feature_extractor(samples)
        print("[MDP LEARNING] Linear regression")
        self.linreg.fit(X, [0] * X.shape[0], n_jobs=4)

        pool = mp.Pool(processes=self.n_jobs)

        while not converged and nb_iter < 100:

            y = pool.map(self._estimate_value_function, samples)

            X = self.feature_extractor(samples)

            prev_theta = np.append(self.linreg.coef_.copy(),
                                   self.linreg.intercept_)
            print("[MDP LEARNING] Linear regression")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
            self.linreg.fit(X_train, y_train, n_jobs=4)
            theta = np.append(self.linreg.coef_, self.linreg.intercept_)
            converged = np.allclose(prev_theta, theta, rtol=0.1)

            nb_iter += 1

            if nb_iter % 10 == 0:
                print("[MDP LEARNING] {} iters ...".format(nb_iter))

        print("[MDP LEARNING] Number of iterations = {}, r2 = {}"
                .format(nb_iter, self.linreg.score(X_test, y_test)))

    def get_action(self, state):
        q = dict()

        for c in itertools.product(self.controller.values,
                                   repeat=np.size(self.control_vars)):

            p = self._get_next_state(state, c, self.controller)
            q[c] = self.value_function(p)

        return max(q.items(), key=operator.itemgetter(1))[0]
