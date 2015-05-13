import operator
import itertools
import numpy as np
from sklearn.linear_model import LinearRegression

class MDP():
    def __init__(self, model, nb_samples, reward, gamma, feature_extractor,
                 discretiser, state_indices=None, action_indices=None,
                 next_action_indices=None, next_state_indices=None):

        self.model_ = model
        self.nb_samples_ = nb_samples
        self.reward = reward
        self.gamma = gamma
        self.feature_extractor = feature_extractor
        self.discretiser = discretiser
        self.linreg = LinearRegression()

        self.state_ = state_indices
        self.controls_ = action_indices
        self.next_controls_ = next_action_indices
        self.next_state_ = next_state_indices

        self.next_indices_ = np.append(self.next_state_, self.next_controls_)
        self.vector_size_ = np.size(self.next_indices_) * 2

    def _get_next_state(self, state, controls):
        x = np.zeros((1, self.vector_size_))
        x[0, self.state_] = state
        x[0, self.controls_] = controls

        p = self.model_.predict(x, self.next_indices_)
        x[0, self.next_indices_] = p[0, :]
        p = x[0, self.next_state_]

        return p

    def value_function(self, s):
        return self.linreg.predict(s)

    def learn(self):
        samples = self.model_.sample(self.nb_samples_)[:, self.state_]

        converged = False
        nb_iter = 0

        q = dict()
        y = np.empty(samples.shape[0])

        self.linreg.coef_ = np.zeros(np.size(self.state_))
        self.linreg.intercept_ = 0

        while not converged:

            for i, s in enumerate(samples):
                for c in itertools.product(self.discretiser.values,
                                           repeat=np.size(self.controls_)):

                    p = self._get_next_state(s, c)

                    q[c] = self.reward(s, c, self.discretiser) \
                         + self.gamma * self.value_function(p)

                y[i] = max(q.values())

            X = self.feature_extractor(samples)

            prev_theta = np.append(self.linreg.coef_.copy(),
                                   self.linreg.intercept_)
            print("[MDP LEARNING] Linear regression")
            self.linreg.fit(X, y, n_jobs=4)
            theta = np.append(self.linreg.coef_, self.linreg.intercept_)
            converged = np.allclose(prev_theta, theta, rtol=0.1)

            nb_iter += 1

            if nb_iter % 10 == 0:
                print("[MDP LEARNING] {} iters ...".format(nb_iter))

        print("[MDP LEARNING] Number of iterations = {}, r2 = {}"
                .format(nb_iter, self.linreg.score(X, y)))

    def get_action(self, state):
        q = dict()

        for c in itertools.product(self.discretiser.values,
                                   repeat=np.size(self.controls_)):

            p = self._get_next_state(state, c)
            q[c] = self.value_function(p)

        return max(q.items(), key=operator.itemgetter(1))[0]
