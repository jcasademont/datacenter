import numpy as np

class Discretiser():
    def __init__(self, nb_intervals, minimum, maximun):
        self.nb_intervals = nb_intervals
        self.minimum = minimum
        self.maximun = maximun
        self.step = (maximun - minimum) / nb_intervals

        self.values = np.arange(nb_intervals + 1) * self.step + minimum
