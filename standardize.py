__author__ = 'markus'

import numpy as np


class Standardize:
    """A simple example class"""

    def __init__(self):
        self.v = None

    def standardize(self, X, skipFirst=False):
        if skipFirst:
            X = X[:, 1:]
        mus = np.mean(X, axis=0)
        sigmas = np.std(X, axis=0)
        X = np.divide(np.subtract(X, mus), sigmas)
        if skipFirst:
            X = np.append(np.ones((X.shape[0], 1)), X, 1)
        return X
