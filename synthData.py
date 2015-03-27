import numpy as np

class SynthData:
    """
    Generating/Synthesyzing data for Coordinate Ascent/Lasso, as well as Dictionary Learning
    Clarification:
         - returns for CA/Lasso data are X, y and beta
         - returns for Dictionary Learning are D, y and w
    Uppercase letter variables are matrices. Lowercase either vectors or scalars.
    Dictionary D (or Matrix X), will be of size n*p
    y will be of size n*1
    w (sparse code) will be of size p*1
    w is sparse, with amount of 0s < p
    Check uniform or gaussian random
    """

    def __init__(self, size=100, features=10):
        self.D = np.array([])
        self.n = size
        self.p = features
        self.w = np.array([])

    def generateDictionary(self, beta0Seperate=True):
        self.D = np.random.randn(self.n, self.p)
        #adding all ones for beta0 ??do I need that??
        if not beta0Seperate:
            self.D = np.append(np.ones((self.n, 1)), self.D, 1)
        return self.D

    def generateWeight(self, beta0Seperate=True):
        #Create a zero column-vector
        self.w = np.zeros((self.p, 1))
        #Get set the first x elements 1; x is a random number between 1 and 8
        self.w[:np.random.randint(1, self.p - 1)] = 1
        #shuffle the vector of zeros and ones
        np.random.shuffle(self.w)
        #might remove that
        if not beta0Seperate:
            self.w = np.append([[1]], self.w, 1)
        #self.w[0][0] = 1
        return self.w

    def generateY(self, noise=True, noiseLevel=0.1, beta0Seperate=True):
        if not noise:
            noiseLevel = 0

        #Generate y with (or without) some noise
        if not beta0Seperate:
            self.y = np.dot(self.D, self.w) + noiseLevel * np.random.randn(self.D.shape[0], 1)
        else:
            self.y = 1 + np.dot(self.D, self.w) + noiseLevel * np.random.randn(self.D.shape[0], 1)
        return self.y

    def generateData(self, D=None, w=None, y=None, noise=True, noiseLevel=0.1, beta0Seperate=True):
        if D is None:
            D = self.generateDictionary(beta0Seperate)
        else:
            self.D = D

        if w is None:
            w = self.generateWeight()
        else:
            self.w = w

        if y is None:
            y = self.generateY(noise, noiseLevel, beta0Seperate)
        else:
            self.y = y

        return D, y, w