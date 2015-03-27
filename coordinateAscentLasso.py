__author__ = 'markus'

import numpy as np
import sys
import matplotlib.pyplot as plt

class CoordinateAscentLasso:
    """Coordinate Ascent for Lasso"""

    def logLikelihood(self, y, X, beta, lam=1, beta0=None):
        if beta0 is None:
            logl = -1 / 2 * np.dot((y - np.dot(X, beta)).T, (y - np.dot(X, beta))) - lam * sum(np.absolute(beta))
        else:
            logl = -1 / 2 * np.dot((y - beta0 - np.dot(X, beta)).T, (y - beta0 - np.dot(X, beta))) - lam * sum(np.fabs(beta))
        return logl[0][0]

    def shrinkage(self, x, lam=1):
        s = np.sign(x) * np.maximum(np.absolute(x) - lam, 0)
        return s[0]

    def coordinateAscentLasso(self, y, X, lam, init=None, drawGraph=False, beta0Seperate=True):
        assert X.shape[0] == y.shape[0] and y.shape[0] > 0, \
            'Matrices must have more than 0 rows and they have to be of the same dimension'
        #np.set_printoptions(suppress=True)
        n = y.shape[0]
        k = X.shape[1]

        if init:
            if beta0Seperate:
                beta0 = init[0]
                beta = init[1]
            else:
                beta = init[0]
        else:
            beta0 = y.mean(axis=0)[0]
            if beta0Seperate:
                beta = np.ones((k, 1))
            else:
                beta = np.ones((k - 1, 1))
                beta = np.append([[beta0]], beta, 0)
                beta0 = None
        #assume default tolerance and number of iterations
        TOL = 1e-5
        MAXIT = 100

        #tracking likelihood
        logls = np.zeros((MAXIT, 1))
        prevlogl = -sys.float_info.max

        logl = self.logLikelihood(y, X, beta, lam, beta0)

        i = 0
        plt.figure(1)

        while logl - prevlogl > TOL and i < MAXIT:
            prevlogl = logl

            #updates
            if beta0Seperate:
                beta0 = (1 / n) * np.sum((y - np.dot(X, beta)))

            for j in range(0, k):

                beta[j] = 0

                XNoj = np.append(X[:, 0:j], X[:, j + 1:k], axis=1)
                betaNoj = np.append(beta[0:j], beta[j + 1:k])
                betaNoj = betaNoj[np.newaxis].T
                if beta0Seperate:
                    yminj = y - beta0 - np.dot(XNoj, betaNoj)
                else:
                    yminj = y - np.dot(XNoj, betaNoj)
                x = np.dot(yminj.T, X[:, j])

                #Note: Why does that give exactly the same solution? Matrix inner product I think...
                #if beta0Seperate:
                #    x = np.dot((y - beta0 - np.dot(X, beta)).T, X[:, j])
                #else:
                #    x = np.dot((y - np.dot(X, beta)).T, X[:, j])

                beta[j] = self.shrinkage(x / np.sum(X[:, j] ** 2), lam / np.sum(X[:, j] ** 2))

            #likelihood for new state
            logl = self.logLikelihood(y, X, beta, lam, beta0)

            assert logl - prevlogl > 0, 'Difference must be bigger than 0'

            logls[i] = logl
            i += 1

        if drawGraph:
            #just plot all the log likelihoods not 0
            plt.plot(logls[logls != 0])
            plt.xlabel('iteration')
            plt.ylabel('log-likelihood')
            plt.show()

        if beta0Seperate:
            return beta0, beta
        else:
            return beta