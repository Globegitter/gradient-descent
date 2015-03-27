__author__ = 'markus'

from time import time
from coordinateAscent import coordinateAscent
from coordinateAscentLasso import CoordinateAscentLasso
from standardize import Standardize
from synthData import SynthData
from sklearn import linear_model
from sklearn import decomposition
import numpy as np


def main():
    sd = SynthData()
    ca = coordinateAscent()
    cal = CoordinateAscentLasso()
    st = Standardize()
    beta0Seperate = True
    lam = 0.1

    X, y, b = sd.generateData(noise=False,  w=np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])[np.newaxis].T)
    #if beta0Seperate:
    #    beta = np.array([1, 1, 1, 1, 0, 0, 0, 0])[np.newaxis].T
    #else:
    #    beta = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])[np.newaxis].T

    #if beta0Seperate:
    #    y = 1 + np.dot(X, beta)
    #else:
    #    X = np.append(np.ones((X.shape[0], 1)), X, 1)
    #    y = np.dot(X, beta)

    print('Fitting the model with Lasso:')
    print('Lambda = ' + str(lam))
    print('beta0, array of betas:')
    t0 = time()
    print(cal.coordinateAscentLasso(y, X, lam, [], False, beta0Seperate))
    dt = time() - t0
    print('done in %.4fs.' % dt)

    print()
    print('Fitting the model with plain \'ol Coordinate Ascent')
    print('beta0, array of betas:')
    t0 = time()
    print(ca.coordinateAscent(y, X, [], False))
    dt = time() - t0
    print('done in %.4fs.' % dt)
    print()

    #print('Dictionary Learning')
    #dl = decomposition.DictionaryLearning(fit_algorithm='cd')
    #print(dl.fit(X))
    #print(np.shape(dl.components_))
    #print(dl.components_)

    print('Fitting the model with LARS (from the scikit library)')
    clf = linear_model.LassoLars(alpha=0.01)
    t0 = time()
    print(clf.fit(X, y))
    dt = time() - t0
    print('done in %.4fs.' % dt)
    print('array of betas:')
    print(clf.coef_)
    return 1

    #y = sd.generateData(D=D, w=w, noiseLevel=0.3)[1]
    #print(ca.coordinateAscent(y, D, [], True))


if __name__ == '__main__':
    main()
