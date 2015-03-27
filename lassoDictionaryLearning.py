__author__ = 'markus'

import numpy as np
from coordinateAscentLasso import CoordinateAscentLasso
import sys
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import linear_model


class LassoDictionaryLearning:
    """Lasso Dictionary learning using Coordinate Ascent"""

    def __init__(self):
        self. c = 0

    def updateDict(self, D, A, B):

        for j in range(D.shape[1]):
            #print('Update dictionary column ' + str(j + 1))
            if A[j, j] == 0:
                self.c += 1
                uj = D[:, j]
            else:
                uj = 1 / A[j, j] * (B[:, j] - np.dot(D, A[:, j])) + D[:, j]
            #print(uj)
            uj = uj[np.newaxis].T
            D[:, j] = (1 / np.maximum(np.absolute(uj), 1) * uj).flatten()

        return D

    def updateDict2(self, X, Y, W):
        xl = X.shape[1]
        for k in range(xl):
            #selecting weights column-vector wise
            print('w ' + str(k + 1) + 'th column-vector')
            w = W[:, k][np.newaxis].T
            #print(w)
            #exit()
            XNok = np.append(X[:, 0:k], X[:, k + 1:xl], axis=1)
            wNok = np.append(W[0:k, :], W[k + 1:W.shape[0], :], axis=0)
            ymk = Y - np.dot(XNok, wNok)
            print(ymk)
            print(np.shape(ymk))
            print('-------')
            print(w)
            print(np.shape(w))
            exit()
            print(np.dot(ymk, w))
            #return
            X[:, k] = np.dot(ymk, w) / sum(w)
        return X

    def updateDict5(self, X, Y, W):
        xw = X.shape[1]
        xh = X.shape[0]
        for k in range(xw):
            for m in range(xh):
                nok = 0
                ymk = 0
                #print(X)
                for i in range(Y.shape[1]):
                    for l in range(xw):
                        if l != k:
                            nok += X[m, l] * W[l, i]
                    #print('Xml * Wli = ')
                    #print(nok)
                    ymk += (Y[m, i] - nok) * W[k, i]

                #print(ymk)
                #print(np.shape(Y[m, :]))
                #X[m, k] = np.dot(Y[m, :], W[k, :]) / (sum(W[k, :] ** 2) + 1e-5)
                sw = sum(W[k, :] ** 2)
                if sw == 0:
                    sw = 1e-5
                X[m, k] = ymk / sw
        return X

    def updateDict6(self, X, Y, W, w0):
        xw = X.shape[1]  # same as W.shape[0]
        xh = X.shape[0]
        #predictY = np.zeros( X.shape[0], size( fitW, 2 ) );
        predictY = np.dot(X, W) + w0.T
        #predictY = self.computePreY(X, W, w0);
        #predictY =
        nok = 0
        #for i in range(Y.shape[1]):
        #    for l in range(xw):
        #        if l != k:
        #            nok += X[0, l] * W[l, i]

        for k in range(xh):
            for r in range(xw):
                a = W[r, :] * (Y[k, :] - predictY[k, :] + np.dot(X[k, r], W[r, :]))
                b = np.sum(W[r, :] ** 2)
                a = np.sum(a)
                if b == 0:
                    #update predictY first
                    predictY[k, :] = predictY[k, :] - X[k, r] * W[r, :]
                    X[k, r] = 0
                else:
                    #update predictY first
                    predictY[k, :] = predictY[k, :] - X[k, r] * W[r, :]
                    X[k, r] = a / b
                    predictY[k, :] = predictY[k, :] + X[k, r] * W[r, :]

        return X

    def updateDict3(self, X, y, w):
        xl = X.shape[0]
        xw = X.shape[1]
        print('Size of X = ' + str(xl) + ' * ' + str(xw))
        for m in range(xl):
            for k in range(xw):
                print(sum(w[k, :]))
                XNok = np.append(X[m, 0:k], X[m, k + 1:xw], axis=1)
                wNok = np.append(w[0:k, :], w[k + 1:w.shape[0], :])[:, np.newaxis]
                print('m before shape(y) = ' + str(m))
                print(np.shape(y[:, m]))
                print(XNok[np.newaxis])
                print(wNok)
                ymk = y[:, m] - np.dot(XNok, wNok)
                ymk = ymk[np.newaxis]
                print('ymk = ')
                print(ymk)
                print('w = ')
                print(w)
                print('m = ')
                print(m)
                X[m, k] = sum(ymk[:, m] * w) / sum(w)
        print('X = ')
        print(X)
        exit()
        return X

    #def updateDict4(self, X, y, w):
    #    s = X.shape[0]
    #    g = X.shape[1]
    #    for m in range(s):
    #        a, b = 0
    #        for k in range(g):

    #TODO: Change y to Y and w W, because they are matrices
    def dictLearning(self):
        #p > n?
        #samples or patients
        features = 90
        nr_atoms = 50
        #features or measurements
        samples = 11

        #Dictionary - Line generates a random normal distributed Matrix with dimensions of R^n*p
        #will be x in lasso
        X = np.abs(np.random.randn(features, nr_atoms))
        self.X = X
        #same as D = np.random.randn(s, r)

        #sparse code R^p*1 with # of unequal zeros (e.g. ones) < DicLen (or n?), rest are zeros
        w = np.zeros((nr_atoms, samples))
        for i in range(samples):
            w[:np.random.randint(1, nr_atoms), i] = 1
            np.random.shuffle(w[:, i])
        self.w = w
        #wTest = np.copy(w)
        #print(w)
        w0 = np.zeros((samples, 1))
        #print(np.dot(w, w.T))

        #Data matrix - R^n*1 => R^dicLent*wWidth
        #will be y in Lasso
        self.y = w0.T + np.dot(X, w) + 0.1 * np.abs(np.random.randn(features, samples))
        y = self.y
        #return
        #print(y)
        #print('y-shape');
        #print(np.shape(y))
        #return

        print('y = ')
        print(np.shape(y))
        print('n*m')
        print('X = ')
        print(np.shape(X))
        print('m*k')
        print('w = ')
        print(np.shape(w))
        print('k*n')
        return

        X = np.abs(np.random.randn(features, nr_atoms))
        #w = np.abs(np.random.randn(dicWidth, wWidth))
        for i in range(samples):
            w[:np.random.randint(1, nr_atoms), i] = 1
            np.random.shuffle(w[:, i])
        print(w)
        #exit()
        #return

        #assume default tolerance and number of iterations
        TOL = 1e-5
        MAXIT = 50
        lam = 0.1

        A = 0
        B = 0

        #Coordinate Ascent Lasso
        cal = CoordinateAscentLasso()
        clf = linear_model.Lasso(alpha=lam, max_iter=5000, warm_start=True)

        currentCost = -0.5 * np.sqrt(np.sum((y - w0.T - np.dot(X, w)) ** 2)) + lam * np.sum(np.abs(w))
        print('Cost before for loop')
        print(currentCost)
        previousCost = -sys.float_info.max

        i = 0

        print(currentCost - previousCost)

        while currentCost - previousCost > TOL and i < MAXIT:
            i += 1
            #each column of weights 'corresponds' to to columns of y.
            #print(w)
            #print('--')
            #print(wTest)
            print('Calculating Lasso...')
            for k in range(samples):
                #updating the weights column-wise
                #coef = cal.coordinateAscentLasso(y[:, k][np.newaxis].T, X, lam)[1].flatten()
                clf.fit(X, y[:, k])

                #print('---Coefficients---')
                #print('sklearn Lasso')
                #print(clf.coef_)
                #print()
                #print('My Lasso')
                #print(coef)
                #print('--------')

                w[:, k] = clf.coef_
                w0[k, 0] = clf.intercept_
                #wTest[:, k] = clf.coef_
            #print(w)
            #print(w)
            #print('--')
            #print(wTest)
            #exit()

            #A = A + np.dot(w, w.T)
            #B = B + np.dot(y, w.T)

            #X = self.updateDict(X, A, B)
            print('Updating dictionary...')
            X = self.updateDict6(X, y, w, w0)
            #print('X after first update:')
            #print(X)
            #X = self.updateDict3(X, y, w)
            #np.sum((y - np.dot(X, w)) ** 2)

            previousCost = currentCost
            currentCost = -0.5 * np.sqrt(np.sum((y - w0.T - np.dot(X, w)) ** 2)) + lam * np.sum(np.abs(w))
            #print('Current cost = ')
            #print(currentCost)
            print('Current cost - Previous cost: ')
            print(str(currentCost) + ' - ' + str(previousCost) + ' = ')
            print(currentCost - previousCost)
            print()

            assert currentCost - previousCost >= 0, 'Difference must be bigger than 0'

        return w, X

ldl = LassoDictionaryLearning()
ldl.dictLearning()
#w, X = ldl.dictLearning()
#print('w = ')
#print(w)
#print('X = ')
#print(X)
#print(ldl.c)

#ldl.y is the data matrix, so R^p*1 in this case. And ldl.y.shape[1] is therefore 1
#w, X, e = decomposition.dict_learning(ldl.y, ldl.y.shape[1], 0.1, method='cd')
#print(w, X)

print('Dictionary Learning')
print('y = ')
print(ldl.y)
print(np.shape(ldl.y))
dl = decomposition.DictionaryLearning(fit_algorithm='cd', alpha=0.1, verbose=True)
print(dl.fit(ldl.y))
#print(np.shape(dl.components_))
print(np.shape(dl.components_))
print(dl.components_)
print('----')
print(np.shape(dl.error_))
print(dl.error_)