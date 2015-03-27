__author__ = 'markus'

from time import time
from coordinateAscent import coordinateAscent
from coordinateAscentLasso import CoordinateAscentLasso
from standardize import Standardize
from synthData import SynthData
from sklearn import linear_model
from sklearn import decomposition
import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
import sys
from random import shuffle
import statistics


def main():
    #sd = SynthData()
	#loads the spambase data
	f = open("data/spambase.data")
	spam_data = np.loadtxt(f,delimiter=',')

	#create indices for k-fold validation
	train_indices = [[], [], [], [], [], [], [], [], []]
	test_indices = []
	for i in range(4601):
		if i % 10 == 9:
			test_indices.append(i)
		else:
			train_indices[i % 10].append(i);

	#shuffle all the indices to minimise chance of having spam and non-spam
	#emails grouped together
	shuffle(test_indices)
	for indices in train_indices:
		shuffle(indices);

	#mean
	print('mean')
	print(len(spam_data[0]))
	#mean = [x for x, i in range(len(spam_data)) and spam_data[i][0]]
	for i in range(len(spam_data[0])):
		print(statistics.mean(spam_data[:,i]))
		print(statistics.stdev(spam_data[:,i]))
		print()
	#print([row[1] for row in spam_data])
	#print(mean)
	sys.exit()


    #ca = coordinateAscent()
	#cal = CoordinateAscentLasso()
    #st = Standardize()
    #beta0Seperate = True
    #lam = 0.1

    #X, y, b = sd.generateData(noise=False,  w=np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])[np.newaxis].T)
    #if beta0Seperate:
    #    beta = np.array([1, 1, 1, 1, 0, 0, 0, 0])[np.newaxis].T
    #else:
    #    beta = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])[np.newaxis].T

    #if beta0Seperate:
    #    y = 1 + np.dot(X, beta)
    #else:
    #    X = np.append(np.ones((X.shape[0], 1)), X, 1)
    #    y = np.dot(X, beta)

    # print('Fitting the model with Lasso:')
    # print('Lambda = ' + str(lam))
    # print('beta0, array of betas:')
    # t0 = time()
    # print(cal.coordinateAscentLasso(y, X, lam, [], False, beta0Seperate))
    # dt = time() - t0
    # print('done in %.4fs.' % dt)
	#
    # print()
    # print('Fitting the model with plain \'ol Coordinate Ascent')
    # print('beta0, array of betas:')
    # t0 = time()
    # print(ca.coordinateAscent(y, X, [], False))
    # dt = time() - t0
    # print('done in %.4fs.' % dt)
    # print()

    #print('Dictionary Learning')
    #dl = decomposition.DictionaryLearning(fit_algorithm='cd')
    #print(dl.fit(X))
    #print(np.shape(dl.components_))
    #print(dl.components_)

    # print('Fitting the model with LARS (from the scikit library)')
    # clf = linear_model.LassoLars(alpha=0.01)
    # t0 = time()
    # print(clf.fit(X, y))
    # dt = time() - t0
    # print('done in %.4fs.' % dt)
    # print('array of betas:')
    # print(clf.coef_)
    # return 1

    #y = sd.generateData(D=D, w=w, noiseLevel=0.3)[1]
    #print(ca.coordinateAscent(y, D, [], True))


if __name__ == '__main__':
    main()
