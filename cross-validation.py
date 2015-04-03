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
from sklearn.metrics import mean_squared_error
import sys
from random import shuffle
import statistics
from sklearn.linear_model import SGDClassifier
from gradient_descent import GradientDescent


def main():
    #sd = SynthData()
	#loads the spambase data
	f = open("data/spambase.data")
	spam_data = np.array(np.loadtxt(f,delimiter=','))

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

	#normalisation of the data
	st = Standardize()
	email_is_spam = spam_data[:,len(spam_data[0]) - 1]
	spam_data = spam_data[:, 0:len(spam_data[0]) - 2]
	spam_data_normalized = st.standardize(spam_data)
	#spam_data_normalized = np.array(spam_data_normalized)
	#group the data into 10 k-folds
	train_spam_data = []
	train_is_spam = []
	for indices in train_indices:
		train_spam_data.append(spam_data_normalized[indices, :]);
		train_is_spam.append(email_is_spam[indices])
	test_spam_data = spam_data_normalized[test_indices, :]
	test_is_spam = email_is_spam[test_indices]

	cal = CoordinateAscentLasso()
	start_lambda = 1 #this will then be divided by 10, 5 times

	# y = train_spam_data[0] * np.ones((len(train_spam_data[0]), 1))
	# y = np.zeros((len(train_spam_data[0]), 1))
	y_true = train_is_spam[0]
	beta0Seperate = True
	# print(len(test_is_spam))
	cal = CoordinateAscentLasso()
	while start_lambda > 0.00001:
		[beta0, beta] = cal.coordinateAscentLasso(y_true, train_spam_data[0], start_lambda)
		print(len(beta))
		# error = mean_squared_error(y_true)
		start_lambda = start_lambda / 10

    #
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
