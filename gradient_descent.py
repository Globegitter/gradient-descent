__author__ = 'markus'

import numpy as np
import math

class GradientDescent:
    """Gradient Descent"""
    def compute_prediction_linear(self, X, theta):
        return np.dot(X, theta)

    def compute_cost_linear(self, X, y, theta):
        m = len(y)  # number of training examples
        # J = (1 / (2 * m)) * np.dot((np.dot(X, theta) - y).T, (np.dot(X, theta) - y))
        # return J[0][0]
        prediction = self.compute_prediction_linear(X, theta)
        error = sum((prediction - y) ** 2) / m
        return error

    def compute_prediction_logistic(self, X, theta):
        return 1 / (1 + np.exp(-np.dot(X, theta)))

    def compute_cost_logistic(self, X, y, theta):
        m = len(y)  # number of training examples
        log_prediction = self.compute_prediction_logistic(X, theta)
        error = sum((log_prediction - y) ** 2) / m
        return error

    def fit_stochastic_linear(self, X, y, theta, lam, num_iters=100):
        np.set_printoptions(suppress=True)
        y = np.reshape(np.array(y), (len(y), 1))
        X = np.array(X)
        m = len(y)  # number of training examples
        # prev_error = sys.float_info.max
        # error = sum((np.dot(X, theta) - y) ** 2) / m
        errors = []

        #iteration = 0

        for iteration in range(num_iters):
            for i in range(m):
                # self.compute_cost_linear(X, y, theta)
                prediction_difference = np.dot(X[i, :], theta)[0] - y[i]
                gradient_sample = (2 / m) * X[i, :] * prediction_difference
                gradient_sample = np.reshape(gradient_sample, (len(gradient_sample), 1))
                theta = theta - (lam * gradient_sample)
            errors.append(self.compute_cost_linear(X, y, theta)[0])
            # cost should go down
            print('The error in iteration ' + str(iteration) + ' is ' + str(errors[iteration]))
            # print(abs(error - prev_error))

        return theta, errors

    def fit_batch_linear(self, X, y, theta, lam, num_iters=100):
        np.set_printoptions(suppress=True)
        y = np.reshape(np.array(y), (len(y), 1))
        X = np.array(X)
        m = len(y)  # number of training examples
        # prev_error = sys.float_info.max
        # error = sum((np.dot(X, theta) - y) ** 2) / m
        errors = []

        #iteration = 0

        for iteration in range(num_iters):
            gradient = 0

            for i in range(m):
                # self.compute_cost_linear(X, y, theta)
                prediction_difference = np.dot(X[i, :], theta) - y[i]
                gradient = gradient + ((2 / m) * X[i, :] * prediction_difference)

            gradient = np.reshape(gradient, (len(gradient), 1))
            theta = theta - (lam * gradient)
            errors.append(self.compute_cost_linear(X, y, theta)[0])
            # cost should go down
            print('The error in iteration ' + str(iteration) + ' is ', errors[iteration])
            # print(abs(error - prev_error))

        return theta, errors

    def fit_stochastic_logistic(self, X, y, theta, lam, num_iters=100):
        y = np.reshape(np.array(y), (len(y), 1))
        X = np.array(X)
        m = len(y)  # number of training examples
        # prev_error = sys.float_info.max
        # error = sum((np.dot(X, theta) - y) ** 2) / m
        errors = []

        #iteration = 0

        for iteration in range(num_iters):
            for i in range(m):
                # self.compute_cost_logistic(X, y, theta)
                prediction_difference = 1/(1 + math.exp(-np.dot(X[i, :], theta))) - y[i]
                gradient_sample = (2 / m) * X[i] * prediction_difference
                gradient_sample = np.reshape(gradient_sample, (len(gradient_sample), 1))
                theta = theta - (lam * gradient_sample)
            errors.append(self.compute_cost_logistic(X, y, theta)[0])
            # cost should go down
            print('The error in iteration ' + str(iteration) + ' is ', errors[iteration])
            # print(abs(error - prev_error))

        return theta, errors

    def fit_batch_logistic(self, X, y, theta, lam, num_iters=100):
        y = np.reshape(np.array(y), (len(y), 1))
        X = np.array(X)
        m = len(y)  # number of training examples
        # prev_error = sys.float_info.max
        # error = sum((np.dot(X, theta) - y) ** 2) / m
        errors = []

        #iteration = 0

        for iteration in range(num_iters):
            gradient = 0

            for i in range(m):
                # self.compute_cost_logistic(X, y, theta)
                prediction_difference = 1/(1 + math.exp(-np.dot(X[i, :], theta))) - y[i]
                gradient += (2 / m) * X[i] * prediction_difference

            gradient = np.reshape(gradient, (len(gradient), 1))
            theta = theta - (lam * gradient)
            errors.append(self.compute_cost_logistic(X, y, theta)[0])
            # cost should go down
            print('The error in iteration ' + str(iteration) + ' is ', errors[iteration])
            # print(abs(error - prev_error))

        return theta, errors