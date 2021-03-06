__author__ = 'markus'

import numpy as np
import math

class GradientDescent:
    """Gradient Descent"""

    def __init__(self, convergence_tolerance=0.00001):
        self.convergence_tolerance = convergence_tolerance

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

    def fit_stochastic_linear(self, X, y, theta, lam, num_iters=1000, caculated_iters=0):
        np.set_printoptions(suppress=True)
        y = np.reshape(np.array(y), (len(y), 1))
        X = np.array(X)
        convergence_tolerance = self.convergence_tolerance
        m = len(y)  # number of training examples
        # prev_error = sys.float_info.max
        # error = sum((np.dot(X, theta) - y) ** 2) / m
        errors = [self.compute_cost_linear(X, y, theta)[0]]

        #iteration = 0

        for iteration in range(1, num_iters + 1):
            for i in range(m):
                # self.compute_cost_linear(X, y, theta)
                prediction_difference = np.dot(X[i, :], theta)[0] - y[i]
                gradient_sample = (2 / m) * X[i, :] * prediction_difference
                gradient_sample = np.reshape(gradient_sample, (len(gradient_sample), 1))
                theta = theta - (lam * gradient_sample)
            errors.append(self.compute_cost_linear(X, y, theta)[0])
            # cost should go down
            # print('The error in iteration ' + str(iteration) + ' is ' + str(errors[iteration]))

            error_diff = errors[iteration - 1] - errors[iteration]
            if error_diff < 0:
                print('Gradient descent overshooting with error difference of ' + str(error_diff) + ' and error of ' +
                      str(errors[iteration]))

            if error_diff < convergence_tolerance and caculated_iters < 1:
                print('Gradient descent converged after ' + str(iteration) + ' iterations with error difference of '
                      + str(error_diff) + ' and error of ' + str(errors[iteration]))
                break

            if caculated_iters > 0 and iteration == caculated_iters:
                print('Gradient descent converged based on first calculated iterations at ' + str(iteration))
                break

        if iteration == num_iters:
            print('Gradient descent converged after reaching max iterations: ' + str(iteration) + '.')

        return theta, errors, iteration

    def fit_batch_linear(self, X, y, theta, lam, num_iters=1000, caculated_iters=0):
        np.set_printoptions(suppress=True)
        y = np.reshape(np.array(y), (len(y), 1))
        X = np.array(X)
        convergence_tolerance = self.convergence_tolerance
        m = len(y)  # number of training examples
        # prev_error = sys.float_info.max
        # error = sum((np.dot(X, theta) - y) ** 2) / m
        errors = [self.compute_cost_linear(X, y, theta)[0]]

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
            # print('The error in iteration ' + str(iteration) + ' is ', errors[iteration])

            error_diff = errors[iteration - 1] - errors[iteration]
            if error_diff < 0:
                print('Gradient descent overshooting with error difference of ' + str(error_diff) + ' and error of ' +
                      str(errors[iteration]))

            if error_diff < convergence_tolerance and caculated_iters < 1:
                print('Gradient descent converged after ' + str(iteration) + ' iterations with error difference of '
                      + str(error_diff) + ' and error of ' + str(errors[iteration]))
                break

            if caculated_iters > 0 and iteration == caculated_iters:
                print('Gradient descent converged based on first calculated iterations at ' + str(iteration))
                break

        if iteration == num_iters:
            print('Gradient descent converged after reach max iterations: ' + str(iteration) + '.')

        return theta, errors, iteration

    def fit_stochastic_logistic(self, X, y, theta, lam, num_iters=1000, caculated_iters=0):
        y = np.reshape(np.array(y), (len(y), 1))
        X = np.array(X)
        convergence_tolerance =self.convergence_tolerance
        m = len(y)  # number of training examples
        # prev_error = sys.float_info.max
        # error = sum((np.dot(X, theta) - y) ** 2) / m
        errors = [self.compute_cost_logistic(X, y, theta)[0]]

        #iteration = 0

        for iteration in range(1, num_iters + 1):
            for i in range(m):
                # self.compute_cost_logistic(X, y, theta)
                prediction_difference = 1/(1 + math.exp(-np.dot(X[i, :], theta))) - y[i]
                gradient_sample = (2 / m) * X[i] * prediction_difference
                gradient_sample = np.reshape(gradient_sample, (len(gradient_sample), 1))
                theta = theta - (lam * gradient_sample)
            errors.append(self.compute_cost_logistic(X, y, theta)[0])
            # cost should go down
            # print('Gradient descent converged after ' + str(iteration) + ' iterations with error difference of '
            #       + str(error_diff))

            error_diff = errors[iteration - 1] - errors[iteration]
            if error_diff < 0:
                print('Gradient descent overshooting with error difference of ' + str(error_diff) + ' and error of ' +
                      str(errors[iteration]))

            if error_diff < convergence_tolerance and caculated_iters < 1:
                print('Gradient descent converged after ' + str(iteration) + ' iterations with error difference of '
                      + str(error_diff) + ' and error of ' + str(errors[iteration]))
                break

            if caculated_iters > 0 and iteration == caculated_iters:
                print('Gradient descent converged based on first calculated iterations at ' + str(iteration))
                break

        if iteration == num_iters:
            print('Gradient descent converged after reach max iterations: ' + str(iteration) + '.')

        return theta, errors, iteration

    def fit_batch_logistic(self, X, y, theta, lam, num_iters=1000, caculated_iters=0):
        y = np.reshape(np.array(y), (len(y), 1))
        X = np.array(X)
        convergence_tolerance = self.convergence_tolerance
        m = len(y)  # number of training examples
        # prev_error = sys.float_info.max
        # error = sum((np.dot(X, theta) - y) ** 2) / m
        errors = [self.compute_cost_logistic(X, y, theta)[0]]

        #iteration = 0

        for iteration in range(1, num_iters + 1):
            gradient = 0

            for i in range(m):
                # self.compute_cost_logistic(X, y, theta)
                prediction_difference = 1/(1 + math.exp(-np.dot(X[i, :], theta))) - y[i]
                gradient += (2 / m) * X[i] * prediction_difference

            gradient = np.reshape(gradient, (len(gradient), 1))
            theta = theta - (lam * gradient)
            errors.append(self.compute_cost_logistic(X, y, theta)[0])
            # cost should go down
            # print('The error in iteration ' + str(iteration) + ' is ', errors[iteration])

            error_diff = errors[iteration - 1] - errors[iteration]
            if error_diff < 0:
                print('Gradient descent overshooting with error difference of ' + str(error_diff) + ' and error of ' +
                      str(errors[iteration]))

            if error_diff < convergence_tolerance and caculated_iters < 1:
                print('Gradient descent converged after ' + str(iteration) + ' iterations with error difference of '
                      + str(error_diff) + ' and error of ' + str(errors[iteration]))
                break

            if caculated_iters > 0 and iteration == caculated_iters:
                print('Gradient descent converged based on first calculated iterations at ' + str(iteration))
                break

        if iteration == num_iters:
            print('Gradient descent converged after reach max iterations: ' + str(iteration) + '.')

        return theta, errors, iteration