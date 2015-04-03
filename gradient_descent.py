__author__ = 'markus'

import numpy as np
import math
import sys


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

    def fit(self, X, y, theta, alpha, num_iters):
        print("X", X.shape)
        print("y", y.shape)
        print("theta", theta.shape)

        m = len(y)  # number of training examples
        J_history = np.zeros((num_iters, 1))
        i = 0
        TOL = 0.0001
        converged = False

        # currJ = self.compute_cost_linear(X, y, theta)
        # prevJ = -sys.float_info.max

        # print('tolerance1', currJ - prevJ)
        while i < num_iters:  # currJ - prevJ > TOL and
            l = np.dot(X, theta) - y
            theta -= (alpha * (1 / m) * np.dot(X.T, l))

            J_history[i, 0] = self.compute_cost_linear(X, y, theta)
            print("Iteration %d | Cost: %f" % (i, J_history[i, 0]))
            #if i > 0:# and abs(J_history[i, 0] - J_history[i - 1, 0]) <= TOL:
            #    print('Converged, iterations: ', abs(J_history[i - 1, 0] - J_history[i, 0]))
                #converged = True
            # prevJ = currJ
            # currJ = J_history[i, 0]
            # print('tolerance2', currJ - prevJ)

            i += 1

        return [J_history, theta]

    # m denotes the number of examples here, not the number of features
    def fit2(self, X, y, theta, alpha, numIterations):
        m = len(y)
        xTrans = X.transpose()
        J_history = np.zeros((numIterations, 1))
        for i in range(0, numIterations):
            hypothesis = np.dot(X, theta)
            loss = hypothesis - y
            # avg cost per example (the 2 in 2*m doesn't really matter here.
            # But to be consistent with the gradient, I include it)
            J_history[i, 0] = np.sum(loss ** 2) / (2 * m)
            print("Iteration %d | Cost: %f" % (i, J_history[i, 0]))
            # avg gradient per example
            gradient = np.dot(xTrans, loss) / m
            # update
            theta = theta - alpha * gradient
            # J_history[i, 0] = self.compute_cost_linear(X, y, theta)
        return theta

    def fit3(self, x, y, alpha, max_iter=10000, ep=0.0001):
        converged = False
        iter = 0
        m = x.shape[0]  # number of samples
        #  initial theta
        t0 = np.ones(x.shape[1])
        t1 = np.ones(x.shape[1])
        # total error, J(theta)
        J = sum(sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)]))
        # Iterate Loop
        while not converged:
            # for each training sample, compute the gradient (d/d_theta j(theta))
            grad0 = 1.0/m * sum(sum([(t0 + t1*x[i] - y[i]) for i in range(m)]))
            grad1 = 1.0/m * sum(sum([(t0 + t1*x[i] - y[i])*x[i] for i in range(m)]))
            # update the theta_temp
            temp0 = t0 - alpha * grad0
            temp1 = t1 - alpha * grad1
            # update theta
            t0 = temp0
            t1 = temp1
            # mean squared error
            e = sum(sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)]))
            if abs(J-e) <= ep:
                print('Converged, iterations: ', iter, e, abs(J-e))
                converged = True
            J = e
            # update error
            iter += 1  # update iter
            if iter == max_iter:
                print('Max interactions exceeded!')
                converged = True
        return t0, t1