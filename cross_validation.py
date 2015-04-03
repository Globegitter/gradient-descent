__author__ = 'markus'

import numpy as np
from gradient_descent import GradientDescent
from standardize import Standardize
from random import shuffle
import matplotlib.pyplot as plt
import sys


# create indices for k-fold validation
def createKFoldIndices(nrIndices=4601, k_fold_nr=10):
    # creates an array of arrays to hold the indices for all the
    # k-fold sets
    k_fold_indices = [[] for i in range(k_fold_nr)]

    # placing the indices in their according sub-arrays
    for i in range(nrIndices):
        k_fold_indices[i % k_fold_nr].append(i)

    # shuffle all the indices to minimise chance of having spam and non-spam
    # emails grouped together
    for indices in k_fold_indices:
        shuffle(indices)

    return k_fold_indices

def get_train_set_partitions(k_fold_number, partition_number):
    train_set_partitions = np.arange(k_fold_number)
    train_set_partitions = np.delete(train_set_partitions, partition_number)
    return train_set_partitions

# returns the partitioned train and test-set
# assuming that partition_number starts at 0 and ends at k-fold - 1
# so if you have 10-fold test it ends at 9
def select_k_fold_partition(k_fold_data, partition_number):
    train_set_partitions = get_train_set_partitions(len(k_fold_data), partition_number)

    # select 9 parts of the k-fold set
    train_set = []
    for i in train_set_partitions:
        # out of the 3-d k_fold_data make a 2-d train_set, which just merges all the selected
        # k-fold partitions together
        train_set.extend(k_fold_data[i])
    # select the remaining part as the test_set, which has x arrays of y length
    # where-as x is the number of samples and y the number of features
    test_set = k_fold_data[partition_number]

    return np.array(train_set), np.array(test_set)


def run_cross_validation(gradient_descent_method, k_fold_nr, lambdas, num_iters, k_fold_data, y_data, k_fold_indices):
    gd = GradientDescent()
    # y_true = np.reshape(train_is_spam[0], (len(train_is_spam[0]), 1))
    # X = train_spam_data[0]
    # print(len(X[0]))
    # theta = np.zeros((len(X[0]), 1))

    # one sub-list for each lambda
    y_test_sets = [[] for i in range(len(lambdas))]
    predictions = [[] for i in range(len(lambdas))]
    test_errors = [[] for i in range(len(lambdas))]
    train_err_history = [[] for i in range(len(lambdas))]

    # train on train data taking all apart from one set and switching around
    for i in range(k_fold_nr):  # k_fold_nr
        for l in range(len(lambdas)):
            lam = lambdas[l]
            print('---------')
            print('Running k-fold set ' + str(i) + ' out of ' + str(k_fold_nr) + ' with lambda value ' + str(lam))
            # print('lambda value', lam)
            X_train_set, X_test_set = select_k_fold_partition(k_fold_data, i)
            y_train_set, y_test_set = select_k_fold_y(y_data, k_fold_indices, i)
            theta = np.zeros((len(X_train_set[0]), 1))
            dynamic_fit_call = getattr(gd, 'fit_' + gradient_descent_method[0] + '_' + gradient_descent_method[1])
            theta, train_errors = dynamic_fit_call(X_train_set, y_train_set, theta, lam, num_iters)
            # print('CALCULATING TEST ERROR')
            dynamic_cost_call = getattr(gd, 'compute_cost_' + gradient_descent_method[1])
            test_error = dynamic_cost_call(X_test_set, y_test_set, theta)[0]
            # print('test error is', test_error)
            # print('-------')
            # print('Making prediction')
            dynamic_predictions_call = getattr(gd, 'compute_prediction_' + gradient_descent_method[1])

            # predictions and the truthful y's are needed for the ROC calculation
            prediction = dynamic_predictions_call(X_test_set, theta)
            predictions[l].extend(prediction)
            y_test_sets[l].extend(y_test_set)
            # test error is need to determine the best lambda for the ROC curve
            test_errors[l].append(test_error)
            # print(test_errors[l])
            train_err_history[l].append(train_errors)
            # theta_res2 = gd.fit2(X, y_true, theta, start_lambda, num_iters)
            # gd.fit3(X, y_true, start_lambda, num_iters)
            # print("lambda = ", start_lambda, "error = ", error)
            # test_errors[i] = np.mean(test_errors[i])

    # getting the average testing error for all 3 lambdas across all k-fold test sets
    # which then makes it easy to choose the lambda with the smallest error
    test_errors = np.mean(test_errors, axis=1).tolist()

    lowest_error_index = test_errors.index(min(test_errors))

    # need to take element[0] since each element is wrapped in an array
    lowest_error_predictions = np.array([element[0] for element in predictions[lowest_error_index]])
    lowest_error_y_test_sets = np.array([element[0] for element in y_test_sets[lowest_error_index]])
    print(lowest_error_y_test_sets)

    return train_err_history, lowest_error_predictions, lowest_error_y_test_sets


def select_k_fold_y(y_data, k_fold_indices, partition_number):
    """
    Takes the complete y_data, a vector/one-dimensional array and splits it into
    the y_test_data as well as the y_train_data

    @param y_data: a one dimensional array
    @param k_fold_indices: the k-fold partition indices
    @param partition_number: the k-fold partition number
    """

    y_train_set = []
    y_test_set = [y_data[indices] for indices in k_fold_indices[partition_number]]

    train_set_partitions = get_train_set_partitions(len(k_fold_indices), partition_number)

    for i in train_set_partitions:
        # takes all the indices from the 2-d k_fold_indices array and flatten them out
        # via extend, since y is just a simple list/vector
        y_train_set.extend([y_data[indices] for indices in k_fold_indices[i]])

    # transform into useful np.array format for gradient descent
    y_train_set = np.reshape(np.array(y_train_set), (len(y_train_set), 1))
    y_test_set = np.reshape(np.array(y_test_set), (len(y_test_set), 1))

    return y_train_set, y_test_set


def get_command_line_args():
    # gradient_descent_method = [['batch', 'linear']]
    # gradient_descent_method = [['stochastic', 'logistic']]
    # gradient_descent_method = [['batch', 'logistic']]
    gradient_descent_methods = [['stochastic', 'linear']]

    nrArgs = len(sys.argv)

    if len(sys.argv) > 9:
        nrArgs = 9

    if len(sys.argv) < 3 or (len(sys.argv) - 1) % 2 != 0:
        print('You have to give an even number of arguments in sets of 2. E.g. \'stochastic linear\'. '
              'Running ' + gradient_descent_methods[0][0] + ' ' + gradient_descent_methods[0][1] + ' as default.')
        return gradient_descent_methods

    gradient_descent_methods = []

    print_msg = 'Running '

    for i in range(1, nrArgs, 2):
        print_msg += sys.argv[i] + ' ' + sys.argv[i + 1] + ' and '
        gradient_descent_methods.append([sys.argv[i], sys.argv[i + 1]])

    print_msg = print_msg[:-4]

    descent = 'descent.'
    if len(sys.argv) > 3:
        descent = 'descents.'

    print_msg += 'gradient ' + descent

    print(print_msg)
    print()

    return gradient_descent_methods


def main():
    np.set_printoptions(suppress=True)
    # loads the spambase data
    f = open("data/spambase.data")
    spam_data = np.array(np.loadtxt(f, delimiter=','))
    k_fold_nr = 10
    nr_samples = len(spam_data)

    # chooses which stochastic gradients to use on the run-through
    # takes defaults or you can provide command line arguments such as:
    # python3 cross_validation.py batch linear
    # python3 cross_validation.py batch linear stochastic linear batch logistic stochastic logistic
    gradient_descent_methods = get_command_line_args()

    k_fold_indices = createKFoldIndices(nr_samples, k_fold_nr)

    st = Standardize()

    # The last column, our complete y_true, which we will always take a subset from
    # based on our current k-fold group
    email_is_spam = spam_data[:, -1]

    # take all the features apart from the last one, since that is our y_true
    # containing all the information if an email actually is a spa,
    spam_data = spam_data[:, 0:-1]

    # normalising the spam_data
    spam_data_normalized = st.standardize(spam_data)

    # the spam data split into k_fold groups
    k_fold_spam_data = []
    for indices in k_fold_indices:
        k_fold_spam_data.append(spam_data_normalized[indices, :])

    lambdas = [1, 0.1, 0.01]
    num_iters = 500

    for gradient_descent_method in gradient_descent_methods:
        train_err_history, predictions, y_test_sets = run_cross_validation(
            gradient_descent_method, k_fold_nr, lambdas, num_iters, k_fold_spam_data, email_is_spam, k_fold_indices)

        train_err_history = np.array(train_err_history)

        # print(train_err_history)
        # print(train_err_history.shape)

        collapsed_train_err_history = [[], [], []]

        for lam_i in range(len(lambdas)):
            lam_err_history = train_err_history[lam_i]
            collapsed_train_err_history[lam_i].extend(np.mean(lam_err_history, axis=0))

        # print(collapsed_train_err_history)
        #
        # print(len(collapsed_train_err_history))
        # print(len(collapsed_train_err_history[0]))
        # print(len(collapsed_train_err_history[1]))
        # print(len(collapsed_train_err_history[2]))
        iterations = [i for i in range(len(collapsed_train_err_history[0]))]
        # print(collapsed_train_err_history)

        # best lambda based on test errors (should have 3 test errors, where-as each is the
        # mean of the 10 k-fold runs)
        # take the best lambda from the 3-d predictions array for the ROC curve
        # y_test as well, that is dependent on the k-group I chose
        print('plotting now')
        plt.figure(1)
        plt.plot(iterations, collapsed_train_err_history[0], 'g-', iterations,
                 collapsed_train_err_history[1], 'b-', iterations, collapsed_train_err_history[2], 'r-')
        plt.legend(['lambda=1.0', 'lambda=0.1', 'lambda=0.01'])

        # plt.plot(iterations, collapsed_train_err_history[1], 'b-', iterations, collapsed_train_err_history[2], 'r-')
        # plt.legend(['lambda=0.1', 'lambda=0.01'])

        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.show()

        step_size = 0.01
        nr_steps = (1 / step_size) + 1

        false_positive_rates = []
        true_positive_rates = []

        for spam_threshold in np.linspace(0, 1, nr_steps):
            spam_indices = [i for i, x in zip(range(len(y_test_sets)), y_test_sets) if x == 1]
            nr_spam = len(spam_indices)
            no_spam_indices = [i for i, x in zip(range(len(y_test_sets)), y_test_sets) if x == 0]
            nr_no_spam = len(no_spam_indices)

            nr_spam_predictions = len([element for element in predictions[spam_indices] if element > spam_threshold])

            nr_no_spam_predictions = len([element for element in predictions[no_spam_indices]
                                          if element > spam_threshold])

            # print('Nr of spam total', nr_spam)
            # print('Nr of spam classified', nr_spam_predictions)

            # x-coordinate = false_positive_rate
            false_positive_rates.append(nr_no_spam_predictions / nr_no_spam)

            # y-coordinate = true_positve_rate
            true_positive_rates.append(nr_spam_predictions / nr_spam)
            # find all the ones in the y_test and what percentage are they in the predictions
            # taken from y_test is true_positive
            # 30 true positives 70 false positive
            # true positive rate is e.g. 30 (classified above given threshold out of the 70) /70 (number of 1s)
            # false positive rate is e.g. 20 (classified above given threshold out of the 30) / 30 (number of 0s)
            # then plot as a point y = true positive rate, x = false positive rate
            # connected up = ROC curve
            # 1 ROC curve for all the 10 k-folds
            # false positives, what are 0s in y_test but are above the threshold in the predictions

        # sort the x and y coordinates so the AUC can be calculated properly
        false_positive_rates.sort()
        true_positive_rates.sort()

        auc_sum = 0

        for i in range(1, len(false_positive_rates)):
            x_subtracted = false_positive_rates[i] - false_positive_rates[i - 1]
            y_added = true_positive_rates[i] + true_positive_rates[i - 1]
            auc_sum += x_subtracted * y_added

        auc = (1/2) * auc_sum

        print('The AUC of my classifier is ' + str(auc))

        plt.figure(2)
        plt.plot(false_positive_rates, true_positive_rates)
        plt.legend(['ROC-Curve'])
        plt.show()


if __name__ == '__main__':
    main()