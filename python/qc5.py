# This file compares the cross validation accuracy and test accuracy for 
# a given C* and varying d

from data_reader import AbloneDataReader
from svmutil import *
import numpy as np
from scipy.sparse import csr_matrix

from matplotlib import pyplot as plt


# parameters for the training set up
no_training_samples = 3133
no_testing_samples = 1044

# number of times cross validation is performed to computed standard deviation
n_trials = 5
max_d = 4
## defining variables
abr = AbloneDataReader(no_training_samples, no_testing_samples)
err_arr = []
std_arr = []

parameters_arr_cv = ['-t 1 -d 1 -c 32 -v 10 -q', '-t 1 -d 2 -c 32 -v 10 -q', '-t 1 -d 3 -c 32 -v 10 -q', '-t 1 -d 4 -c 32 -v 10 -q']

parameters_arr = ['-t 1 -d 1 -c 32 -q', '-t 1 -d 2 -c 32 -q', '-t 1 -d 3 -c 32 -q', '-t 1 -d 4 -c 32 -q']

# abr.shuffle_data()
x_train, y_train = abr.get_train_data()
x_test, y_test = abr.get_test_data()

err_arr_cv = []
err_arr_test = []
nsv_arr = [] # number of support vectors for each d
nmsv_arr = [] # number of marginal support vectors for each d

for p in range(len(parameters_arr)):
    # computes cross valiation error for different d
    prob  = svm_problem(y_train, x_train, isKernel=False)
    #updating parameter degree
    parameters = parameters_arr_cv[p]
    param = svm_parameter(parameters)
    acc, _ = svm_train(prob, param)
    err_arr_cv.append(1 - acc/100)
    
    # training on the entire training data
    prob  = svm_problem(y_train, x_train)
    parameters = parameters_arr[p]
    param = svm_parameter(parameters)
    m = svm_train(prob, param)
    _, (ACC, MSE, SCC), _ = svm_predict(y_test, x_test, m)
    err_arr_test.append(1 - ACC/100)

    nsv_arr.append(m.get_nr_sv())
    tmp = m.get_sv_coef()
    nmsv = 0
    for i in range(len(tmp)):
        if not abs(abs(tmp[i][0]) - 32) < 0.001:
            nmsv += 1
            
    nmsv_arr.append(nmsv)

fig, ax = plt.subplots(1, 2)
x = np.arange(1,5)
ax[0].plot(x, err_arr_cv, label = "cv error")
ax[0].plot(x, err_arr_test, label = "test error")
ax[0].grid()
ax[0].legend()

ax[1].plot(x, nsv_arr, label = "number of support vectors")
ax[1].plot(x, nmsv_arr, label = "number of marginal support vectors")

ax[1].grid()
ax[1].legend()

plt.show()

