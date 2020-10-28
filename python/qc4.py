# This file computes the 10 fold corss validation error on the data set
# it also plots the errors for different C and d.

from data_reader import AbloneDataReader
from svmutil import *
import numpy as np
from scipy.sparse import csr_matrix

from matplotlib import pyplot as plt


# parameters for the training set up
no_training_samples = 3133
no_testing_samples = 1044

# number of times cross validation is performed to computed standard deviation
max_d = 4
## defining variables
abr = AbloneDataReader(no_training_samples, no_testing_samples)
err_arr = []
std_arr = []

parameters_arr = ['-t 1 -d 1 -c 0.0625 -v 10 -q', '-t 1 -d 1 -c 0.125 -v 10 -q', '-t 1 -d 1 -c 0.25 -v 10 -q', '-t 1 -d 1 -c 0.5 -v 10 -q', \
                    '-t 1 -d 1 -c 2 -v 10 -q', '-t 1 -d 1 -c 4 -v 10 -q', '-t 1 -d 1 -c 8 -v 10 -q', \
                    '-t 1 -d 1 -c 16 -v 10 -q', '-t 1 -d 1 -c 32 -v 10 -q', '-t 1 -d 1 -c 64 -v 10 -q', '-t 1 -d 1 -c 128 -v 10 -q', '-t 1 -d 1 -c 256 -v 10 -q']


x_train, y_train = abr.get_train_data()

for d in range(0,max_d):
    err_arr.append([])
    std_arr.append([])
    print("d has changed ...")
    for p in range(len(parameters_arr)):
        tmp_arr = []
        prob  = svm_problem(y_train, x_train, isKernel=False)
        #updating parameter degree
        parameters = parameters_arr[p][0:8] + str(d+1) + parameters_arr[p][9:]
        print(parameters)
        param = svm_parameter(parameters)
        _, acc_arr = svm_train(prob, param)
        tmp_arr = 1 - np.array(acc_arr)/100.0
        err_arr[d].append(np.mean(tmp_arr))
        std_arr[d].append(np.std(tmp_arr))
    
fig, ax = plt.subplots(max_d, 1)

x_axis = np.arange(-4,8)

for d in range(max_d):
    ax[d].plot(x_axis, np.add(err_arr[d], std_arr[d]), label = "mean + std")
    ax[d].plot(x_axis, np.subtract(err_arr[d], std_arr[d]), label = "mean - std")
    ax[d].plot(x_axis, err_arr[d], label = "mean cv for d = " + str(d+1))
    ax[d].grid()
    ax[d].set_ylim(0.1,0.7)
    ax[d].legend()


plt.show()

