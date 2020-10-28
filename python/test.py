import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from svmutil import *

from matplotlib import pyplot as plt


col = ["class", "a", "b", "c", "d"]

data = pd.read_csv("./data/test.data", names = col, delimiter = " ")

x_train = np.zeros((len(data["a"]), 4))
y_train = np.zeros(len(data["a"]))


for i in range(len(x_train)):
    
    x_train[i][0] = data["a"][i][2:]
    x_train[i][1] = data["b"][i][2:]
    x_train[i][2] = data["c"][i][2:]
    x_train[i][3] = data["d"][i][2:]

    y_train[i] = data["class"][i]
    

x_t = x_train[0:2316]
y_t = y_train[0:2316]

x_train_csr = csr_matrix(x_t, np.shape(x_t))
scale_param = csr_find_scale_param(x_train_csr, lower=0)
x_t = csr_scale(x_train_csr, scale_param)

parameters_arr = ['-t 1 -d 1 -c 0.015 -v 10 -q', '-t 1 -d 1 -c 0.03 -v 10 -q', '-t 1 -d 1 -c 0.0625 -v 10 -q', '-t 1 -d 1 -c 0.125 -v 10 -q', '-t 1 -d 1 -c 0.25 -v 10 -q', '-t 1 -d 1 -c 0.5 -v 10 -q', \
                    '-t 1 -d 1 -c 2 -v 10 -q', '-t 1 -d 1 -c 4 -v 10 -q', '-t 1 -d 1 -c 8 -v 10 -q', \
                    '-t 1 -d 1 -c 16 -v 10 -q', '-t 1 -d 1 -c 32 -v 10 -q', '-t 1 -d 1 -c 64 -v 10 -q', '-t 1 -d 1 -c 128 -v 10 -q']


err_arr = []
std_arr = []
max_d = 4
for d in range(0,max_d):
    err_arr.append([])
    std_arr.append([])
    print("d has changed ...")
    for p in range(len(parameters_arr)):
        tmp_arr = []
        prob  = svm_problem(y_t, x_t, isKernel=False)
        #updating parameter degree
        parameters = parameters_arr[p][0:8] + str(d+1) + parameters_arr[p][9:]
        print(parameters)
        param = svm_parameter(parameters)
        _, acc_arr = svm_train(prob, param)
        tmp_arr = 1 - np.array(acc_arr)/100.0
        err_arr[d].append(np.mean(tmp_arr))
        std_arr[d].append(np.std(tmp_arr))
    
fig, ax = plt.subplots(max_d, 1)

x_axis = np.arange(-6,7)

for d in range(max_d):
    ax[d].plot(x_axis, np.add(err_arr[d], std_arr[d]), label = "mean + std")
    ax[d].plot(x_axis, np.subtract(err_arr[d], std_arr[d]), label = "mean - std")
    ax[d].plot(x_axis, err_arr[d], label = "mean cv for d = " + str(d+1))
    ax[d].grid()
    ax[d].set_ylim(0.0,0.4)
    ax[d].legend()


plt.show()