## This is an implementation to train data with an SVM that improves sparsity
from data_reader import AbloneDataReader
from svmutil import *
import numpy as np
from scipy.sparse import csr_matrix

from matplotlib import pyplot as plt


def compute_poly_kernel(x_i, x_j, d, coef0):
    
    return (x_i.dot(x_j) + coef0)**d

# parameters for the training set up
no_training_samples = 3133
no_testing_samples = 1044

# number of times cross validation is performed to computed standard deviation
max_d = 4
## defining variables
abr = AbloneDataReader(no_training_samples, no_testing_samples)

x_train, y_train = abr.get_train_data()
x_test, y_test = abr.get_test_data()

# updating data to the form that maximises sparsity
# bringing it out of sparsity to compute the kernel
x_train = x_train.toarray()
x_test = x_test.toarray()

err_cv_arr = []
err_test_arr = []
for d in range(1, max_d + 1):

    print("transforming given data to the desired form for the given d ...")
    x_train_ker = np.zeros((no_training_samples, no_training_samples))
    x_test_ker = np.zeros((no_testing_samples, no_training_samples))

    # bringing the given training data into the form that maximises sparsity
    for i in range(no_training_samples):
        for j in range(no_training_samples):
            x_train_ker[i , j] = y_train[j]*compute_poly_kernel(x_train[i], x_train[j], d, 0) 
    
    # doing the same for the test data
    for i in range(no_testing_samples):
        for j in range(no_training_samples):
            x_test_ker[i , j] = y_train[j]*compute_poly_kernel(x_test[i], x_train[j], d, 0) 

    x_train_csr = csr_matrix(x_train_ker, np.shape(x_train))
    x_test_csr = csr_matrix(x_test_ker, np.shape(x_test))

    print("training ....")
    params = '-c 32 -v 10 -q'
    # computes cross valiation error for different d
    prob  = svm_problem(y_train, x_train_csr, isKernel=False)
    param = svm_parameter(params)
    acc, _ = svm_train(prob, param)

    err_cv_arr.append(1 - acc/100)
    # computes test errors for different d
    params = '-c 32 -q'
    prob  = svm_problem(y_train, x_train_csr, isKernel=False)
    param = svm_parameter(params)
    m = svm_train(prob, param)
    _, (ACC, MSE, SCC), _ = svm_predict(y_test, x_test_csr, m)

    err_test_arr.append(1 - ACC/100)
    
x_arr = np.arange(1,5)

plt.plot(x_arr, err_cv_arr, label = "CV error")
plt.plot(x_arr, err_test_arr, label = "test error")
plt.legend()
plt.grid()
plt.show()