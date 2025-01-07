import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
from public_tests import *

def main():
    # part1()
    X_train, y_train = load_data("data/ex2data2.txt")
    X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
    np.random.seed(1)
    initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
    initial_b = 0.5
    lambda_ = 0.5

    cost = compute_cost_reg(X_mapped, y_train, initial_w, initial_b, lambda_)
    print("Regularized cost :", cost)
    compute_cost_reg_test(compute_cost_reg)

    dj_db, dj_dw = compute_gradient_reg(X_mapped, y_train, initial_w, initial_b, lambda_)
    print(f"dj_db: {dj_db}", )
    print(f"First few elements of regularized dj_dw:\n {dj_dw[:4].tolist()}", )
    compute_gradient_reg_test(compute_gradient_reg)

def part1():
    X_train, y_train = load_data("data/ex2data1.txt")
    print("First five elements in X_train are:\n", X_train[:5])
    print("Type of X_train:",type(X_train))
    print("First five elements in y_train are:\n", y_train[:5])
    print("Type of y_train:",type(y_train))
    print ('The shape of X_train is: ' + str(X_train.shape))
    print ('The shape of y_train is: ' + str(y_train.shape))
    print ('We have m = %d training examples' % (len(y_train)))

    value = 0
    print (f"sigmoid({value}) = {sigmoid(value)}")

    print ("sigmoid([ -1, 0, 1, 2]) = " + str(sigmoid(np.array([-1, 0, 1, 2]))))
    sigmoid_test(sigmoid)

    m, n = X_train.shape
    initial_w = np.zeros(n)
    initial_b = 0.
    cost = compute_cost(X_train, y_train, initial_w, initial_b)
    print('Cost at initial w and b (zeros): {:.3f}'.format(cost))

    test_w = np.array([0.2, 0.2])
    test_b = -24.
    cost = compute_cost(X_train, y_train, test_w, test_b)
    print('Cost at test w and b (non-zeros): {:.3f}'.format(cost))
    compute_cost_test(compute_cost)

    dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
    print(f'dj_db at initial w and b (zeros):{dj_db}' )
    print(f'dj_dw at initial w and b (zeros):{dj_dw.tolist()}' )
    test_w = np.array([ 0.2, -0.5])
    test_b = -24
    dj_db, dj_dw  = compute_gradient(X_train, y_train, test_w, test_b)
    print('dj_db at test w and b:', dj_db)
    print('dj_dw at test w and b:', dj_dw.tolist())
    compute_gradient_test(compute_gradient)

    np.random.seed(1)
    tmp_w = np.random.randn(2)
    tmp_b = 0.3
    tmp_X = np.random.randn(4, 2) - 0.5
    tmp_p = predict(tmp_X, tmp_w, tmp_b)
    print(f'Output of predict: shape {tmp_p.shape}, value {tmp_p}')
    predict_test(predict)

    #plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")
    #plt.ylabel('Exam 2 score')
    #plt.xlabel('Exam 1 score')
    #plt.legend(loc="upper right")
    #plt.show()

# UNQ_C1
# GRADED FUNCTION: sigmoid
def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """

    ### START CODE HERE ###
    g = 1 / (1 + np.exp(-z))
    ### END SOLUTION ###

    return g


# UNQ_C2
# GRADED FUNCTION: compute_cost
def compute_cost(X, y, w, b, *argv):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns:
      total_cost : (scalar) cost
    """
    m, n = X.shape
    ### START CODE HERE ###
    lambda_ = argv[0] if len(argv) > 0 else 0
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)

    cost = cost/m

    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)
    reg_cost = (lambda_/(2*m)) * reg_cost

    total_cost = cost + reg_cost
    ### END CODE HERE ###
    return total_cost

# UNQ_C3
# GRADED FUNCTION: compute_gradient
def compute_gradient(X, y, w, b, *argv):
    """
    Computes the gradient for logistic regression

    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w.
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    ### START CODE HERE ###
    lambda_ = argv[0] if len(argv) > 0 else 0
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)
        err_i  = f_wb_i  - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    ### END CODE HERE ###


    return dj_db, dj_dw

# UNQ_C4
# GRADED FUNCTION: predict
def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w

    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model

    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """
    # number of training examples
    m, n = X.shape
    p = np.zeros(m)

    ### START CODE HERE ###
    # Loop over each example
    for i in range(m):
        z_wb = sigmoid(np.dot(X[i],w) + b)
        p[i] = 1 if z_wb > 0.5 else 0
    ### END CODE HERE ###
    return p

# UNQ_C5
def compute_cost_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar, float) Controls amount of regularization
    Returns:
      total_cost : (scalar)     cost
    """

    m, n = X.shape

    # Calls the compute_cost function that you implemented above
    cost_without_reg = compute_cost(X, y, w, b)

    # You need to calculate this value
    reg_cost = 0.

    ### START CODE HERE ###
    for j in range(n):
        reg_cost += (w[j]**2)
    reg_cost = (lambda_/(2*m)) * reg_cost
    ### END CODE HERE ###

    # Add the regularization cost to get the total cost
    total_cost = cost_without_reg + reg_cost

    return total_cost

# UNQ_C6
def compute_gradient_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the gradient for logistic regression with regularization

    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar,float)  regularization constant
    Returns
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b.
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w.

    """
    m, n = X.shape

    dj_db, dj_dw = compute_gradient(X, y, w, b)

    ### START CODE HERE ###
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    ### END CODE HERE ###

    return dj_db, dj_dw

if __name__ == "__main__":
    main()