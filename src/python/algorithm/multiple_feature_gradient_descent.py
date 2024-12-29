import copy

import numpy as np

def predicted_value_of_the_training_model_item(vector_x, vector_w, b):
    """
    Computes the predicted value of the training model item
    Args:
        vector_x (array(n,)): feature/input values of a single training data item
        vector_w (array(n,)): weight of the features
        b (scalar): bias parameter of the model, usually the y-intercept or the base value
    Return:
        f_wb (scalar): the predicted value of the training model item
    """
    return np.dot(vector_x, vector_w) + b

def cost_of_error(vector_x, vector_y, vector_w, bias):
    """
    Calculates the cost of the error of the model
    Args:
        vector_x (array(m, n)): multiple feature/input values of the training data set
        vector_y (array(m,  )): target/output values of the training data set
        vector_w (array(   n)): weight values for the feature set
        bias (scalar): bias parameter of the model, usually the y-intercept or the base value
    Return:
        cost (scalar): the cost of the error of the model
    """
    m, _ = vector_x.shape #(number of training data set, number of features)
    cost = 0

    for i in range(m):
        x = vector_x[i] # feature set of the training data set
        y = vector_y[i] # target value of the training data set
        f_wb = predicted_value_of_the_training_model_item(x, vector_w, bias)
        cost = cost + (f_wb - y) ** 2
    total_cost = 1 / (2 * m) * cost

    return total_cost

def derivative(vector_x, vector_y, vector_w, bias):
    """
    The derivatives of J(w,b), also known as the partial derivatives, represent how the cost function J(w,b)
    changes with respect to the model parameters w (weights) and b (bias). In simpler terms, they tell us the
    direction and rate of change of the cost function for small changes in w and b.
    Args:
        vector_x (array(m, n)): multiple feature/input values of the training data set
        vector_y (array(m,  )): target/output values of the training data set
        vector_w (array(   n)): weight values for the feature set
        bias (scalar): bias parameter of the model, usually the y-intercept or the base value
    Returns
        dj_dw (ndarray (n,)): The derivative of the cost with respect to the parameters w.
        dj_db (scalar):       The derivative of the cost with respect to the parameter b.
     """
    m, n = vector_x.shape #(number of training data set, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        x = vector_x[i] # feature set of the training data set
        y = vector_y[i] # target value of the training data set
        f_wb = predicted_value_of_the_training_model_item(x, vector_w, bias)
        prediction_error = f_wb - y
        for j in range(n):
            dj_dw[j] += prediction_error * x[j] # derivative of weight
        dj_db += prediction_error # derivative of bias

    # ∂J/∂w = 1/m * Σ (f_wb - y) * x
    dj_dw = dj_dw / m

    # ∂J/∂b = 1/m * Σ (f_wb - y)
    dj_db = dj_db / m

    return dj_dw, dj_db

def compute_gradient_descent(vector_x, vector_y, w_in, b_in, alpha, num_iters):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking
    num_iters gradient steps with learning rate alpha

    Args:
        vector_x    (array(m, n)) : multiple feature/input values of the training data set
        vector_y    (array(m,  )) : target/output values of the training data set
        w_in        (array(   n)) : initial model parameters
        b_in        (scalar)      : initial model parameter
        alpha       (float)       : Learning rate
        num_iters   (int)         : number of iterations to run gradient descent
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b]
      """

    j_history = [] # cost history for graphing
    w = copy.deepcopy(w_in)
    b = copy.deepcopy(b_in)
    for i in range(num_iters):
        dj_dw, dj_db = derivative(vector_x, vector_y, w, b) # derivative of the J(w,b) with respect to w and b
        w = w - alpha * dj_dw # update the weight
        b = b - alpha * dj_db # update the bias
        j_history.append(cost_of_error(vector_x, vector_y, w, b))

    return w, b, j_history