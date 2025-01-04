import copy
import numpy as np
from . import prediction
from . import cost_function

def derivative_of_cost_of_single_feature_model(vector_of_feature_x, vector_of_output_y, weight, bias):
    """
    Computes the gradient for linear regression with a single feature.
    Args:
        vector_of_feature_x (ndarray (m,)): feature/input values of the training data set
        vector_of_output_y (ndarray (m,)): target/output values of the training data set
        weight (scalar): model parameter weight
        bias (scalar): model parameter bias
    Returns:
        dj_dw (scalar): The derivative of the cost with respect to the parameter weight
        dj_db (scalar): The derivative of the cost with respect to the parameter bias
    """
    m = len(vector_of_feature_x) # number of training data set
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        x = vector_of_feature_x[i] # feature/input value of the training data set
        y = vector_of_output_y[i] # target/output value of the training data set
        f_wb = prediction.prediction_of_single_feature_model(x, weight, bias)
        prediction_error = f_wb - y
        dj_dw_i = prediction_error * x # derivative of weight
        dj_db_i = prediction_error # derivative of bias
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def derivative_of_cost_of_multiple_feature_model(vector_of_features_x, vector_of_output_y, vector_of_weight, bias):
    """
    Computes the gradient for linear regression with multiple features.
    Args:
        vector_of_features_x (array(m, n)): multiple feature/input values of the training data set
        vector_of_output_y (array(m,)): target/output values of the training data set
        vector_of_weight (array(n,)): weight values for the feature set
        bias (scalar): bias parameter of the model, usually the y-intercept or the base value
    Returns:
        dj_dw (ndarray (n,)): The derivative of the cost with respect to the parameters weights
        dj_db (scalar): The derivative of the cost with respect to the parameter bias
    """
    m, n = vector_of_features_x.shape #(number of training data set, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        x = vector_of_features_x[i] # feature set of the training data set
        y = vector_of_output_y[i] # target value of the training data set
        f_wb = prediction.prediction_of_multiple_feature_model(x, vector_of_weight, bias)
        prediction_error = f_wb - y
        for j in range(n):
            dj_dw[j] += prediction_error * x[j] # derivative of weight
        dj_db += prediction_error # derivative of bias

    # ∂J/∂w = 1/m * Σ (f_wb - y) * x
    dj_dw = dj_dw / m

    # ∂J/∂b = 1/m * Σ (f_wb - y)
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent_of_single_feature_model(vector_of_feature_x, vector_of_output_y, initial_weight, initial_bias, learning_rate_alpha, no_of_iterations):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking
    no_of_iterations gradient steps with learning rate alpha

    Args:
      vector_of_feature_x (ndarray (m,))  : Data, m examples
      vector_of_output_y (ndarray (m,))  : target values
      initial_weight (scalar): initial value of model parameter weight
      initial_bias (scalar): initial value of model parameter bias
      learning_rate_alpha (float): Learning rate
      no_of_iterations (int): number of iterations to run gradient descent

    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      cost_history (List): History of cost values
    """
    cost_history = []
    w = copy.deepcopy(initial_weight)
    b = copy.deepcopy(initial_bias)
    for _ in range(no_of_iterations):
        dj_dw, dj_db = derivative_of_cost_of_single_feature_model(vector_of_feature_x, vector_of_output_y, w, b)
        w = w - learning_rate_alpha * dj_dw # update the weight
        b = b - learning_rate_alpha * dj_db # update the bias
        cost = cost_function.calculate_error_of_single_feature_model(vector_of_feature_x, vector_of_output_y, w, b)
        cost_history.append(cost)

    return w, b, cost_history

def gradient_descent_of_multiple_feature_model(vector_of_features_x, vector_of_output_y, initial_weight, initial_bias, learning_rate_alpha, no_of_iterations):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking
    no_of_iterations gradient steps with learning rate alpha

    Args:
        vector_of_features_x (array(m, n)): multiple feature/input values of the training data set
        vector_of_output_y (array(m,)): target/output values of the training data set
        initial_weight (array(n,)): initial model parameters
        initial_bias (scalar): initial model parameter
        learning_rate_alpha (float): Learning rate
        no_of_iterations (int): number of iterations to run gradient descent

    Returns:
      w (array(n,)): Updated values of parameters after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      cost_history (List): History of cost values
    """

    cost_history = [] # cost history for graphing
    w = copy.deepcopy(initial_weight)
    b = copy.deepcopy(initial_bias)
    for _ in range(no_of_iterations):
        dj_dw, dj_db = derivative_of_cost_of_multiple_feature_model(vector_of_features_x, vector_of_output_y, w, b)
        w = w - learning_rate_alpha * dj_dw # update the weight
        b = b - learning_rate_alpha * dj_db # update the bias
        cost = cost_function.calculate_error_of_multiple_feature_model(vector_of_features_x, vector_of_output_y, w, b)
        cost_history.append(cost)

    return w, b, cost_history