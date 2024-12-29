"""
cost_function.py

This module provides functions to calculate the cost of the error of the model
using Mean Squared Error (MSE) for linear regression models.

Functions:
    calculate_error_of_single_feature_model(vector_of_feature_x, vector_of_output_y, weight, bias):
        Calculates the cost of the error of the model using Mean Squared Error (MSE) for a single-feature model.
        Formula: MSE = (1 / (2 * m)) * Σ (f_wb - y)^2

    calculate_error_of_multiple_feature_model(vector_of_features_x, vector_of_output_y, vector_of_weight, bias):
        Calculates the cost of the error of the model using Mean Squared Error (MSE) for a multiple-feature model.
        Formula: MSE = (1 / (2 * m)) * Σ (f_wb - y)^2

Usage:
    The functions in this module are used in the linear regression process to calculate the cost
    of the error based on the predicted values and the actual target values. This cost is then
    used to update the model parameters during the training process.

    Example usage in linear regression:
        from cost_function import calculate_error_of_single_feature_model, calculate_error_of_multiple_feature_model

        # Calculate the cost of error for single-feature model
        cost = calculate_error_of_single_feature_model(vector_of_feature_x, vector_of_output_y, weight, bias)

        # Calculate the cost of error for multiple-feature model
        cost = calculate_error_of_multiple_feature_model(vector_of_features_x, vector_of_output_y, vector_of_weight, bias)
"""
from . import prediction

def calculate_error_of_single_feature_model(vector_of_feature_x, vector_of_output_y, weight, bias):
    """
    Calculates the cost of the error of the model using Mean Squared Error (MSE) for a single-feature model
    Args:
        vector_of_feature_x (array(m,)): feature/input values of the training data set
        vector_of_output_y (array(m,)): target/output values of the training data set
        weight (scalar): weight of the feature
        bias (scalar): bias parameter of the model, usually the y-intercept or the base value
    Return:
        cost (scalar): the cost of the error of the model
    Formula:
        MSE = (1 / (2 * m)) * Σ (f_wb - y)^2
    """
    m = len(vector_of_feature_x) # number of training data set
    cost = 0

    for i in range(m):
        x = vector_of_feature_x[i] # feature/input value of the training data set
        y = vector_of_output_y[i] # target/output value of the training data set
        f_wb = prediction.prediction_of_single_feature_model(x, weight, bias)
        cost = cost + (f_wb - y) ** 2
    total_cost = 1 / (2 * m) * cost

    return total_cost

def calculate_error_of_multiple_feature_model(vector_of_features_x, vector_of_output_y, vector_of_weight, bias):
    """
    Calculates the cost of the error of the model using Mean Squared Error (MSE) for a multiple-feature model
    Args:
        vector_of_features_x (array(m, n)): multiple feature/input values of the training data set
        vector_of_output_y (array(m,)): target/output values of the training data set
        vector_of_weight (array(n,)): weight values for the feature set
        bias (scalar): bias parameter of the model, usually the y-intercept or the base value
    Return:
        cost (scalar): the cost of the error of the model
    Formula:
        MSE = (1 / (2 * m)) * Σ (f_wb - y)^2
    """
    m, _ = vector_of_features_x.shape #(number of training data set, number of features)
    cost = 0

    for i in range(m):
        x = vector_of_features_x[i] # feature set of the training data set
        y = vector_of_output_y[i] # target value of the training data set
        f_wb = prediction.prediction_of_multiple_feature_model(x, vector_of_weight, bias)
        cost = cost + (f_wb - y) ** 2
    total_cost = 1 / (2 * m) * cost

    return total_cost