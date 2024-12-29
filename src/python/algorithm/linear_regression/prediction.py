"""
prediction.py

This module provides functions to compute the predicted values of the training model items
for both single-feature and multiple-feature linear regression models.

Functions:
    prediction_of_single_feature_model(feature_x, weight, bias):
        Computes the predicted value of the training model item for a single-feature model.
        Formula: f_wb = weight * feature_x + bias

    prediction_of_multiple_feature_model(vector_of_feature_x, vector_of_weight, bias):
        Computes the predicted value of the training model item for a multiple-feature model.
        Formula: f_wb = np.dot(vector_of_feature_x, vector_of_weight) + bias

Usage:
    The functions in this module are used in the linear regression process to predict the output
    values based on the input features, weights, and bias. These predictions are then used to
    calculate the cost and update the model parameters during the training process.

    Example usage in linear regression:
        from prediction import prediction_of_single_feature_model, prediction_of_multiple_feature_model

        # Single-feature prediction
        predicted_value = prediction_of_single_feature_model(feature_x, weight, bias)

        # Multiple-feature prediction
        predicted_value = prediction_of_multiple_feature_model(vector_of_feature_x, vector_of_weight, bias)
"""
import numpy as np

def prediction_of_single_feature_model(feature_x, weight, bias):
    """
    Computes the predicted value of the training model item
    Args:
        feature_x (scalar): feature/input value of a single training data item
        weight (scalar): weight of the feature
        bias (scalar): bias parameter of the model, usually the y-intercept or the base value
    Return:
        f_wb (scalar): the predicted value of the training model item
    """
    return weight * feature_x + bias

def prediction_of_multiple_feature_model(vector_of_feature_x, vector_of_weight, bias):
    """
    Computes the predicted value of the training model item
    Args:
        vector_of_feature_x (array(n,)): feature/input values of a single training data item
        vector_of_weight (array(n,)): weight of the features
        bias (scalar): bias parameter of the model, usually the y-intercept or the base value
    Return:
        f_wb (scalar): the predicted value of the training model item
    """
    return np.dot(vector_of_feature_x, vector_of_weight) + bias