import numpy as np

def compute_cost_array(training_values_x, training_values_y, w_array, b):
    cost_array = np.zeros(len(w_array))
    for i in range(len(w_array)):
        weight = w_array[i]
        cost = compute_cost(training_values_x, training_values_y, weight, b)
        cost_array[i] = cost
    return cost_array


def compute_cost(training_values_x, training_values_y, w, b):
    m: int = len(training_values_x)
    sum_of_squared_errors: int = 0
    for i in range(m):
        f_wb_array = compute_model_output_list(training_values_x, w, b)
        squared_error = (f_wb_array[i] - training_values_y[i]) ** 2
        sum_of_squared_errors += squared_error

    cost = (1 / (2 * m)) * sum_of_squared_errors
    return cost


def compute_model_output_list(training_values_x, w, b):
    m: int = len(training_values_x)
    f_wb_array = np.zeros(m)
    for i in range(m):
        f_wb_array[i] = w * training_values_x[i] + b

    return f_wb_array