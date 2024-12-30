import copy

import numpy as np
import matplotlib.pyplot as plt

import algorithm.linear_regression.feature_normalization as fn
import algorithm.linear_regression.gradient_descent as gd

def main():
    x_normalized, y = __normalized_features()
    initial_w = np.zeros(4)
    initial_b = 0.0
    alpha = 0.009
    iterations = 500
    w_final, b_final, cost_history = gd.gradient_descent_of_multiple_feature_model(x_normalized, y, initial_w, initial_b, alpha, iterations)
    fig, axes = plt.subplots(1, 1, figsize=(12, 3), sharey=True)

    gradient_ax = axes
    gradient_ax.plot(copy.deepcopy(cost_history))
    gradient_ax.set_title("Cost vs. iteration")
    gradient_ax.set_ylabel('Cost')
    gradient_ax.set_xlabel('iteration step')

    plt.show()

def __normalized_features():
    multi_featured_vector_x, y = __read_raw_data()
    x_normalized = fn.z_score_normalization_matrix(multi_featured_vector_x)
    return x_normalized, y

def __read_raw_data():
    file_path = '../resources/data/houses.txt'
    data = np.loadtxt(file_path, delimiter=',')
    x = data[:,:4] # columns 0 to 3 are the features
    y = data[:,4] # column 4 is the target
    return x, y

if __name__ == "__main__":
    main()