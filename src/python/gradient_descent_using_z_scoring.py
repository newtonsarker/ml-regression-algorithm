import numpy as np
import matplotlib.pyplot as plt

import algorithm.linear_regression.prediction as pred
import algorithm.linear_regression.feature_normalization as fn
import algorithm.linear_regression.gradient_descent as gd

def main():
    x_normalized, y = __normalized_features()
    initial_w = np.zeros(4)
    initial_b = 0.0
    alpha = 0.009
    iterations = 500
    w_final, b_final, cost_history = gd.gradient_descent_of_multiple_feature_model(x_normalized, y, initial_w, initial_b, alpha, iterations)
    fig, axes = plt.subplots(1, 4, figsize=(4, 4), sharey=True)

    col1 = x_normalized[:,0]
    col1pred = pred.prediction_of_multiple_feature_model(col1, w_final[0], b_final)
    ax1 = axes[0]
    ax1.scatter(col1, y, marker='x', c='r', label="Actual Value")
    ax1.plot(col1, col1pred, label="Predicted Value");

    col2 = x_normalized[:,1]
    col2pred = pred.prediction_of_multiple_feature_model(col2, w_final[1], b_final)
    ax2 = axes[1]
    ax2.scatter(col2, y, marker='x', c='r', label="Actual Value")
    ax2.plot(col2, col2pred, label="Predicted Value");

    col3 = x_normalized[:,2]
    col3pred = pred.prediction_of_multiple_feature_model(col3, w_final[2], b_final)
    ax3 = axes[2]
    ax3.scatter(col3, y, marker='x', c='r', label="Actual Value")
    ax3.plot(col3, col3pred, label="Predicted Value");

    col4 = x_normalized[:,3]
    col4pred = pred.prediction_of_multiple_feature_model(col4, w_final[3], b_final)
    ax4 = axes[3]
    ax4.scatter(col4, y, marker='x', c='r', label="Actual Value")
    ax4.plot(col4, col4pred, label="Predicted Value");

    plt.legend()
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