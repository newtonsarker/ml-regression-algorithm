import numpy as np
import algorithm.linear_regression.gradient_descent as gd

def main():
    training_values_x = np.array([1.0, 2.0])
    training_values_y = np.array([300.0, 500.0])
    w_in = 0
    b_in = 0
    alpha = 8.0e-1
    num_iters = 10
    gd.gradient_descent_of_single_feature_model(training_values_x, training_values_y, w_in, b_in, alpha, num_iters)

if __name__ == "__main__":
    main()