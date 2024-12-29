import numpy as np
import matplotlib.pyplot as plot
import algorithm.linear_regression.gradient_descent as gd

def main():
    training_values_x = np.array([1.0, 2.0])
    training_values_y = np.array([300.0, 500.0])
    w_in = 0
    b_in = 0
    alpha = 5.0e-7
    num_iters = 9000000
    w_final, b_final, J_hist = gd.gradient_descent_of_single_feature_model(training_values_x, training_values_y, w_in, b_in, alpha, num_iters)

    fig, (ax1, ax2) = plot.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
    ax1.plot(J_hist)
    ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
    ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
    ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost')
    ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step')
    plot.show()

if __name__ == "__main__":
    main()