import numpy as np
import math
import matplotlib.pyplot as plot
import algorithm.multiple_feature_gradient_descent as gd

def main():

    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])

    b_init = 785.1811367994083
    w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
    print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

    # initialize parameters
    initial_w = np.zeros_like(w_init)
    initial_b = 0.
    # some gradient descent settings
    iterations = 1000
    alpha = 5.0e-7
    # run gradient descent
    w_final, b_final, J_hist = gd.compute_gradient_descent(X_train, y_train, initial_w, initial_b,alpha, iterations)
    print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
    for i in range(len(J_hist)):
        if i % math.ceil(len(J_hist) / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_hist[-1]:8.2f}   ")

    m,_ = X_train.shape
    for i in range(m):
        print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

    # plot cost versus iteration
    fig, (ax1, ax2) = plot.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
    ax1.plot(J_hist)
    ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
    ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
    ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost')
    ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step')
    plot.show()


if __name__ == "__main__":
    main()