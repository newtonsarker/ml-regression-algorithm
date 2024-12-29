import copy

import numpy as np
import matplotlib.pyplot as plt
import algorithm.linear_regression.gradient_descent as gd

def main():
    __draw_graph()

def __draw_graph():
    x, y = __read_data()
    feature_labels = ['size', 'bedrooms', 'floors', 'age']
    fig, axes = plt.subplots(2, 4, figsize=(12, 3), sharey=True)

    # feature graphs
    __draw_feature_graph(axes[0, 0], copy.deepcopy(x[:,0]), copy.deepcopy(y), feature_labels[0])
    __draw_feature_graph(axes[0, 1], copy.deepcopy(x[:,1]), copy.deepcopy(y), feature_labels[1])
    __draw_feature_graph(axes[0, 2], copy.deepcopy(x[:,2]), copy.deepcopy(y), feature_labels[2])
    __draw_feature_graph(axes[0, 3], copy.deepcopy(x[:,3]), copy.deepcopy(y), feature_labels[3])

    # gradient descent
    initial_w = np.zeros(4)
    initial_b = 0.0
    alpha = 1e-7
    iterations = 10
    w_final, b_final, cost_history = gd.gradient_descent_of_multiple_feature_model(x, y, initial_w, initial_b, alpha, iterations)

    # plot cost versus iteration
    gradient_ax = axes[1, 0]
    gradient_ax.plot(copy.deepcopy(cost_history))
    gradient_ax.set_title("Cost vs. iteration")
    gradient_ax.set_ylabel('Cost')
    gradient_ax.set_xlabel('iteration step')

    # plot cost versus iteration (tail)
    gradient_ax1 = axes[1, 1]
    gradient_ax1_data = copy.deepcopy(cost_history[100:])
    gradient_ax1.plot(100 + np.arange(len(gradient_ax1_data)), gradient_ax1_data)
    gradient_ax1.set_title("Cost vs. iteration (tail)")
    gradient_ax1.set_ylabel('Cost')
    gradient_ax1.set_xlabel('iteration step')

    plt.show()

def __draw_feature_graph(ax, x, y, feature_label):
    ax.scatter(x, y)
    ax.set_xlabel(feature_label)
    ax.set_ylabel("Price (1000's)")

def __read_data():
    file_path = '../resources/data/houses.txt'
    data = np.loadtxt(file_path, delimiter=',')
    x = data[:,:4] # columns 0 to 3 are the features
    y = data[:,4] # column 4 is the target
    return x, y

if __name__ == "__main__":
    main()