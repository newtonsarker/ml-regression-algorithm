import numpy as np
import matplotlib.pyplot as plt
import algorithm.logistic_regression.regression as r

dlc = dict(dlblue = '#0096ff', dlorange = '#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0')
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]
#plt.style.use('./deeplearning.mplstyle')

def main():
    x = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1)

    w = np.array([1, 2])
    b = 2
    cost = r.logistic_regression_cost(x, y, w, b)
    print(f"Cost: {cost}")

    fig,ax = plt.subplots(1,1,figsize=(4,4))
    plot_data(x, y, ax)

    ax.axis([0, 4, 0, 3.5])
    ax.set_ylabel('$x_1$')
    ax.set_xlabel('$x_0$')
    plt.show()



def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc='best' ):
    """ plots logistic data with two axis """
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(-1,)  #work with 1D or 1D y vectors
    neg = neg.reshape(-1,)

    # Plot examples
    ax.scatter(X[pos, 0], X[pos, 1], marker='x', s=s, c = 'red', label=pos_label)
    ax.scatter(X[neg, 0], X[neg, 1], marker='o', s=s, label=neg_label, facecolors='none', edgecolors=dlblue, lw=3)
    ax.legend(loc=loc)

    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False

if __name__ == "__main__":
    main()