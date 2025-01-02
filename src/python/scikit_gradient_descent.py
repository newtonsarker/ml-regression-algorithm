import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


def main():
    np.set_printoptions(precision = 2)
    multi_featured_input, y = __read_raw_data()

    scaler = StandardScaler()
    normalized_input = scaler.fit_transform(multi_featured_input)
    print(f"Peak to Peak range by column in Raw        X:{np.ptp(multi_featured_input, axis=0)}")
    print(f"Peak to Peak range by column in Normalized X:{np.ptp(normalized_input, axis=0)}")

    sgdr = SGDRegressor(max_iter=1000)
    sgdr.fit(normalized_input, y)
    print(sgdr)
    print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

    b_norm = sgdr.intercept_
    w_norm = sgdr.coef_
    print(f"model parameters:                   w: {w_norm}, b:{b_norm}")

    # make a prediction using sgdr.predict()
    y_pred_sgd = sgdr.predict(normalized_input)
    # make a prediction using w,b.
    y_pred = np.dot(normalized_input, w_norm) + b_norm
    print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

    print(f"Prediction on training set:\n{y_pred[:4]}" )
    print(f"Target values \n{y[:4]}")

    # plot predictions and targets vs original features
    feature_names = ['size(sqft)','bedrooms','floors','age']
    fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(normalized_input[:, i], y, color='red', label = 'target')
        ax[i].set_xlabel(feature_names[i])
        ax[i].scatter(normalized_input[:,i], y_pred, color='green', label = 'predict')
    ax[0].set_ylabel("Price"); ax[0].legend();
    fig.suptitle("target versus prediction using z-score normalized model")
    plt.show()

def __read_raw_data():
    file_path = '../resources/data/houses.txt'
    data = np.loadtxt(file_path, delimiter=',')
    x = data[:,:4] # columns 0 to 3 are the features
    y = data[:,4] # column 4 is the target
    return x, y


if __name__ == "__main__":
    main()