import numpy as np

def logistic_regression_cost(x, y, w, b):
    """
    Compute the logistic regression cost function.

    The cost function for logistic regression measures the performance of a classification model
    whose output is a probability value between 0 and 1. It is also known as the log-loss or binary cross-entropy loss.

    Args:
        x (ndarray): Input data matrix with shape (m, n), where m is the number of examples and n is the number of features.
        y (ndarray): Target values, with shape (m,).
        w (ndarray): Weight vector with shape (n,), representing the coefficients for each feature.
        b (float): Bias term, a scalar added to the weighted sum of the features.

    Returns:
        float: The computed cost.
    """

    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        # z_i = np.dot(x[i],w) + b
        z_i = prediction_model(x[i], w, b) # predicted value
        f_wb_i = sigmoid(z_i) # predicted probability
        loss = -y[i] * np.log(f_wb_i) - (1-y[i]) * np.log(1-f_wb_i)
        cost += loss

    cost = cost / m
    return cost

def sigmoid_predictions(x, w, b):
    """
    Compute the logistic regression predictions.

    Args:
        x (ndarray): Input data matrix with shape (m, n), where m is the number of examples and n is the number of features.
        w (ndarray): Weight vector with shape (n,), representing the coefficients for each feature.
        b (float): Bias term, a scalar added to the weighted sum of the features.

    Returns:
        ndarray: Predicted probabilities, with shape (m,).
    """
    predictions = prediction_model(x, w, b)
    return sigmoid(predictions)

def prediction_model(x, w, b):
    """
    Compute the prediction model for linear regression.

    The expression `x @ w + b` is used to compute the predicted values in a linear regression model using matrix operations.
    Here's a breakdown of each component:
        - `x`: This is the input data matrix with shape (m, n), where m is the number of examples and n is the number of features.
        - `w`: This is the weight vector with shape (n,), representing the coefficients for each feature.
        - `b`: This is the bias term, a scalar that is added to the weighted sum of the features.
        - `@`: This is the matrix multiplication operator in Python, which performs the dot product between `x` and `w`.

    The `@` operator performs the dot product between each row of `x` and the weight vector `w`,
    resulting in a vector of predicted values. Adding `b` to this vector adjusts each prediction by the bias term.

    Args:
        x (ndarray): Input data matrix with shape (m, n), where m is the number of examples and n is the number of features.
        w (ndarray): Weight vector with shape (n,), representing the coefficients for each feature.
        b (float): Bias term, a scalar added to the weighted sum of the features.

    Returns:
        ndarray: Predicted values, with shape (m,).
    """
    return x @ w + b

def sigmoid(z):
    """
    Compute the sigmoid of z.

    The sigmoid function, sometimes called the logistic function, is a mathematical function
    that takes any real number as input and outputs a value between 0 and 1.

    In the context of the sigmoid function, e is Euler’s number, a special mathematical
    constant approximately equal to 2.71828. Euler’s number is the base of the natural logarithm,
    and it appears in a wide variety of mathematical contexts, particularly those involving
    continuous growth or decay (for example, compound interest, population growth models, radioactive decay, etc.).

    NumPy has a function called `exp()`, which offers a convenient way to calculate the exponential
    (e^z) of all elements in the input array (z).

    Args:
        z (ndarray): A scalar or numpy array of any size.

    Returns:
        ndarray: Sigmoid of z, with the same shape as z.
    """
    return 1 / (1 + np.exp(-z))


def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost
    """

    m,n  = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b                                      #(n,)(n,)=scalar, see np.dot
        f_wb_i = sigmoid(z_i)                                          #scalar
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)      #scalar

    cost = cost/m                                                      #scalar

    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          #scalar
    reg_cost = (lambda_/(2*m)) * reg_cost                              #scalar

    total_cost = cost + reg_cost                                       #scalar
    return total_cost                                                  #scalar

def compute_gradient_logistic_reg(X, y, w, b, lambda_):
    """
    Computes the gradient for linear regression

    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b.
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                            #(n,)
    dj_db = 0.0                                       #scalar

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw