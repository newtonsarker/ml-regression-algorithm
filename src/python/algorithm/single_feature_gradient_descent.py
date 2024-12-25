import math

def projected_value_of_the_training_model_item(x, weight, bias):
    return weight * x + bias

def cost_of_error(training_values_x, training_values_y, weight, bias):
    m = len(training_values_x)
    cost = 0

    for i in range(m):
        x = training_values_x[i]
        y = training_values_y[i]
        f_wb = projected_value_of_the_training_model_item(x, weight, bias)
        cost = cost + (f_wb - y) ** 2
    total_cost = 1 / (2 * m) * cost

    return total_cost

def gradient(training_values_x, training_values_y, weight, bias):
    """
    Gradient meaning: the degree of steepness of a graph at any point.
    Computes the gradient for linear regression
    Args:
      training_values_x (ndarray (m,)): feature/input values of training data set
      training_values_y (ndarray (m,)): target/output values of training data set
      weight(scalar):                   model parameter weight
      bias(scalar):                     model parameter bias
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b
     """
    m = len(training_values_x)
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        x = training_values_x[i]
        y = training_values_y[i]
        f_wb = projected_value_of_the_training_model_item(x, weight, bias)
        dj_dw_i = (f_wb - y) * x # derivative of weight
        dj_db_i = f_wb - y # derivative of bias
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def compute_gradient_descent(training_values_x, training_values_y, w_in, b_in, alpha, num_iters):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking
    num_iters gradient steps with learning rate alpha

    Args:
      training_values_x (ndarray (m,))  : Data, m examples
      training_values_y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient

    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b]
      """

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient(training_values_x, training_values_y, w, b)

        # Update Parameters using equation (3) above
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_of_error(training_values_x, training_values_y, w, b))
            p_history.append([w, b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history  # return w and J,w history for graphing