import numpy as np
import algorithm.logistic_regression.gradient_descent as gd

def main():
    x = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1])
    w = np.array([2.,3.])
    b = 1.0
    alpha = 0.1
    iterations = 10000

    dj_db_tmp, dj_dw_tmp = gd.derivative_of_cost_of_multiple_feature_model(x, y, w, b)
    print(f"dj_db: {dj_db_tmp}" )
    print(f"dj_dw: {dj_dw_tmp.tolist()}" )

    final_w, final_b, cost_history = gd.gradient_descent_of_multiple_feature_model(x, y, w, b, alpha, iterations)
    print(f"final_w: {final_w.tolist()}")
    print(f"final_b: {final_b}")
    print(f"cost_history: {cost_history}")

if __name__ == "__main__":
    main()