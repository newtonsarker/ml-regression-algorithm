import numpy as np
import algorithm.linear_regression.cost_function as cf

def main():
    np.random.seed(1)
    X_tmp = np.random.rand(5,6)
    y_tmp = np.array([0,1,0,1,0])
    w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
    b_tmp = 0.5
    lambda_tmp = 0.7
    cost_tmp = cf.compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
    dj_db_tmp, dj_dw_tmp =  cf.compute_gradient_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

    print("Regularized cost:", cost_tmp)
    print(f"dj_db: {dj_db_tmp}", )
    print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )


if __name__ == "__main__":
    main()