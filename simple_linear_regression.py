import numpy as np
import matplotlib.pyplot as plot
import algorithm.cost_calculator as cost_calc

def main():
    training_values_x = np.array([1.0, 2.0])
    training_values_y = np.array([300.0, 500.0])

    w = 100
    b = 10
    predicted_values = cost_calc.compute_model_output_list(training_values_x, w, b)

    w_range = np.array([10, 300])
    w_array = np.arange(*w_range, 5)
    cost_array = cost_calc.compute_cost_array(training_values_x, training_values_y, w_array, b)

    plot.style.use('./resources/deeplearning.mplstyle')
    figure, axes = plot.subplots(1, 2, constrained_layout=True, figsize=(8, 4))
    draw_prediction_graph(training_values_x, training_values_y, predicted_values, axes[0])
    draw_cost_intuition(training_values_x, training_values_y, w_array, b, cost_array, axes[1])
    plot.legend()
    plot.show()

def draw_prediction_graph(training_values_x, training_values_y, predicted_values, axes):
    axes.set_title("Housing Prices")
    axes.set_xlabel('Size (1000 sqft)')
    axes.set_ylabel('Price (in 1000s of dollars)')
    axes.scatter(training_values_x, training_values_y, marker='x', c='r', label='Training Values')
    axes.plot(training_values_x, predicted_values, c='b',label='Predictions')
    axes.vlines(training_values_x, training_values_y, predicted_values, linestyles='dotted', color='g', label='Cost of error')
    axes.legend()

def draw_cost_intuition(training_values_x, training_values_y, w_array, bias, cost_array, axes):
    weight_point = 150
    cost_point = cost_calc.compute_cost(training_values_x, training_values_y, weight_point, bias)

    axes.set_title("Cost xxx")
    axes.set_xlabel('Weight')
    axes.set_ylabel('Cost')
    axes.plot(w_array, cost_array, c='b', label='Predictions')
    axes.scatter(weight_point, cost_point, s=100, c='r', zorder=10, label=f"cost at w={weight_point}")
    axes.hlines(y=cost_point, xmin=axes.get_xlim()[0], xmax=weight_point, colors='purple', linestyles='dotted')
    axes.vlines(x=weight_point, ymin=axes.get_ylim()[0], ymax=cost_point, colors='purple', linestyles='dotted')
    axes.legend()

if __name__ == "__main__":
    main()