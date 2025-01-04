import numpy as np
import regression as r

def sigmoid_predictions_valid_input():
    x = np.array([[1, 2], [3, 4]])
    w = np.array([0.5, -0.5])
    b = 1.0
    expected = r.sigmoid(np.array([0.5, 0.5]))
    result = r.sigmoid_predictions(x, w, b)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

def sigmoid_predictions_zero_weights():
    x = np.array([[1, 2], [3, 4]])
    w = np.array([0, 0])
    b = 1.0
    expected = r.sigmoid(np.array([1.0, 1.0]))
    result = r.sigmoid_predictions(x, w, b)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

def sigmoid_predictions_large_values():
    x = np.array([[1000, 2000], [3000, 4000]])
    w = np.array([0.5, -0.5])
    b = 1.0
    expected = r.sigmoid(np.array([-499.0, -499.0]))
    result = r.sigmoid_predictions(x, w, b)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

def sigmoid_valid_input():
    z = np.array([0, 2, -2])
    expected = np.array([0.5, 0.88079708, 0.11920292])
    result = r.sigmoid(z)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

def sigmoid_large_values():
    z = np.array([1000, -1000])
    expected = np.array([1.0, 0.0])
    result = r.sigmoid(z)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

def sigmoid_zero_input():
    z = np.array([0])
    expected = np.array([0.5])
    result = r.sigmoid(z)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def main():
    sigmoid_predictions_valid_input()
    sigmoid_predictions_zero_weights()
    sigmoid_predictions_large_values()
    sigmoid_valid_input()
    sigmoid_large_values()
    sigmoid_zero_input()
    print("All tests pass")


if __name__ == "__main__":
    main()