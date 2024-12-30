from typing import List
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def mean(vector_of_x: List[float]) -> float:
    """
    Calculate the mean value of a list of floats.

    The mean value, often referred to as the average, is a measure of central tendency used in statistics and mathematics to summarize a set of data points. It is calculated by dividing the sum of all the values in the data set by the total number of values.

    Formula: μ = ( Σ x ) / n
    Where:
    - μ (mu) is the mean value of the list
    - Σ (sigma) x is the sum of all the values in the list
    - n is the total number of values in the list

    Parameters
    ----------
    vector_of_x : List[float]
        A list of float values.

    Returns
    -------
    float
        The mean value of the list.
    """
    return float(np.mean(vector_of_x))

def mean_normalization(vector_of_x: List[float]) -> List[float]:
    """
    Perform mean normalization on a list of floats.

    Mean normalization is a technique used in data preprocessing, particularly in machine learning and statistics,
    to scale data such that the mean of each feature becomes 0, and the values are adjusted relative to their range.
    This helps standardize the data, making it easier for algorithms to process and improving the performance of models.

    Formula: normalized_x = (x - mean(vector_of_x)) / range(vector_of_x)
    Where:
    - normalized_x is the normalized value of x
    - x is the original value of x
    - mean(vector_of_x) is the mean value of the list of x values
    - range(vector_of_x) is the range of the list of x values (max - min)

    Parameters
    ----------
    vector_of_x : List[float]
        A list of float values to be normalized.

    Returns
    -------
    List[float]
        A list of normalized float values.
    """
    mean_x: float = mean(vector_of_x)
    range_x: float = np.ptp(vector_of_x)  # np.ptp calculates the range (max - min) of the list

    if range_x == 0:
        return vector_of_x

    with ThreadPoolExecutor() as executor:
        normalized_vector_of_x = list(executor.map(lambda x: (x - mean_x) / range_x, vector_of_x))

    return normalized_vector_of_x

def standard_deviation(vector_of_x: List[float]) -> float:
    """
    Calculate the standard deviation of a list of floats.

    The standard deviation is a measure of the amount of variation or dispersion of a set of values. It is a statistic used in statistics and probability theory to quantify the amount of variation or dispersion in a set of values.

    Formula: σ = sqrt( Σ (x - μ)^2 / n )
    Where:
    - σ (sigma) is the standard deviation of the list
    - Σ (sigma) (x - μ)^2 is the sum of the squared differences between each value and the mean value
    - n is the total number of values in the list

    Parameters
    ----------
    vector_of_x : List[float]
        A list of float values.

    Returns
    -------
    float
        The standard deviation of the list.
    """
    return float(np.std(vector_of_x))

def z_score_normalization(vector_of_x: List[float]) -> List[float]:
    """
    Perform Z-score normalization on a list of floats.

    Z-score normalization, also known as standardization, is a technique used in data preprocessing, particularly in machine learning and statistics,
    to scale data such that it has a mean of 0 and a standard deviation of 1. This helps standardize the data,
    making it easier for algorithms to process and improving the performance of models.

    Formula: z = (x - μ) / σ
    Where:
    - z is the normalized value of x
    - x is the original value of x
    - μ is the mean value of the list of x values
    - σ is the standard deviation of the list of x values

    Parameters
    ----------
    vector_of_x : List[float]
        A list of float values to be normalized.

    Returns
    -------
    List[float]
        A list of normalized float values.
    """
    mean_x: float = mean(vector_of_x)
    std_dev_x: float = standard_deviation(vector_of_x)

    if std_dev_x == 0:
        return vector_of_x

    with ThreadPoolExecutor() as executor:
        normalized_vector_of_x = list(executor.map(lambda x: (x - mean_x) / std_dev_x, vector_of_x))

    return normalized_vector_of_x

def z_score_normalization_matrix(matrix_of_x: np.ndarray) -> np.ndarray:
    """
    Perform Z-score normalization on a matrix of floats.

    Z-score normalization, also known as standardization, is a technique used in data preprocessing, particularly in machine learning and statistics,
    to scale data such that it has a mean of 0 and a standard deviation of 1. This helps standardize the data,
    making it easier for algorithms to process and improving the performance of models.

    Formula: z = (x - μ) / σ
    Where:
    - z is the normalized value of x
    - x is the original value of x
    - μ is the mean value of the list of x values
    - σ is the standard deviation of the list of x values

    Parameters
    ----------
    matrix_of_x : np.ndarray
        A matrix of float values to be normalized.

    Returns
    -------
    np.ndarray
        A matrix of normalized float values.
    """
    mu = np.mean(matrix_of_x, axis=0)
    sigma = np.std(matrix_of_x, axis=0)
    normalized_matrix_of_x = (matrix_of_x - mu)/sigma
    return normalized_matrix_of_x