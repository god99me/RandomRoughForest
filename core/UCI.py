import numpy as np
from sklearn.datasets import load_iris


def create_random_matrix(row, col, scale=1):
    return np.random.random((row, col)) * scale


def create_iris_data_sets():
    return load_iris()

