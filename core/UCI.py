import numpy as np


def create_random_matrix(row, col, scale=1):
    return np.random.random((row, col)) * scale

