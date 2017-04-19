import unittest

import numpy as np
from core.RandomRoughForest import RandomRoughForest
from core.UCI import create_iris_data_sets
from core.Preprocessing import normalize


class TestRandomRoughForest(unittest.TestCase):

    def test_get_reduce(self):
        iris = create_iris_data_sets()
        normalized = normalize(iris.data)
        hstack = np.concatenate((normalized, iris.target.reshape(150, 1)), axis=1)
        rrf = RandomRoughForest(hstack)
        result = rrf.get_reduced()
        print(result)

