import unittest

import pandas as pd

from core.Bagging import Bagging
from sklearn.datasets import load_iris


class TestBagging(unittest.TestCase):

    def setUp(self):
        iris = load_iris()
        iris_data = pd.DataFrame(iris.data)
        iris_target = pd.Series(iris.target)
        self.iris = pd.concat([iris_data, iris_target], axis=1)
        self.types = [False, False, False, False, True]

    def test_fetch_features(self):

        result = Bagging.fetch_features(self.iris, self.types, list(map(lambda x: x / 10.0, range(7, 19, 2))))
        print(result)

    def test_fetch_samples(self):
        fetched_rows = Bagging.fetch_samples(150)
        # print(fetched_rows)
        print(len(fetched_rows))

    def test_fetch_data(self):
        data = Bagging.fetch_data(self.iris, [0, 2, 44, 50], [2, 3])
        print(data)

    def test_fetch_decision(self):
        decision = Bagging.fetch_decision(self.iris, [0, 2, 44, 50])
        print(decision)

    def test_update_oob_list(self):
        oob_list = list(range(self.iris.shape[0]))
        bagging = Bagging(self.iris, self.types, list(map(lambda x: x / 10.0, range(7, 19, 2))), oob_list)
        bagging.train()
        result, y = bagging.classify()

        # print(result)
        # print(y.values)
        validator = pd.Series(result)
        ans = validator[validator != y]
        print(len(ans))








