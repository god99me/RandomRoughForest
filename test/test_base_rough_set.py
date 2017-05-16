import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from core.BaseRoughSet import BaseRoughSet
from core.Preprocessing import normalize


class TestBaseRoughSet(unittest.TestCase):

    def setUp(self):
        iris = load_iris()
        iris_data = pd.DataFrame(iris.data)
        iris_target = pd.Series(iris.target)
        self.iris = pd.concat([iris_data, iris_target], axis=1)

    def test_partition(self):
        array_1d = normalize([1, 2, 3, 4, 4, 3, 3, 3, 2])
        expected = [(0,), (1, 8), (2, 5, 6, 7), (3, 4), (3, 4), (2, 5, 6, 7), (2, 5, 6, 7), (2, 5, 6, 7), (1, 8)]
        result = BaseRoughSet.partition(array_1d, True)
        self.assertEquals(result, expected)

        continuous_1d = normalize([0.8, 0.6, 0.3, 0.2, 0.1, 0.33])
        # for radius in map(lambda x: x/10.0, range(1, 100, 5)):
        result = BaseRoughSet.partition(continuous_1d, False, 0.1)
        self.assertEqual(set(result), {(0, 1, 2, 3, 4, 5)})

    def test_partition_all(self):
        result = BaseRoughSet.partition_all(self.iris, [False, False, False, False, True], 4)
        print(result.head(5))

    def test_calc_pos_set(self):
        pos_set_list = []
        for radius in map(lambda x: x / 10.0, range(1, 50, 2)):
            result = BaseRoughSet.partition_all(self.iris, [False, False, False, False, True], radius)
            pos_set = BaseRoughSet.calc_pos_set(result)
            pos_set_list.append(len(pos_set))
        print(pos_set_list)

    def test_calc_core(self):
        core_list = []
        for radius in map(lambda x: x / 10.0, range(1, 50, 2)):
            result = BaseRoughSet.partition_all(self.iris, [False, False, False, False, True], radius)
            core = BaseRoughSet.calc_core(result)
            core_list.append(len(core))
        print(core_list)

    def test_calc_each_feature_attr_dependency(self):
        n_features = self.iris.shape[1] - 1
        attrs = [[] for i in range(n_features)]

        for radius in map(lambda x: x / 100.0, range(1, 1000, 2)):
            result = BaseRoughSet.partition_all(self.iris, [False, False, False, False, True], radius)
            attr_dependency = BaseRoughSet.calc_each_feature_attr_dependency(result)
            for i in range(n_features):
                attrs[i].append(attr_dependency[i])
            print(attr_dependency)
        X = list(map(lambda x: x / 100.0, range(1, 1000, 2)))
        for i in range(n_features):
            plt.plot(X, attrs[i])

        plt.show()

    def test_calc_feat_attr_dependency_without_one_feat(self):
        n_features = self.iris.shape[1] - 1
        counts = np.array([0. for i in range(n_features)])
        threshold = 0.015
        for radius in map(lambda x: x / 100.0, range(1, 1000, 5)):
            result = BaseRoughSet.partition_all(self.iris, [False, False, False, False, True], radius)
            dependency = BaseRoughSet.calc_feat_attr_dependency_without_one_feat(result)
            count = np.where(dependency < threshold, 1, 0)
            counts += count
            print(dependency)
        print(counts)
        X = np.arange(n_features)
        plt.bar(X, counts, align='center', alpha=0.5)
        plt.show()

    def test_calc_red(self):
        # core_list = []
        # reds = []
        for radius in map(lambda x: x / 10.0, range(1, 50, 2)):
            result = BaseRoughSet.partition_all(self.iris, [False, False, False, False, True], radius)

            core = BaseRoughSet.calc_core(result)
            # core_list.append(len(core))
            print("core: ", core)

            core_pos_set = BaseRoughSet.calc_pos_set(result.iloc[:, list(core)])

            red_set = set()
            for i in range(5):
                red = BaseRoughSet.calc_red(result, core, core_pos_set, 0.15)
                red_set.add(tuple(red))

            print("reduct: ", red_set)
            # reds.append(len(red))

        # print(core_list)
        # print(reds)
