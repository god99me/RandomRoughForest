import unittest

import pandas as pd
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
        for radius in map(lambda x: x / 10.0, range(1, 100, 2)):
            result = BaseRoughSet.partition_all(self.iris, [False, False, False, False, True], radius)
            attr_dependency = BaseRoughSet.calc_each_feature_attr_dependency(result)
            print(attr_dependency)

    def test_calc_feat_attr_dependency_without_one_feat(self):
        for radius in map(lambda x: x / 10.0, range(1, 50, 2)):
            result = BaseRoughSet.partition_all(self.iris, [False, False, False, False, True], radius)
            dependency = BaseRoughSet.calc_feat_attr_dependency_without_one_feat(result)
            print(dependency)





