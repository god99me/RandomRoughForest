from unittest import TestCase
import numpy as np

import core.UCI as UCI
from core.BaseRoughSet import BaseRoughSet
from core.Preprocessing import check_variances_threshold
from core.Preprocessing import normalize


class TestBaseRoughSet(TestCase):

    def setUp(self):
        iris = UCI.create_iris_data_sets()
        self.rough_set = BaseRoughSet(iris.data, iris.target)

    def test_get_split_feat(self):
        rough_set = self.rough_set
        feat_matrix = rough_set.get_split_feat(2)

    def test_get_split_desc(self):
        roughs_set = self.rough_set
        desc_dict = roughs_set.get_split_desc()
        self.assertIsNotNone(desc_dict)
        self.assertIsInstance(desc_dict, dict)

    def test_get_pos_set(self):
        attr = np.array([[[0, 1]],
                         [[1]],
                         [[2]],
                         [[3]],
                         [[4]],
                         [[5]],
                         [[6]],
                         [[7]]])
        desc = np.arange(8)
        rough_set = BaseRoughSet(attr, desc)
        col, count = rough_set.get_pos_set([], 0)
        # self.assertEqual(col, attr)
        self.assertEqual(count, 7)

    def test_get_reduced(self):
        # feat = np.array([[0.0909, 1.,     1],
        #                  [0,      0.3750, 1],
        #                  [0.4091, 0,      1],
        #                  [0.6364, 0.75,   1],
        #                  [1.,     0.3750, 2],
        #                  [0.9091, 0.5,    2],
        #                  [0.9545, 0.6250, 2],
        #                  [0.6818, 0.6250, 1]])
        feat = np.array([[5.1, 3.5, 1],
                         [4.9, 3.0, 1],
                         [5.8, 2.7, 1],
                         [6.3, 3.3, 1],
                         [7.1, 3.0, 2],
                         [6.9, 3.1, 2],
                         [7.0, 3.2, 2],
                         [6.4, 3.2, 1]])
        desc = np.array(['Setosa',
                         'Setosa',
                         'Virginica',
                         'Virginica',
                         'Virginica',
                         'Versicolor',
                         'Versicolor',
                         'Versicolor'])

        # pre process
        nor_feat = normalize(feat)
        # check_variances_threshold(desc)

        rough_set = BaseRoughSet(nor_feat, desc)
        reduced = rough_set.get_reduced()
        print(reduced)
