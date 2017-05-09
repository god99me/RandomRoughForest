from unittest import TestCase
import numpy as np

import core.UCI as UCI
from core.BaseRoughSet import BaseRoughSet
from core.Preprocessing import normalize


class TestBaseRoughSet(TestCase):

    def setUp(self):
        iris = UCI.create_iris_data_sets()
        self.rough_set = BaseRoughSet(iris.data, iris.target)

        # dummy test case
        self.feat = np.array([[5.1, 3.5, 1],
                              [4.9, 3.0, 1],
                              [5.8, 2.7, 1],
                              [6.3, 3.3, 1],
                              [7.1, 3.0, 2],
                              [6.9, 3.1, 2],
                              [7.0, 3.2, 2],
                              [6.4, 3.2, 1]])
        self.desc = np.array(['Setosa',
                              'Setosa',
                              'Virginica',
                              'Virginica',
                              'Virginica',
                              'Versicolor',
                              'Versicolor',
                              'Versicolor'])

    def test_get_split_feat(self):
        rough_set = self.rough_set
        feat_matrix = rough_set.get_split_feat(2)

    def test_get_split_desc(self):
        roughs_set = self.rough_set
        desc_dict = roughs_set.get_split_desc()
        self.assertIsNotNone(desc_dict)
        self.assertIsInstance(desc_dict, dict)

    def test_get_reduced(self):
        feat = self.feat
        desc = self.desc

        # pre process
        nor_feat = normalize(feat)
        # check_variances_threshold(desc)

        rough_set = BaseRoughSet(nor_feat, desc)
        reduced = rough_set.get_reduced()
        self.assertEqual(reduced, {(0, 1): 5, (0, 1, 2): 5, (0,): 3, (1, 2): 5})

    def test_get_core(self):
        feat = self.feat
        desc = self.desc

        nor_feat = normalize(feat)

        rough_set = BaseRoughSet(nor_feat, desc)

        core = rough_set.get_core()
        self.assertEqual(core, [1])

    def test_remove_unrelated_attribute_and_core(self):
        feat = self.feat
        desc = self.desc

        nor_feat = normalize(feat)

        rough_set = BaseRoughSet(nor_feat, desc)

        remains = rough_set.remove_unrelated_attribute_and_core()
        self.assertEqual(remains, [0, 2])

    def test_get_reduced_from_core(self):
        feat = self.feat
        desc = self.desc

        nor_feat = normalize(feat)

        rough_set = BaseRoughSet(nor_feat, desc)

        remains = rough_set.get_reduced_from_core()
        self.assertEqual(remains, {(1, 2): 5, (1, 0): 5})

