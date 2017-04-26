import unittest

import core.UCI as UCI
import numpy as np
from core.Bagging import Bagging
from core.Preprocessing import check_variances_threshold
from core.Preprocessing import normalize


class TestBagging(unittest.TestCase):

    def setUp(self):
        iris = UCI.create_iris_data_sets()
        self.data = np.concatenate((normalize(iris.data), iris.target.reshape(150, 1)), axis=1)

    def test_bagging(self):
        bagging = Bagging(self.data, "ReducedSubspace")
        bags = bagging.get_bags()
        print(bags)





