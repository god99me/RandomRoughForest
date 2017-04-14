import unittest

import core.UCI as UCI

from core.Bagging import Bagging
from core.Bagging import Bag
from core.Preprocessing import check_variances_threshold
from core.Preprocessing import normalize


class TestBagging(unittest.TestCase):

    def setUp(self):
        data = UCI.create_random_matrix(4, 5, scale=10)
        check_variances_threshold(data)
        self.data = normalize(data)

    def test_bagging(self):
        bagging = Bagging(self.data)
        bag = bagging.get_bag()
        self.assertTrue(isinstance(bag, Bag))



