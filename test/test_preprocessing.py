import unittest

import core.UCI as UCI

from core.Preprocessing import check_variances_threshold
from core.Preprocessing import normalize


class TestPreproces(unittest.TestCase):

    def setUp(self):
        self.data = UCI.create_random_matrix(4, 5, scale=10)
        self.assertIsNotNone(self.data)

    def test_variances_threshold(self):
        is_validate = check_variances_threshold(self.data, .3)
        self.assertTrue(is_validate)

    def test_normalize(self):
        self.assertIsNotNone(normalize(self.data))




