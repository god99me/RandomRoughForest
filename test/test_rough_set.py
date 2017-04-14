import unittest

from core.Preprocessing import check_variances_threshold
from core.Preprocessing import normalize
from sklearn.datasets import load_iris


class TestRoughSet(unittest.TestCase):

    def setUp(self):
        pass