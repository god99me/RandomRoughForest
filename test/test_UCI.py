import unittest
import core.UCI as UCI


class TestUCI(unittest.TestCase):

    def test_random_matrix(self):
        data = UCI.create_random_matrix(4, 3, 1)
        self.assertIsNotNone(data)

    def test_load_iris(self):
        iris_data = UCI.create_iris_datasets()
        self.assertIsNotNone(iris_data)
