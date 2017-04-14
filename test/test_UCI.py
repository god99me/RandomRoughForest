import unittest
from core.UCI import create_random_matrix


class TestUCI(unittest.TestCase):

    def test_random_matrix(self):
        data = create_random_matrix(4, 3, 1)
        self.assertIsNotNone(data)
