import numpy as np
from unittest import TestCase
from core.strategy.RandomSubspace import RandomSubspace


class TestRandomSubspaceStrategy(TestCase):
    # TODO: Include more test case
    # (100, 100)
    # (100, 90)
    # (1, 1)
    # (0, 0)
    # (100, 100, 100)
    # (100)
    # None or any
    def test_get_space(self):
        row_idx, col_idx = RandomSubspace.get_subspace((100, 100))
        self.assertIsInstance(row_idx, (list, np.ndarray))
        self.assertIsInstance(col_idx, (list, np.ndarray))

        row_num = len(row_idx)
        col_num = len(col_idx)
        self.assertLessEqual(row_num, 100)
        self.assertGreaterEqual(row_num, 1)
        self.assertEquals(col_num, 10)
