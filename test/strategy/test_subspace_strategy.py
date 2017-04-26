import numpy as np
from unittest import TestCase
from core.strategy.SubspaceStrategy import SubspaceStrategy


class TestSubspaceStrategy(TestCase):

    def test_get_subspace(self):
        try:
            SubspaceStrategy.get_subspace((100, 100))
        except NotImplementedError as e:
            self.assertIsNotNone(e)

    def test_get_rand_rows_with_replacement(self):
        rows = SubspaceStrategy.get_rand_rows_with_replacement(100)
        self.assertIsInstance(rows, (np.ndarray, list))

    def test_get_sqrt_n_rand_cols(self):
        cols = SubspaceStrategy.get_sqrt_n_rand_cols(100)
        self.assertIsInstance(cols, (np.ndarray, list))

