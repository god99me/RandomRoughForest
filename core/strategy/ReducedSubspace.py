from core.strategy.SubspaceStrategy import SubspaceStrategy
import numpy as np


class ReducedSubspace(SubspaceStrategy):

    @staticmethod
    def get_subspace(shape):
        assert shape is not None
        rows, cols = shape
        assert isinstance(cols, (list, np.ndarray))
        row_idx = SubspaceStrategy.get_rand_rows_with_replacement(rows)
        # col_idx = list(range(cols))
        return row_idx, None
