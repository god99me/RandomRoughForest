from core.strategy.SubspaceStrategy import SubspaceStrategy


class RandomSubspace(SubspaceStrategy):

    @staticmethod
    def get_subspace(shape):
        rows, cols = shape
        row_idx = SubspaceStrategy.get_rand_rows_with_replacement(rows)
        col_idx = SubspaceStrategy.get_sqrt_n_rand_cols(cols)
        return row_idx, col_idx
