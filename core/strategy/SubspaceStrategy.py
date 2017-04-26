import math
import random


class SubspaceStrategy(object):

    @staticmethod
    def get_subspace(shape):
        raise NotImplementedError("The SubspaceStrategy is supposed to be interface or abstract class")

    @staticmethod
    def get_rand_rows_with_replacement(rows):
        loop_times = rows
        row_list = set()
        for i in range(loop_times):
            row_list.add(random.randint(0, loop_times - 1))
        row_idx = list(row_list)
        return row_idx

    @staticmethod
    def get_sqrt_n_rand_cols(cols):
        col_list = random.sample(range(0, cols - 1), math.ceil(math.sqrt(cols)))
        col_idx = sorted(col_list)
        return col_idx
