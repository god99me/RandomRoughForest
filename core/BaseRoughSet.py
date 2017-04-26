import numpy as np
from functools import reduce


class BaseRoughSet(object):

    def __init__(self, attr, desc):
        self.attr = attr
        self.desc = desc
        self.pos = []
        self.attr_length = len(self.attr[0])
        self.desc_dict = self.get_split_desc()
        self.split_matrix = self.get_split_feat()
        self.max_dependency = 0
        self.core = self.set_core()

    def get_split_feat(self, lam=2):
        attr_data = self.attr
        row_num, col_num = attr_data.shape
        delta = np.std(attr_data, axis=0) / lam

        matrix_3d = []
        for i in range(row_num):
            matrix_3d.append([])
            for j in range(col_num):
                matrix_3d[i].append([])

        for j in range(col_num):
            for i in range(row_num):
                for k in range(i, row_num):
                    dist = abs(attr_data[k][j] - attr_data[i][j])
                    if dist < delta[j]:
                        matrix_3d[i][j].append(k)
                        if i != k:
                            matrix_3d[k][j].append(i)

        return matrix_3d

    def get_split_desc(self):
        desc_dict = {}

        for k, v in enumerate(self.desc):
            if v not in desc_dict.keys():
                desc_dict[v] = []
            desc_dict[v].append(k)

        return desc_dict

    def get_pos_set(self, cols):
        data = np.array(self.split_matrix)

        if len(cols) == 1:
            n = data[:, cols[0]]
        else:
            n = []
            for row_idx in range(len(data)):
                # deep copy
                n.append(reduce(set.intersection, map(set, data[row_idx, cols])))

        pos_set = set()
        for row in n:
            if self._is_subset(row):
                pos_set |= set(row)
        return np.array(n), len(pos_set)

    def _is_subset(self, row):
        row = set(row)

        for value in self.desc_dict.values():
            if row.issubset(value):
                return True

        return False

    def set_core(self):
        num = self.attr_length
        core = []

        # all condition attributes
        total_pos_set, total_dependency = self.get_pos_set(np.array(list(range(num))))

        for idx in range(num):
            cols = np.delete(np.array(list(range(num))), idx)
            pos_set, dependency = self.get_pos_set(cols)
            if dependency != total_dependency:
                core.append(idx)
        return np.array(core)

    def get_core(self):
        return self.core.copy()

    def remove_unrelated_attribute_and_core(self):
        core = self.get_core()  # deep copy
        remains = list(set(list(range(self.attr_length))) ^ set(core))
        base_dependency = self.get_pos_set(core)[1]

        for remain in remains:
            cols = list(core.copy())
            cols.append(remain)
            if self.get_pos_set(cols) == base_dependency:
                remains.remove(remain)

        return remains

    def get_reduced(self):

        result = {}
        subset = []
        nums = self.attr_length
        self._subsets_helper(0, nums, subset, result)
        return result

    def get_reduced_from_core(self):
        reduct = self.core.copy()
        result = {}
        remains = self.remove_unrelated_attribute_and_core()
        base_dependency = self.get_pos_set(reduct)[1]
        pre_remains_length = -1

        while pre_remains_length != len(remains):
            reduct = list(self.core.copy())
            dependency = base_dependency
            pre_remains_length = len(remains)
            for remain in remains:
                reduct.append(remain)
                new_dependency = self.get_pos_set(reduct)[1]
                if new_dependency > dependency:
                    dependency = new_dependency
                    remains.remove(remain)
                else:
                    reduct.pop()
            if dependency > base_dependency:  # If there is a case where reduct happens to equal to core
                result[tuple(reduct)] = dependency
        return result

    def _subsets_helper(self, pos, nums, subset, result):

        for i in range(pos, nums):

            subset.append(i)

            inter_sect_row, dependency = self.get_pos_set(subset)
            if dependency >= self.max_dependency:
                self.max_dependency = dependency
                k = tuple(subset)
                result[k] = dependency

            self._subsets_helper(i + 1, nums, subset, result)
            subset.pop()


