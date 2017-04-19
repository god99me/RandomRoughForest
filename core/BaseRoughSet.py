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

    def get_split_feat(self, lammba=2):
        attr_data = self.attr
        row_num, col_num = attr_data.shape
        delta = np.std(attr_data, axis=0) / lammba

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
                n.append(reduce(set.intersection, map(set, data[row_idx, cols])))
            # n = reduce(set.intersection, map(set, data[:, cols]))

        pos_set = set()
        for row in n:
            if self._is_subset(row):
                # pos_set.add(row)
                pos_set |= set(row)
        return np.array(n), len(pos_set)

    def _is_subset(self, row):
        row = set(row)

        for value in self.desc_dict.values():
            if row.issubset(value):
                return True

        return False

    def get_reduced(self):
        result = {}
        subset = []
        nums = self.attr_length
        stack = []
        self._subsets_helper(0, nums, subset, result)
        return result

    def _subsets_helper(self, pos, nums, subset, result):

        for i in range(pos, nums):

            subset.append(i)

            inter_sect_row, dependency = self.get_pos_set(subset)
            if dependency >= self.max_dependency:
                self.max_dependency = dependency
                if dependency not in result:
                    result[dependency] = []
                result[dependency].append(subset.copy())

            self._subsets_helper(i + 1, nums, subset, result)
            subset.pop()


