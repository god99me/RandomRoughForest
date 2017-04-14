import numpy as np


class BaseRoughSet(object):

    def __init__(self, attr, desc):
        self.attr = attr
        self.desc = desc

    def get_split_feat(self, lammba):
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

    def get_post_set(self):
        pass

    def get_reduced(self):
        pass
