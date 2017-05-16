import numpy as np
import pandas as pd


class BaseRoughSet(object):

    @staticmethod
    def partition(column, is_numeric, radius=0):
        n_rows = len(column)

        cluster = [[] for i in range(n_rows)]

        if not is_numeric:
            std_col = np.std(column)
            radius = std_col / radius
        else:
            radius = 0

        for i in range(n_rows):
            base = column[i]

            for j in range(i, n_rows):
                if abs(column[j] - base) > radius:
                    continue
                if i != j:
                    cluster[i].append(j)
                cluster[j].append(i)

            cluster[i] = tuple(cluster[i])

        return pd.Series(cluster)

    @staticmethod
    def partition_all(columns, types, radius=0):
        columns = pd.DataFrame(columns)
        n_rows, n_cols = columns.shape
        result = BaseRoughSet.partition(columns.iloc[:, 0], types[0], radius)

        for i in range(1, n_cols):
            transfer = BaseRoughSet.partition(columns.iloc[:, i], types[i], radius)
            result = pd.concat([result, transfer], axis=1)

        # recover columns index, dummy fix
        result.columns = range(n_cols)
        return result

    @staticmethod
    def calc_core(data):
        rows, cols = data.shape
        core = set()
        base_pos_set = BaseRoughSet.calc_pos_set(data)
        base = len(base_pos_set)

        for i in range(cols - 1):
            df_temp = data.drop(i, axis=1)
            # in case index label wrong
            assert data.shape[1] - df_temp.shape[1] == 1
            temp_pos_set = BaseRoughSet.calc_pos_set(df_temp)
            temp_len = len(temp_pos_set)

            if temp_len < base:
                core.add(i)

        return core

    @staticmethod
    def calc_feat_attr_dependency_without_one_feat(data):
        rows, cols = data.shape
        dependencies = []
        all_pos_set = BaseRoughSet.calc_pos_set(data)
        all_dependency = BaseRoughSet.calc_attr_dependency(len(all_pos_set), rows)
        dependencies.append(all_dependency)

        for i in range(cols - 1):
            df_temp = data.drop(i, axis=1)
            # in case index label wrong
            assert data.shape[1] - df_temp.shape[1] == 1
            temp_pos_set = BaseRoughSet.calc_pos_set(df_temp)
            dependency = BaseRoughSet.calc_attr_dependency(len(temp_pos_set), rows)
            dependencies.append(dependency)

        return dependencies

    @staticmethod
    def calc_pos_set(data):
        rows, cols = data.shape
        if cols < 2:
            raise Exception("columns number should larger than 1")
        pos_set = set()

        for i in range(rows):
            temp = set()
            for j in range(cols - 1):
                if len(temp) == 0:
                    temp = set(data.iloc[i, j])
                temp &= set(data.iloc[i, j])
            if temp.issubset(data.iloc[i, cols - 1]):
                pos_set |= temp

        return pos_set

    @staticmethod
    def calc_each_feature_attr_dependency(data):
        n_rows, n_cols = data.shape

        attrs = []
        for i in range(n_cols - 1):
            temp_df = pd.concat([data.iloc[:, i], data.iloc[:, n_cols - 1]], axis=1)
            pos_set = BaseRoughSet.calc_pos_set(temp_df)
            attr_dependency = BaseRoughSet.calc_attr_dependency(len(pos_set), n_rows)
            attrs.append(attr_dependency)

        return attrs

    @staticmethod
    def calc_red():
        pass

    @staticmethod
    def calc_attr_dependency(pos_set_size, all_size):
        return pos_set_size / all_size

    @staticmethod
    def calc_sig():
        pass
