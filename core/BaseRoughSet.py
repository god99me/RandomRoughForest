import random
import numpy as np
import pandas as pd


def partition(column, radius=0):
    n_rows = len(column)
    is_numeric = column.dtype.name.find('float') != -1

    cluster = [[] for i in range(n_rows)]

    if is_numeric:
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


def partition_all(columns, radius=0):
    columns = pd.DataFrame(columns)
    n_rows, n_cols = columns.shape
    result = partition(columns.iloc[:, 0], radius)

    for i in range(1, n_cols):
        transfer = partition(columns.iloc[:, i], radius)
        result = pd.concat([result, transfer], axis=1)

    # recover columns index, dummy fix
    result.columns = range(n_cols)
    return result


def calc_core(data):
    n_rows, n_cols = data.shape
    core = set()
    base_pos_set = calc_pos_set(data)
    base = len(base_pos_set)

    for i in range(n_cols - 1):
        df_temp = data.drop(i, axis=1)
        # in case index label wrong
        assert data.shape[1] - df_temp.shape[1] == 1
        temp_pos_set = calc_pos_set(df_temp)
        temp_len = len(temp_pos_set)

        # if temp_len < base:
        #     core.add(i)

        if base / n_rows - temp_len / n_rows > 0.01:
            core.add(i)

    return core


def calc_feat_attr_dependency_without_one_feat(data):
    rows, cols = data.shape
    dependencies = []
    all_pos_set = calc_pos_set(data)
    all_dependency = calc_attr_dependency(len(all_pos_set), rows)
    # dependencies.append(all_dependency)

    for i in range(cols - 1):
        df_temp = data.drop(i, axis=1)
        # in case index label wrong
        assert data.shape[1] - df_temp.shape[1] == 1
        temp_pos_set = calc_pos_set(df_temp)
        dependency = calc_attr_dependency(len(temp_pos_set), rows)
        dependencies.append(dependency)

    return all_dependency - np.array(dependencies)


def calc_pos_set(data):
    rows, cols = data.shape
    pos_set = set()
    if cols < 2:
        # raise Exception("columns number should larger than 1")
        return pos_set

    for i in range(rows):
        temp = set()
        for j in range(cols - 1):
            if len(temp) == 0:
                temp = set(data.iloc[i, j])
            temp &= set(data.iloc[i, j])
        if temp.issubset(data.iloc[i, cols - 1]):
            pos_set |= temp

    return pos_set


def calc_each_feature_attr_dependency(data):
    n_rows, n_cols = data.shape

    attrs = []
    for i in range(n_cols - 1):
        temp_df = pd.concat([data.iloc[:, i], data.iloc[:, n_cols - 1]], axis=1)
        pos_set = calc_pos_set(temp_df)
        attr_dependency = calc_attr_dependency(len(pos_set), n_rows)
        attrs.append(attr_dependency)

    return attrs


def calc_red(data, core, core_pos_set, threshold=0, shuffle=True):
    n_rows, n_cols = data.shape
    remained = list(set(range(n_cols - 1)).difference(core))
    red = list(core)

    # core_dependency = BaseRoughSet.calc_attr_dependency(len(core_pos_set), n_rows)
    base = len(core_pos_set)

    if shuffle:
        random.shuffle(remained)

    for i in remained:
        red.append(i)
        pos_set = calc_pos_set(data.iloc[:, red])
        if len(pos_set) - base < threshold:
            red.remove(i)
            continue
        base = len(pos_set)

    return set(red)


def calc_attr_dependency(pos_set_size, all_size):
    return pos_set_size / all_size


def calc_sig():
    pass


def float_range(start, stop, steps):
    '''Computes a range of floating value.

        Input:
            start (float)  : Start value.
            end   (float)  : End value
            steps (integer): Number of values

        Output:
            A list of floats

        Example:
            >>> print float_range(0.25, 1.3, 5)
            [0.25, 0.51249999999999996, 0.77500000000000002, 1.0375000000000001, 1.3]
    '''
    return [start + float(i) * (stop - start) / (float(steps) - 1) for i in range(steps)]


