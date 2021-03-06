import numpy as np
import pandas as pd


def check_variances_threshold(data, threshold=0.):
    variances = np.var(data, axis=0)
    if np.any(variances < threshold):
        msg = "Some feature doesn't meet the threshold {0:.5f}"
        raise ValueError(msg.format(threshold))
    return True


def normalize_all(data):
    # In case raise divide 0 error
    check_variances_threshold(data)

    min_tuple = data.min(axis=0)
    max_tuple = data.max(axis=0)
    return (data - min_tuple) / (max_tuple - min_tuple)


def normalize(data):
    df = pd.DataFrame(data)
    return (df - df.min()) / (df.max() - df.min())





