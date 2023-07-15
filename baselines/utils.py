import os.path as osp
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp


def gen_avail_mask(df):
    """ Fixed: replace str and negative_num as np.nan by a simpler method """
    df = df.apply(pd.to_numeric, errors='coerce')  # convert each element to numeric: if error, fill nan
    df[df < 0] = np.nan  # replace negative num with nan
    nan_mask = df.isna().values  # nan values
    avail_mask = np.logical_not(nan_mask)
    return avail_mask


def read_train_test_split(file_path):
    if isinstance(file_path, pd.DataFrame):
        gauge_df = file_path
    elif osp.exists(file_path):
        gauge_df = pd.read_csv(file_path)
    else:
        raise FileNotFoundError("'{}' does not exist!".format(file_path))

    is_test = gauge_df["is_test"].values
    train_mask = np.where(is_test == 0, True, False)
    test_mask = np.where(is_test == 1, True, False)
    valid_mask = np.where(is_test == 2, True, False)  # can delete this statement
    return train_mask, test_mask, valid_mask


def normalize(mx):
    """Row-normalize matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.  # set inf to zero
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

