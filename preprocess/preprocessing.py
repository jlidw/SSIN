import numpy as np
import pandas as pd
from glob import glob
import pickle
import os
import argparse
import tqdm


""" This version, fixed the unnecessary redundancy when generating data.
For Training data:
    All sequence has the same length, stored in one np.ndarray;
    All sequence share one r_pos_mat.
For testing data, no need to generate gauge_list; only one timestamp is ok for each sequence.
An extra invalid_index is generated to indicate the positions of missing or invalid values in each sequence;
"""


def gen_invalid_mask_and_fill_nan(df):
    """ Fixed: replace str and negative_num as np.nan by a simpler method """
    df = df.apply(pd.to_numeric, errors='coerce')  # convert each element to numeric: if error, fill nan
    df[df < 0] = np.nan  # replace negative num with nan
    invalid_mask = df.isna().values  # nan values
    return invalid_mask, df.fillna(0)


def generate_train_data(info_path, avail_time_path, data_dir, col_name,
                        relative_pos_mat_path, out_dir, out_name):
    """used_feats: target rainfall values;  attr_feats: other needed radar features"""
    info_df = pd.read_csv(info_path)
    is_test = info_df["is_test"].values
    train_mask = np.where(is_test == 0, True, False)

    avail_timestamps = np.load(avail_time_path, allow_pickle=True)
    ori_r_pos_mat = np.load(relative_pos_mat_path)

    # Calculate the mean/std/max/min from the training data, for standardization/normalization
    stat_dict = generate_stat_from_train_data(info_path, relative_pos_mat_path)

    # Do standardization for position info
    info_df["lat"] = (info_df["lat"].values - stat_dict["lat_mean"]) / stat_dict["lat_std"]
    info_df["lon"] = (info_df["lon"].values - stat_dict["lon_mean"]) / stat_dict["lon_std"]
    ori_r_pos_mat[:, :, 0] = (ori_r_pos_mat[:, :, 0] - stat_dict["r_dist_mean"]) / stat_dict["r_dist_std"]
    ori_r_pos_mat[:, :, 1] = (ori_r_pos_mat[:, :, 1] - stat_dict["r_angle_mean"]) / stat_dict["r_angle_std"]

    train_indexes = np.where(train_mask)[0]
    idx_i, idx_j = np.ix_(train_indexes, train_indexes)
    r_pos_mat = ori_r_pos_mat[idx_i, idx_j, :]  # all training seqs share one relative position matrix

    file_paths = sorted(glob(data_dir))
    value_list, invalid_mask_list = [], []
    timestamp_list = []

    file_iter = tqdm.tqdm(enumerate(file_paths),
                               desc="Training data: ",
                               total=len(file_paths),
                               bar_format="{l_bar}{r_bar}")

    for i, _ in file_iter:
        _path = file_paths[i]

        timestamp = _path.split("/")[-1][:-4]
        if timestamp not in avail_timestamps:
            continue

        data_df = pd.read_csv(_path).rename(columns={'gauge': 'station'})
        assert np.all(data_df["station"].values == info_df["station"].values)

        train_df = data_df.loc[train_mask, :].reset_index(drop=True).copy()  # only fetch training nodes
        invalid_mask, train_df[col_name] = gen_invalid_mask_and_fill_nan(train_df[col_name])
        train_df[col_name] = train_df[col_name].astype(float)

        values = train_df[[col_name]].values
        if values.ndim == 1:
            values = np.expand_dims(values, axis=1)

        value_list.append(values)
        invalid_mask_list.append(invalid_mask)
        timestamp_list.append(timestamp)  # timestamp for each training seq

    print("Data length", len(value_list))

    data_dict = {}
    data_dict["train_data"] = np.array(value_list)
    data_dict["invalid_masks"] = np.array(invalid_mask_list)
    data_dict["timestamps"] = np.array(timestamp_list)
    data_dict["r_pos_mat"] = r_pos_mat

    out_dir = f"{out_dir}/train"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{out_name}.pkl", "wb") as fp:
        pickle.dump(data_dict, fp)


def generate_stat_from_train_data(info_path, relative_pos_mat_path):
    info_df = pd.read_csv(info_path)
    ori_r_pos_mat = np.load(relative_pos_mat_path)

    is_test = info_df["is_test"].values
    train_mask = np.where(is_test == 0, True, False)
    train_info_df = info_df.loc[train_mask, :]

    lat_mean, lat_std, lat_max, lat_min = train_info_df["lat"].mean(), train_info_df["lat"].std(ddof=0), \
                                          train_info_df["lat"].max(), train_info_df["lat"].min()
    lon_mean, lon_std, lon_max, lon_min = train_info_df["lon"].mean(), train_info_df["lon"].std(ddof=0), \
                                          train_info_df["lon"].max(), train_info_df["lon"].min()

    indexes = np.where(train_mask)[0]
    idx_i, idx_j = np.ix_(indexes, indexes)
    r_pos_mat = ori_r_pos_mat[idx_i, idx_j, :]

    r_dist_mat = r_pos_mat[:, :, 0]
    r_angle_mat = r_pos_mat[:, :, 1]

    r_dist_mean, r_dist_std, r_dist_max, r_dist_min = np.mean(r_dist_mat), np.std(r_dist_mat), \
                                                      np.max(r_dist_mat), np.min(r_dist_mat),
    r_angle_mean, r_angle_std, r_angle_max, r_angle_min = np.mean(r_angle_mat), np.std(r_angle_mat), \
                                                          np.max(r_angle_mat), np.min(r_angle_mat)

    stat_dict = {}
    stat_dict["lat_mean"], stat_dict["lat_std"], stat_dict["lat_max"], stat_dict["lat_min"] = \
        lat_mean, lat_std, lat_max, lat_min
    stat_dict["lon_mean"], stat_dict["lon_std"], stat_dict["lon_max"], stat_dict["lon_min"] = \
        lon_mean, lon_std, lon_max, lon_min

    stat_dict["r_dist_mean"], stat_dict["r_dist_std"], stat_dict["r_dist_max"], stat_dict["r_dist_min"] = \
        r_dist_mean, r_dist_std, r_dist_max, r_dist_min
    stat_dict["r_angle_mean"], stat_dict["r_angle_std"], stat_dict["r_angle_max"], stat_dict["r_angle_min"] = \
        r_angle_mean, r_angle_std, r_angle_max, r_angle_min

    print("Calculates the statistics of training data. Done!")

    # with open("./data/hk_data_stats.pkl".format(out_name), "wb") as fp:
    #     pickle.dump(stat_dict, fp)

    return stat_dict


def generate_test_data(info_path, avail_time_path, data_dir, col_name,
                       relative_pos_mat_path, out_dir, out_name):
    info_df = pd.read_csv(info_path)
    is_test = info_df["is_test"].values
    test_mask = np.where(is_test == 1, True, False)

    avail_timestamps = np.load(avail_time_path, allow_pickle=True)
    ori_r_pos_mat = np.load(relative_pos_mat_path)

    # Calculate the mean/std/max/min from the training data, for standardization/normalization
    stat_dict = generate_stat_from_train_data(info_path, relative_pos_mat_path)

    # Do standardization for position info
    info_df["lat"] = (info_df["lat"].values - stat_dict["lat_mean"]) / stat_dict["lat_std"]
    info_df["lon"] = (info_df["lon"].values - stat_dict["lon_mean"]) / stat_dict["lon_std"]
    ori_r_pos_mat[:, :, 0] = (ori_r_pos_mat[:, :, 0] - stat_dict["r_dist_mean"]) / stat_dict["r_dist_std"]
    ori_r_pos_mat[:, :, 1] = (ori_r_pos_mat[:, :, 1] - stat_dict["r_angle_mean"]) / stat_dict["r_angle_std"]

    file_paths = sorted(glob(data_dir))
    value_list, invalid_mask_list, valid_test_mask_list = [], [], []
    gauge_list, timestamp_list = [], []

    file_iter = tqdm.tqdm(enumerate(file_paths),
                          desc="Testing data: ",
                          total=len(file_paths),
                          bar_format="{l_bar}{r_bar}")

    for i, _ in file_iter:
        _path = file_paths[i]

        timestamp = _path.split("/")[-1][:-4]
        if timestamp not in avail_timestamps:
            continue

        data_df = pd.read_csv(_path).rename(columns={'gauge': 'station'})
        assert np.all(data_df["station"].values == info_df["station"].values)

        invalid_mask, data_df[col_name] = gen_invalid_mask_and_fill_nan(data_df[col_name])
        data_df[col_name] = data_df[col_name].astype(float)

        valid_mask = np.logical_not(invalid_mask)
        valid_test_mask = np.logical_and(valid_mask, test_mask)  # only get valid test nodes

        values = data_df[[col_name]].values
        if values.ndim == 1:
            values = np.expand_dims(values, axis=1)

        # get testing timestamps and gauge names
        # gauges, timestamps = get_gauge_timestamp_from_data(avail_df, idx_test, timestamp)

        value_list.append(values)
        invalid_mask_list.append(invalid_mask)
        valid_test_mask_list.append(valid_test_mask)
        # gauge_list.append(gauges)
        timestamp_list.append(timestamp)  # timestamps for each test node

    print("Data length", len(value_list))

    data_dict = {}
    data_dict["test_data"] = np.array(value_list)
    data_dict["invalid_masks"] = np.array(invalid_mask_list)
    data_dict["test_masks"] = np.array(valid_test_mask_list)
    data_dict["r_pos_mat"] = ori_r_pos_mat
    # data_dict["gauges"] = gauge_list
    data_dict["timestamps"] = np.array(timestamp_list)

    out_dir = f"{out_dir}/test"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{out_name}.pkl", "wb") as fp:
        pickle.dump(data_dict, fp)


def get_gauge_timestamp_from_data(data_df, idx_test, timestamp):
    gauges = data_df["station"].values[idx_test]
    timestamps = np.array([timestamp]).repeat(len(idx_test))

    return gauges, timestamps


if __name__ == "__main__":
    base_dir = "../data"
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="hk")
    parser.add_argument('--generate_type', type=str, default="training")
    parser.add_argument('--start_year', type=int, default=2008)
    parser.add_argument('--end_year', type=int, default=2012)
    args = parser.parse_args()

    dataset = args.dataset
    start_year = args.start_year
    end_year = args.end_year

    if dataset.lower() == "bw":
        dataset_dir = f"{base_dir}/BW_132_data"
        info_path = f"{dataset_dir}/BW_stations_info.csv"
        dist_mat_path = f"{dataset_dir}/dist_angle_mat.npy"
        data_dir = f"{dataset_dir}/rain_csv/*/*.csv"
        out_dir = f"{dataset_dir}/pkl_data"
        col_name = "rainfall"  # col name for used data
    elif dataset.lower() == "hk":
        dataset_dir = f"{base_dir}/HK_123_data"
        info_path = f"{dataset_dir}/hko_stations_info.csv"
        dist_mat_path = f"{dataset_dir}/dist_angle_mat.npy"
        data_dir = f"{dataset_dir}/rain_csv/*/*.csv"
        out_dir = f"{dataset_dir}/pkl_data"
        col_name = "rainfall"  # col name for used data
    else:
        raise NotImplementedError

    assert isinstance(start_year, int)
    assert isinstance(end_year, int)
    assert start_year <= end_year
    if start_year == end_year:
        avail_time_path = f"{dataset_dir}/{start_year}_avail_used_timestamps.npy"
        out_name = f"{start_year}_data"
    else:
        avail_time_path = f"{dataset_dir}/{start_year}-{end_year}_avail_used_timestamps.npy"
        out_name = f"{start_year}-{end_year}_data"

    os.makedirs(out_dir, exist_ok=True)

    if args.generate_type == "training":
        generate_train_data(info_path, avail_time_path, data_dir, col_name, dist_mat_path, out_dir, out_name)
    elif args.generate_type == "testing":
        generate_test_data(info_path, avail_time_path, data_dir, col_name, dist_mat_path, out_dir, out_name)
