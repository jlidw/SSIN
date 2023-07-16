import os
import os.path as osp
import numpy as np
import pandas as pd
import zipfile
from glob import glob
import datetime
import argparse
import tqdm
import pickle
import h5py


def generate_train_test_split(info_path, dataset, out_dir, test_rate=0.2, seed=42):
    df = pd.read_csv(info_path)

    num_station = len(df)
    test_num = round(num_station * test_rate)  # randomly select 20%, add to test set
    print("Dataset:", dataset, "total station num:", num_station, "test num:", test_num)

    is_test = np.zeros(len(df)).astype(int).tolist()
    df["is_test"] = is_test  # 0: train, 1: test; -1: invalid

    test_idx = df[df["is_test"] == 0].sample(n=test_num, random_state=seed).index.tolist()
    df.loc[test_idx, "is_test"] = 1

    print(df["is_test"].sum())

    out_path = "{}/bay_stations_info.csv".format(out_dir)
    df.to_csv(out_path)


def generate_train_data(data_path, info_path, relative_pos_mat_path, adj_attn_mask_path, out_dir, out_name):
    df = pd.read_hdf(data_path)
    data_arr = df.values  # [num_samples, num_station]
    if data_arr.ndim == 2:
        data_arr = np.expand_dims(data_arr, axis=2)

    info_df = pd.read_csv(info_path)
    is_test = info_df["is_test"].values
    train_mask = np.where(is_test == 0, True, False)

    ori_r_pos_mat = np.load(relative_pos_mat_path)
    adj_attn_mask = np.load(adj_attn_mask_path)  # traffic data is not fully-connected; have additional attn_mask.

    # Calculate the mean/std/max/min from the training data, for standardization/normalization
    stat_dict = generate_stat_from_train_data(info_path, relative_pos_mat_path)

    # Do standardization for position info
    ori_r_pos_mat = (ori_r_pos_mat - stat_dict["r_dist_mean"]) / stat_dict["r_dist_std"]

    train_indexes = np.where(train_mask)[0]
    idx_i, idx_j = np.ix_(train_indexes, train_indexes)
    r_pos_mat = ori_r_pos_mat[idx_i, idx_j, :]  # all training seqs share one relative position matrix
    adj_attn_mask = adj_attn_mask[idx_i, idx_j]
    print(adj_attn_mask)

    data_arr = data_arr[:, train_indexes, :]
    num_samples = data_arr.shape[0]
    print("Data length", num_samples)
    timestamp_arr = df.index.values

    data_dict = {}
    data_dict["train_data"] = data_arr
    data_dict["invalid_masks"] = None
    data_dict["timestamps"] = timestamp_arr
    data_dict["r_pos_mat"] = r_pos_mat
    data_dict["adj_attn_mask"] = adj_attn_mask

    out_dir = f"{out_dir}/pkl_data/train"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{out_name}.pkl", "wb") as fp:
        pickle.dump(data_dict, fp)


def generate_stat_from_train_data(info_path, relative_pos_mat_path):
    info_df = pd.read_csv(info_path)
    ori_r_pos_mat = np.load(relative_pos_mat_path)

    is_test = info_df["is_test"].values
    train_mask = np.where(is_test == 0, True, False)
    train_info_df = info_df.loc[train_mask, :]

    indexes = np.where(train_mask)[0]
    idx_i, idx_j = np.ix_(indexes, indexes)
    r_dist_mat = ori_r_pos_mat[idx_i, idx_j, :]

    r_dist_mean, r_dist_std, r_dist_max, r_dist_min = np.mean(r_dist_mat), np.std(r_dist_mat), \
                                                      np.max(r_dist_mat), np.min(r_dist_mat),

    stat_dict = {}
    stat_dict["r_dist_mean"], stat_dict["r_dist_std"], stat_dict["r_dist_max"], stat_dict["r_dist_min"] = \
        r_dist_mean, r_dist_std, r_dist_max, r_dist_min
    print("Calculates the statistics of training data. Done!")

    # with open("./data/hk_data_stats.pkl".format(out_name), "wb") as fp:
    #     pickle.dump(stat_dict, fp)

    return stat_dict


def generate_test_data(data_path, info_path, relative_pos_mat_path, adj_attn_mask_path, out_dir, out_name):
    df = pd.read_hdf(data_path)
    data_arr = df.values  # [num_samples, num_station]
    if data_arr.ndim == 2:
        data_arr = np.expand_dims(data_arr, axis=2)

    info_df = pd.read_csv(info_path)
    is_test = info_df["is_test"].values
    test_mask = np.where(is_test == 1, True, False)

    ori_r_pos_mat = np.load(relative_pos_mat_path)
    adj_attn_mask = np.load(adj_attn_mask_path)
    print(adj_attn_mask)

    # Calculate the mean/std/max/min from the training data, for standardization/normalization
    stat_dict = generate_stat_from_train_data(info_path, relative_pos_mat_path)

    # Do standardization for position info
    ori_r_pos_mat = (ori_r_pos_mat - stat_dict["r_dist_mean"]) / stat_dict["r_dist_std"]

    num_samples = data_arr.shape[0]
    print("Data length", num_samples)
    timestamp_arr = df.index.values

    data_dict = {}
    data_dict["test_data"] = data_arr
    data_dict["invalid_masks"] = None
    data_dict["test_masks"] = test_mask  # only one 2D matrix since NO invalid nodes
    data_dict["r_pos_mat"] = ori_r_pos_mat
    data_dict["adj_attn_mask"] = adj_attn_mask
    data_dict["timestamps"] = timestamp_arr

    out_dir = f"{out_dir}/pkl_data/test"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{out_name}.pkl", "wb") as fp:
        pickle.dump(data_dict, fp)


def generate_test_index_set(info_path, out_path):
    info_df = pd.read_csv(info_path)
    is_test = info_df["is_test"].values
    test_mask = np.where(is_test == 1, True, False)
    test_indexes = np.where(test_mask)[0]

    print("Test node num: ", len(test_indexes))
    print("test_indexes:", test_indexes)
    np.save(out_path, test_indexes)


if __name__ == "__main__":
    base_dir = "../data"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="bay")
    parser.add_argument('--generate_type', type=str, default="training")
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    dataset = args.dataset
    seed = args.seed

    if dataset == "bay":
        data_path = f"{base_dir}/PEMS-BAY/raw_data/pems-bay.h5"
        info_path = f"{base_dir}/PEMS-BAY/bay_stations_info.csv"
        relative_pos_mat_path = f"{base_dir}/PEMS-BAY/dist_mat.npy"
        adj_attn_mask_path = f"{base_dir}/PEMS-BAY/adj_attn_mask.npy"
        out_dir = f"{base_dir}/PEMS-BAY"
    else:
        raise NotImplementedError

    # generate_train_test_split(info_path, dataset, out_dir, test_rate=0.2, seed=seed)

    out_name = f"bay_data"
    if args.generate_type == "training":
        generate_train_data(data_path, info_path, relative_pos_mat_path, adj_attn_mask_path, out_dir, out_name)
    elif args.generate_type == "testing":
        generate_test_data(data_path, info_path, relative_pos_mat_path, adj_attn_mask_path, out_dir, out_name)

    # out_path = f"{out_dir}/test_index.npy"
    # generate_test_index_set(info_path, out_path)


