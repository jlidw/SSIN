import os
import os.path as osp
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from pykrige.ok import OrdinaryKriging
import argparse
import tqdm
import sys
sys.path.append('../..')
from SSIN.baselines.utils import read_train_test_split, gen_avail_mask, normalize


# --------------------------------- Inverse Distance Weighting (IDW) Method -------------------------------- #
def calc_idw(timestamp, df, col_name, train_mask, test_mask, dist_mat, power_p=2):
    labels = df[col_name].values

    idx_train = np.where(train_mask)[0]
    idx_test = np.where(test_mask)[0]

    calib_values = labels.copy()[idx_train]
    idx_i, idx_j = np.ix_(idx_test, idx_train)

    idw_mat = dist_mat[idx_i, idx_j]  # one row denotes one test node and its neighbors (training nodes)
    idw_mat = normalize(np.power(idw_mat, power_p))  # normalize by rows: 1/(d^2)

    predict = idw_mat.dot(calib_values.astype(float))

    out_df = pd.DataFrame()
    out_df["timestamp"] = np.array([timestamp]).repeat(len(idx_test))
    out_df["station"] = df["station"].values[idx_test]
    out_df[col_name] = labels[idx_test]
    out_df["pred"] = predict

    return out_df


def main_idw(info_path, data_dir, col_name, out_dir, suffix, in_dist_path, avail_time_path, data_year, power_p=2):
    out_dir = "{}/IDW".format(out_dir)
    in_dist_mat = np.load(in_dist_path)

    os.makedirs(out_dir, exist_ok=True)
    ret_path = "{}/{}.csv".format(out_dir, suffix)
    if os.path.exists(ret_path):
        return None

    graph_path_list = []
    for y in data_year:
        data_path = osp.join(data_dir, str(y), "*.csv")
        graph_path_list += list(glob(data_path))

    avail_timestamps = np.load(avail_time_path, allow_pickle=True)

    gauge_df = pd.read_csv(info_path)
    is_test = gauge_df["is_test"].values

    file_iter = tqdm.tqdm(enumerate(graph_path_list),
                          desc="Training data: ",
                          total=len(graph_path_list),
                          bar_format="{l_bar}{r_bar}")

    out_df_list = []
    for i, _ in file_iter:
        graph_path = graph_path_list[i]
        timestamp = graph_path.split("/")[-1][:-4]

        if timestamp not in avail_timestamps:
            continue

        df = pd.read_csv(graph_path)  # data df
        station_col = df.columns.values[0]
        df = df[[station_col, col_name]]
        df.rename(columns={station_col: "station"}, inplace=True)

        df["is_test"] = gauge_df["is_test"]  # add the data_split identifier as a column

        select_mask = np.where(is_test != -1, 1, 0)  # 0: train, 1: test; -1: unselected
        avail_mask = gen_avail_mask(df[col_name])  # remove nodes with invalid values

        used_mask = np.logical_and(select_mask, avail_mask)
        used_idx = np.where(used_mask)[0]

        idx_i, idx_j = np.ix_(used_idx, used_idx)
        _dist_mat = in_dist_mat[idx_i, idx_j]  # delete unselected/invalid nodes from distance matrix

        df = df.iloc[used_idx].reset_index(drop=True)  # remove unselected and invalid data
        df[col_name] = df[col_name].astype(float)
        train_mask, test_mask, val_mask = read_train_test_split(df)

        out_df = calc_idw(timestamp, df, col_name, train_mask, test_mask, _dist_mat, power_p=power_p)
        out_df_list.append(out_df)

    tot_out_df = pd.concat(out_df_list, axis=0, ignore_index=True)
    print("Timestamp Num: ", len(set(tot_out_df["timestamp"].values)))

    tot_out_df.to_csv(ret_path)


# ----------------------------------- Ordinary Kriging (OK) Method ---------------------------------- #
def kriging_algo(timestamp, df_path, col_name, coord_df, train_mask, test_mask, mode="OK", variogram="spherical"):
    if mode == "OK":
        df = pd.read_csv(df_path)
        station_col = df.columns.values[0]
        df = df[[station_col, col_name]]
        df.rename(columns={station_col: "station"}, inplace=True)
    else:
        raise NotImplementedError

    data_arr = pd.merge(coord_df, df, how='inner', on=['station']).iloc[:, 1:].values  # lat, lon, rainfall, 3_mean, ...

    avail_mask = gen_avail_mask(df[col_name])
    train_mask = np.logical_and(train_mask, avail_mask)
    test_mask = np.logical_and(test_mask, avail_mask)

    df.loc[np.logical_not(avail_mask), col_name] = 0  # set invalid value to zero
    df[col_name] = df[col_name].astype(float)
    labels = df[col_name].values.copy()

    idx_train = np.where(train_mask)[0]
    idx_test = np.where(test_mask)[0]

    train_data = data_arr[idx_train].astype(float)
    test_data = data_arr[idx_test].astype(float)

    if mode == "OK":
        # x: longitude,  y: latitude, z: value
        OK = OrdinaryKriging(train_data[:, 1], train_data[:, 0], train_data[:, 2],
                             variogram_model=variogram,
                             coordinates_type="geographic",
                             weight=False,
                             verbose=False,
                             enable_plotting=False,
                             pseudo_inv=True)

        z_values, sigma = OK.execute('points', test_data[:, 1], test_data[:, 0])
    else:
        raise TypeError("Non-supported Kriging Mode!")

    out_df = pd.DataFrame()
    out_df["timestamp"] = np.array([timestamp]).repeat(len(idx_test))
    out_df["station"] = df["station"].values[idx_test]
    out_df[col_name] = labels[idx_test]
    out_df["pred"] = z_values

    return out_df


def main_kriging(info_path, data_dir, col_name, out_dir, suffix,
                 avail_time_path, data_year, kriging_mode="OK", variogram="spherical"):
    out_dir = "{}/{}".format(out_dir, kriging_mode)
    os.makedirs(out_dir, exist_ok=True)
    ret_path = "{}/{}-{}.csv".format(out_dir, variogram, suffix)
    if os.path.exists(ret_path):
        return None

    avail_timestamps = np.load(avail_time_path, allow_pickle=True)

    graph_path_list = []
    for y in data_year:
        data_path = osp.join(data_dir, str(y), "*.csv")
        graph_path_list += list(glob(data_path))

    train_mask, test_mask, valid_mask = read_train_test_split(info_path)
    coord_df = pd.read_csv(info_path, usecols=["station", "lat", "lon"])

    file_iter = tqdm.tqdm(enumerate(graph_path_list),
                          desc="Training data: ",
                          total=len(graph_path_list),
                          bar_format="{l_bar}{r_bar}")

    out_df_list = []
    for i, _ in file_iter:
        graph_path = graph_path_list[i]
        timestamp = graph_path.split("/")[-1][:-4]

        if timestamp not in avail_timestamps:
            continue

        out_df = kriging_algo(timestamp, graph_path, col_name, coord_df,
                              train_mask, test_mask, mode=kriging_mode, variogram=variogram)
        out_df_list.append(out_df)

    tot_out_df = pd.concat(out_df_list, axis=0, ignore_index=True)
    print("Timestamp Num: ", len(set(tot_out_df["timestamp"].values)))

    tot_out_df.to_csv(ret_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="hk")
    parser.add_argument('--start_year', type=int, default=2008)
    parser.add_argument('--end_year', type=int, default=2012)
    parser.add_argument('--out_dir', type=str, default="./output")

    parser.add_argument('--method', type=str, default="kriging")  # kriging, idw
    parser.add_argument('--variogram', type=str, default="spherical")
    parser.add_argument('--power_p', type=int, default=2)

    args = parser.parse_args()
    start_year = args.start_year
    end_year = args.end_year

    base_dir = "../data"
    if args.dataset.lower() == "bw":
        dataset_dir = f"{base_dir}/BW_132_data"
        info_path = f"{dataset_dir}/BW_stations_info.csv"
        in_dist_path = f"{dataset_dir}/inverse_dist_mat.npy"
        data_dir = f"{dataset_dir}/rain_csv"
        out_dir = f"{args.out_dir}/BW_result"
        col_name = "rainfall"  # col name for used data
    elif args.dataset.lower() == "hk":
        dataset_dir = f"{base_dir}/HK_123_data"
        info_path = f"{dataset_dir}/hk_stations_info.csv"
        in_dist_path = f"{dataset_dir}/inverse_dist_mat.npy"
        data_dir = f"{dataset_dir}/rain_csv"
        out_dir = f"{args.out_dir}/HK_result"
        col_name = "rainfall"  # col name for used data
    else:
        raise NotImplementedError

    assert isinstance(start_year, int)
    assert isinstance(end_year, int)
    assert start_year <= end_year
    avail_time_path = f"{dataset_dir}/{start_year}-{end_year}_avail_used_timestamps.npy"
    data_year = range(start_year, end_year + 1)
    out_name = f"{start_year}-{end_year}_data"

    if args.method == "kriging":
        main_kriging(info_path, data_dir, col_name, out_dir, out_name, avail_time_path,
                     data_year=data_year, kriging_mode="OK", variogram=args.variogram)
    elif args.method == "idw":
        out_suffix = "{}_power{}".format(out_name, args.power_p)
        main_idw(info_path, data_dir, col_name, out_dir, out_suffix, in_dist_path,
                 avail_time_path, data_year=data_year, power_p=args.power_p)

