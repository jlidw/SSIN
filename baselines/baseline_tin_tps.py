import os
import os.path as osp
import numpy as np
import pandas as pd
from glob import glob
from matplotlib.tri import Triangulation, LinearTriInterpolator, CubicTriInterpolator
import scipy
import scipy.interpolate
import argparse
import tqdm
import sys
sys.path.append('../..')
from SSIN.baselines.utils import read_train_test_split, gen_avail_mask


# Interpolation with Triangular Irregular Network (TIN) or Thin Plate Spline (TPS)
def interp_algo(timestamp, df, col_name, coord_df, train_mask, test_mask, method, kind="linear"):
    data_arr = pd.merge(coord_df, df, how='inner', on=['station']).iloc[:, 1:].values  # easting, northing, rainfall

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

    if method.lower() == "tin":
        # lat, lon
        triFn = Triangulation(train_data[:, 0], train_data[:, 1])  # triangulation function
        if kind == "linear":
            interpolator = LinearTriInterpolator(triFn, train_data[:, 2])  # rainfall values
        elif kind == "cubic":
            interpolator = CubicTriInterpolator(triFn, train_data[:, 2])
        else:
            raise NotImplementedError
        pred_z_values = interpolator(test_data[:, 0], test_data[:, 1])  # interpolated rainfall value
    elif method.lower() == "spline":
        interpolator = scipy.interpolate.Rbf(train_data[:, 0], train_data[:, 1], train_data[:, 2], function="thin_plate")
        pred_z_values = interpolator(test_data[:, 0], test_data[:, 1])  # interpolated rainfall value
    else:
        raise NotImplementedError

    out_df = pd.DataFrame()
    out_df["timestamp"] = np.array([timestamp]).repeat(len(idx_test))
    out_df["station"] = df["station"].values[idx_test]
    out_df[col_name] = labels[idx_test]
    out_df["pred"] = pred_z_values

    return out_df


def main_interp(info_path, data_dir, col_name, out_dir, out_suffix, avail_time_path, data_year, method, kind):
    if method.lower() == "tin":
        out_dir = "{}/{}_{}".format(out_dir, method, kind)
    else:
        out_dir = "{}/{}".format(out_dir, method)

    os.makedirs(out_dir, exist_ok=True)
    ret_path = "{}/{}.csv".format(out_dir, out_suffix)
    if os.path.exists(ret_path):
        return None

    graph_path_list = []
    for y in data_year:
        data_path = osp.join(data_dir, str(y), "*.csv")
        graph_path_list += list(glob(data_path))

    avail_timestamps = np.load(avail_time_path, allow_pickle=True)

    train_mask, test_mask, valid_mask = read_train_test_split(info_path)
    # coord_df = pd.read_csv(info_path, usecols=["station", "easting", "northing"])
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

        df = pd.read_csv(graph_path)
        station_col = df.columns.values[0]
        df = df[[station_col, col_name]]
        df.rename(columns={station_col: "station"}, inplace=True)

        out_df = interp_algo(timestamp, df, col_name, coord_df, train_mask, test_mask, method, kind)
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

    parser.add_argument('--method', type=str, default="spline")  # tin or spline
    parser.add_argument('--kind', type=str, default="linear")  # For TIN: linear, cubic

    args = parser.parse_args()
    start_year = args.start_year
    end_year = args.end_year

    base_dir = "../data"
    if args.dataset.lower() == "bw":
        dataset_dir = f"{base_dir}/BW_132_data"
        info_path = f"{dataset_dir}/BW_stations_info.csv"
        data_dir = f"{dataset_dir}/rain_csv"
        out_dir = f"{args.out_dir}/BW_result"
        col_name = "rainfall"  # col name for used data
    elif args.dataset.lower() == "hk":
        dataset_dir = f"{base_dir}/HK_123_data"
        info_path = f"{dataset_dir}/hk_stations_info.csv"
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

    method = args.method
    kind = args.kind

    main_interp(info_path, data_dir, col_name, out_dir, out_name, avail_time_path, data_year, method=method, kind=kind)

