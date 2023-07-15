import numpy as np
import pandas as pd
from geographiclib.geodesic import Geodesic
import os
from math import radians, cos, sin, asin, sqrt


def calc_dist_angle_mat(info_path, out_path):
    df = pd.read_csv(info_path)
    lons, lats = df["lon"].values, df["lat"].values
    dist_angle_mat = np.zeros((len(lons), len(lons), 2))

    for i in range(len(lons)):
        for j in range(len(lons)):
            dist = Geodesic.WGS84.Inverse(lats[i], lons[i], lats[j], lons[j])
            dist_angle_mat[i, j, 0] = dist["s12"] / 1000.0  # distance, km
            dist_angle_mat[i, j, 1] = dist["azi1"]  # azimuth at the first point in degrees

    print(dist_angle_mat.shape)
    # print(dist_angle_mat)
    np.save(out_path, dist_angle_mat)


def calc_inverse_dist_mat(info_path, out_path):
    """ Baseline Inverse Distance Weighting (IDW) needs this inverse_dist_mat """
    df = pd.read_csv(info_path)
    lons, lats = df["lon"].values, df["lat"].values
    dist_mat = np.zeros((len(lons), len(lons)))

    for i in range(len(lons)):
        for j in range(len(lons)):
            dist = Geodesic.WGS84.Inverse(lats[i], lons[i], lats[j], lons[j])
            dist_mat[i, j] = dist["s12"] / 1000.0  # distance, km

    in_dist_mat = np.power(dist_mat, -1)
    print(in_dist_mat.shape)
    # print(dist_angle_mat)
    np.save(out_path, in_dist_mat)


if __name__ == "__main__":
    base_dir = "../data"

    # HK dataset
    info_path = f"{base_dir}/HK_123_data/hko_stations_info.csv"
    out_dir = f"{base_dir}/HK_123_data"

    # BW dataset
    # info_path = f"{base_dir}/BW_132_data/BW_stations_info.csv"
    # out_dir = f"{base_dir}/BW_132_data"

    os.makedirs(out_dir, exist_ok=True)

    out_name = "dist_angle_mat.npy"
    out_path = f"{out_dir}/{out_name}"
    calc_dist_angle_mat(info_path, out_path)
