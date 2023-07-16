import os
import numpy as np
import csv
import pickle
import argparse
import pandas as pd


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)
        distanceA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)
        # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
        if id_filename:
            if id_filename.endswith("txt"):
                with open(id_filename, 'r') as f:
                    sensor_ids = f.read().strip().split('\n')
                    id_dict = {int(i): idx for idx, i in enumerate(sensor_ids)}  # 把节点id（idx）映射成从0开始的索引
            elif id_filename.endswith("csv"):
                sensor_ids = pd.read_csv(id_filename)["sensor"].values
                id_dict = {int(i): idx for idx, i in enumerate(sensor_ids)}
            else:
                raise NotImplementedError

            with open(distance_df_filename, 'r') as f:
                f.readline()  # 略过表头那一行
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])

                    if i in sensor_ids and j in sensor_ids:
                        if distance != 0:
                            A[id_dict[i], id_dict[j]] = 1
                        distanceA[id_dict[i], id_dict[j]] = distance
            return A, distanceA
        else:  # distance file中的id直接从0开始
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distanceA[i, j] = distance
            return A, distanceA


def get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)
        distanceA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)
        # distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
        if id_filename:
            if id_filename.endswith("txt"):
                with open(id_filename, 'r') as f:
                    sensor_ids = f.read().strip().split('\n')
                    id_dict = {int(i): idx for idx, i in enumerate(sensor_ids)}  # 把节点id（idx）映射成从0开始的索引
            elif id_filename.endswith("csv"):
                sensor_ids = pd.read_csv(id_filename)["sensor"].values
                id_dict = {int(i): idx for idx, i in enumerate(sensor_ids)}
            else:
                raise NotImplementedError

            with open(distance_df_filename, 'r') as f:
                f.readline()  # 略过表头那一行
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])

                    if i in sensor_ids and j in sensor_ids:
                        if distance != 0:
                            A[id_dict[i], id_dict[j]] = 1
                            A[id_dict[j], id_dict[i]] = 1
                        distanceA[id_dict[i], id_dict[j]] = distance
                        distanceA[id_dict[j], id_dict[i]] = distance
            return A, distanceA
        else:  # distance file中的id直接从0开始
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    A[j, i] = 1
                    distanceA[i, j] = distance
                    distanceA[j, i] = distance
            return A, distanceA


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="bay")
    parser.add_argument('--direction', type=str, default="one")  # one, two

    args = parser.parse_args()

    dataset = args.dataset
    direction = args.direction

    if dataset == "bay":
        num_of_vertices = 325
        distance_df_filename = "../data/PEMS-BAY/raw_data/distances_bay_2017.csv"
        id_filename = "../data/PEMS-BAY/raw_data/graph_sensor_locations_bay.csv"
        out_dir = "../data/PEMS-BAY/"
        os.makedirs(out_dir, exist_ok=True)
    else:
        raise NotImplementedError

    if direction == "one":  # `one` means undirected; this is used for SSIN experiments
        adj_mx, distance_mx = get_adjacency_matrix_2direction(distance_df_filename, num_of_vertices, id_filename)
        suffix = "one"
    elif direction == "two":  # `two` mean bidirected
        adj_mx, distance_mx = get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename)
        suffix = "two"
    else:
        raise NotImplementedError

    add_self_loop = True
    if add_self_loop:
        adj_mx = adj_mx + np.identity(adj_mx.shape[0])
        # distance_mx = distance_mx + np.identity(distance_mx.shape[0])  # distance_mx does not add self-loop

    if distance_mx.ndim == 2:
        distance_mx = np.expand_dims(distance_mx, axis=2)
        print("distance_mx shape: ", distance_mx.shape)

    # Traffic data is not fully-connected; have additional attn_mask for attention operation.
    np.save(f"{out_dir}/adj_attn_mask_{suffix}.npy", adj_mx)

    # Traffic data use travel distance instead of geographic distance;
    # hence we just use the travel distance for SSIN and SpaFormer model,
    # ignoring the azimuth angle between two locations.
    np.save(f"{out_dir}/dist_mat_{suffix}.npy", distance_mx)

