import torch
import random
import numpy as np
import os.path as osp
import pandas as pd
import os


class Paths:
    def __init__(self, output_path):
        self.output_path = output_path

        self.checkpoints_path = f'{output_path}/train/checkpoints_path'
        self.runs_path = f'{output_path}/train/runs'
        self.test_ret_path = f'{output_path}/test'
        self.create_paths()

    def create_paths(self):
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.checkpoints_path, exist_ok=True)
        os.makedirs(self.runs_path, exist_ok=True)
        os.makedirs(self.test_ret_path, exist_ok=True)


class SelfStandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def save_args(args, cfg, out_dir):
    out_path = osp.join(out_dir, 'args_settings.txt')
    with open(out_path, 'w') as f:
        f.writelines("ArgumentParser" + '\n')
        for key, value in vars(args).items():
            f.writelines(key + ': ' + str(value) + '\n')

        f.writelines('\n' + "Configs" + '\n')
        for key, value in vars(cfg).items():
            if key.startswith("__") or key.endswith("__"):
                pass
            else:
                f.writelines(key + ': ' + str(value) + '\n')


def save_running_time(out_dir, run_time, affix=None):
    out_path = osp.join(out_dir, 'args_settings.txt')
    with open(out_path, 'a') as f:
        if affix is None:
            f.writelines('\n' + f"Total running time: {run_time} hours" + '\n')
        else:
            f.writelines('\n' + f"{affix}: {run_time} " + '\n')


def init_seeds(args, deterministic=True):
    seed = args.seed
    args.cuda = torch.cuda.is_available()

    random.seed(seed)  # new
    np.random.seed(seed)
    torch.manual_seed(seed)  # for cpu seed
    if args.cuda:  # for gpu seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # new: all gpu

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_gauge_timestamp_from_data(data_df, idx_test, timestamp):
    gauges = data_df["gauge"].values[idx_test]
    timestamps = np.array([timestamp]).repeat(len(idx_test))

    return gauges, timestamps


def save_csv_results(out_path, timestamp_list, gauge_list, labels_list, preds_list):
    timestamp_arr = np.concatenate(timestamp_list)
    # gauge_arr = np.concatenate(gauge_list)
    real_rain = np.concatenate(labels_list)
    pred_rain = np.concatenate(preds_list)

    out_df = pd.DataFrame()
    out_df['timestamp'] = timestamp_arr
    # out_df['gauge'] = gauge_arr
    out_df['label'] = real_rain
    out_df['pred'] = pred_rain

    out_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    pass
