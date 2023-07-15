import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calc_RMSE(predict, label):
    rmse = np.sqrt(mean_squared_error(predict, label))
    return rmse


def calc_MAE(predict, label):
    mae = mean_absolute_error(predict, label)
    return mae


def calc_NSE(predict, label):  # The Nash–Sutcliffe’s efficiency
    nse = 1 - np.sum(np.square(label - predict)) / np.sum(np.square(label - np.mean(label)))
    return nse


def calc_scores(predict, label):
    if torch.is_tensor(predict):
        predict = predict.type_as(label).cpu().detach().numpy()
    if torch.is_tensor(label):
        label = label.cpu().detach().numpy()

    rmse = calc_RMSE(predict, label)
    mae = calc_MAE(predict, label)
    nse = calc_NSE(predict, label)
    return rmse, mae, nse


if __name__ == "__main__":
    pass

