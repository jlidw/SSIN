# SSIN
The code is for our paper ["SSIN: Self-Supervised Learning for Rainfall Spatial Interpolation"](https://dl.acm.org/doi/10.1145/3589321) 
and this paper has been accepted by SIGMOD 2023.

## About Rainfall Spatial Interpolation
### Why rainfall spatial interpolation is challenging？

### Spatial Interpolation vs. Time Series Imputation

##  Datasets
Two real-world hourly raingauge datasets, HK and BW, are collected and used in this paper. 
The PEMS-BAY dataset was first released by [DCRNN](https://github.com/liyaguang/DCRNN/tree/master).
You can download these [datasets](https://drive.google.com/drive/folders/1tiS5UjcspNKcWL8RA7J3PxqhwciR5Lg3) and place them in the `data` folder.

## Instructions
`attn_tvm`:
* Include the files about the TVM kernel implementation.
* `lib`: includes generated TVM kernels (\".so\" file).

`baselines`:
* Include the implementation of IDW, OK, TIN, and TPS.
* For GNN-based solutions, you can refer to their original code: [KCN](https://github.com/tufts-ml/KCN) and [IGNNK](https://github.com/Kaimaoge/IGNNK).

`dataset_collator`:
* `create_data.py`: generate the masked sequences which will be provided to Trainer.py for training and testing.

`networks`:
* Include files about the network layers and the model architecture.

`postprocess`:
* Calculate the RMSE, MAE, and NSE for predicted results.

`preprocess`:
* `preprocessing.py`: preprocess data and general the `pkl` data for training/testing. 
* `dist_angle.py`: generate one matrix that stores the distance and azimuth between all location pairs.

`utils`:
* Some configs and useful functions.

## Run
```
python main_train.py --dataset=hk
```

```
python main_train.py --dataset=bw
```

```
python main_train.py --dataset=bay
```

## Citation
```
@article{li2023ssin,
  title={SSIN: Self-Supervised Learning for Rainfall Spatial Interpolation},
  author={Li, Jia and Shen, Yanyan and Chen, Lei and Ng, Charles Wang Wai},
  journal={Proceedings of the ACM on Management of Data},
  volume={1},
  number={2},
  pages={1--21},
  year={2023},
  publisher = {Association for Computing Machinery},
  url = {https://doi.org/10.1145/3589321}
}
```
