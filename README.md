# SSIN
The code is for our paper ["SSIN: Self-Supervised Learning for Rainfall Spatial Interpolation"](https://dl.acm.org/doi/10.1145/3589321) 
and this paper has been accepted by SIGMOD 2023.

## About Rainfall Spatial Interpolation
### Spatial Interpolation vs. Time Series Imputation:
Spatial interpolation is to “predict” data for any locations with **no historical observations** according to sparse station observations. This problem is fundamentally different and more challenging than multivariate time-series imputation, which assumes data at certain locations is partially missing across time.

### Rainfall vs. Other Meteorological Variables:
The discontinuity of rainfall (usually zero accumulations) means more complex spatial distribution, while other meteorological variables (e.g., temperature and humidity) usually show smoother distribution.

##  Datasets
Two real-world hourly raingauge datasets, HK and BW, are collected and used in this paper. Besides, we take traffic spatial interpolation as another use case and employ one commonly used real-world dataset, PEMS-BAY, to conduct additional experiments.

You can download the processed datasets from [Google Drive](https://drive.google.com/drive/folders/1tiS5UjcspNKcWL8RA7J3PxqhwciR5Lg3) and place them in the `data` folder.

### Raw Data
* **HK**: This dataset is provided by Hong Kong Observatory (HKO) and the Geotechnical Engineering Office (GEO). Now we only have permission for using, we are communicating with them on how to release data.
* **BW**: [Climate Data Center (CDC) of the German Weather Service (DWD)](https://www.dwd.de/EN/climate_environment/cdc/cdc_node_en.html).
* **PEMS-BAY**: It is first released by [DCRNN](https://github.com/liyaguang/DCRNN/tree/master).

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
