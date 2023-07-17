# SSIN
The code is for our paper ["SSIN: Self-Supervised Learning for Rainfall Spatial Interpolation"](https://dl.acm.org/doi/10.1145/3589321) 
and this paper has been accepted by SIGMOD 2023.

## About Rainfall Spatial Interpolation
### Spatial Interpolation vs. Time Series Imputation:
Spatial interpolation is to “predict” data for **any locations with no historical observations** according to sparse station observations. This problem is fundamentally different and more challenging than multivariate time-series imputation, which assumes data at **certain locations** is **partially missing across time**.

### Rainfall vs. Other Meteorological Variables:
The **intermittency** of rainfall (usually zero accumulations) means more complex spatial distribution, while other meteorological variables (e.g., temperature and humidity) usually show smoother distribution.

##  Datasets
Two real-world hourly raingauge datasets, **HK** and **BW**, are collected and used in this paper. Besides, we take traffic spatial interpolation as another use case and employ one commonly used real-world dataset, **PEMS-BAY**, to conduct additional experiments.

### Processed Data
Download the processed datasets from [Google Drive](https://drive.google.com/drive/folders/1tiS5UjcspNKcWL8RA7J3PxqhwciR5Lg3) and place them in the `data` folder.

#### How to select rainy timestamps?
Since rainfall is intermittent, performing spatial interpolating for all zeros is meaningless, and too many all-zero data may negatively affect model training. We perform data selection to filter out timestamps with zero/tiny rain to form the final dataset used (HK: 3855 valid timestamps; BW: 3640 valid timestamps). We follow the data selection process below:
* **HK** dataset: Geotechnical Engineering Office (GEO) published annual reports about rainstorm events ([Report Link](https://www.cedd.gov.hk/eng/publications/geo/geo-reports/index.html), the title starts with "Factual Report on Hong Kong Rainfall and Landslides"). We directly select rainy hours from these rainstorm days. For each hour on rainstorm days, if the num of stations owning valid rainfall values >= 5, then it is selected; otherwise, it is discarded.
* **BW** dataset: No records of rainstorms/heavy rain are available. Hence, we first select rainy days and then select rainy hours. For each day, if the accumulated rainfall at any station >= 25mm, then this day is selected as a valid rainy day. For each valid rainy day, we follow the same hour selection as the HK dataset: for each hour on a rainy day, if the num of stations owning valid rainfall values >= 5, then it is selected; otherwise, it is discarded.

### Raw Data
* **HK**: It is provided by Hong Kong Observatory (HKO) and Geotechnical Engineering Office (GEO); we are communicating with them on how to release data.
* **BW**: The raw data can be downloaded directly from [Climate Data Center (CDC)](https://www.dwd.de/EN/climate_environment/cdc/cdc_node_en.html) of the German Weather Service (DWD); the raw data is quite large (over 500MB) and includes many redundant and noisy info.
* **PEMS-BAY**: This dataset is first released by [DCRNN](https://github.com/liyaguang/DCRNN/tree/master).

## Baselines
In the `baselines` folder, you can find the implementation of IDW, OK, TIN, and TPS:
* **IDW**: self-implementation.
* **OK**: by using [pykrige.ok.OrdinaryKriging](https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/generated/pykrige.ok.OrdinaryKriging.html).
* **TIN**: by using [matplotlib.tri](https://matplotlib.org/stable/api/tri_api.html).
* **TPS**: by using [scipy.interpolate.Rbf](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html).

For GNN-based baselines, please refer to their original code: [KCN](https://github.com/tufts-ml/KCN) and [IGNNK](https://github.com/Kaimaoge/IGNNK).


## Instructions
`attn_tvm`:
* Include the files about the TVM kernel implementation.
* `lib`: includes generated TVM kernels (\".so\" file).

`baselines`:
* Include the implementation of IDW, OK, TIN, and TPS.

`dataset_collator`:
* `create_data.py`: generate the masked sequences which will be provided to Trainer.py for training and testing.

`networks`:
* Include files about the network layers and the model architecture.

`postprocess`:
* Calculate the RMSE, MAE, and NSE for predicted results.

`preprocess`:
* `dist_angle.py`: for HK/BW dataset, generate one matrix that stores the distance and azimuth between all location pairs.
* `generate_traffic_adj_mx.py`: for PEMS-BAY dataset, generate the distance matrix and additional adj_attn_mask (since traffic data is not fully connected, it needs an additional adj_attn_mask for attention operation.). 
* `preprocessing.py`: preprocess HK/BW dataset and general the `pkl` data for training/testing.
* `preprocess_pems_bay.py`:  preprocess PEMS-BAY dataset and general the `pkl` data for training/testing.

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
  publisher = {Association for Computing Machinery}
}
```
