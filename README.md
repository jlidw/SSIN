# SSIN
The code and datasets are being released...

The code is for our paper ["SSIN: Self-Supervised Learning for Rainfall Spatial Interpolation"](https://dl.acm.org/doi/10.1145/3589321) 
and this paper has been accepted by SIGMOD 2023.

##  Datasets
Two real-world raingauge datasets, HK and BW, are used in this paper. Download the [datasets](https://drive.google.com/drive/folders/1tiS5UjcspNKcWL8RA7J3PxqhwciR5Lg3) and place them in the `data` folder.


## Instructions
`preprocess`:
* preprocessing.py: preprocess the data and general training pkl data and testing pkl data for convenience.
* dist_angle.py: generate one matrix that stores the distance and azimuth between all location pairs.

`dataset_collator`:
* create_data.py: generate the masked data which will be provided to Trainer.py for training and testing.

`networks`:
* Include files about the network layers and the model architecture.

`attn_tvm`:
* Includes the files about the TVM kernel implementation.
* `lib`: includes generated TVM kernels (\".so\" file).

`postprocess`:
* Calculate the RMSE, MAE and NSE for the predicted results.

`utils`:
* Some configs and useful functions.

## Run
```
python main_train.py --dataset=hk or python main_train.py --dataset=bw
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
