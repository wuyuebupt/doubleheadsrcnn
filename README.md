# Double Heads RCNN

This is the implementation of CVPR 2020 paper "Rethinking Classification and Localization for Object Detection". The code is based on the [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).


If the paper and code helps you, we would appreciate your kindly citations of our paper.
```
@inproceedings{wu2020rethinking,
  title={Rethinking Classification and Localization for Object Detection},
  author={Wu, Yue and Chen, Yinpeng and Yuan, Lu and Liu, Zicheng and Wang, Lijuan and Li, Hongzhi and Fu, Yun},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

### Contents
1. [Installation](#installation)
2. [Models](#models)
2. [Running](#running)

### Installation 
Follow the [maskrcnn-benchmark](./OLD_README.md) to install code and set up the dataset.

A docker image is also provided 
```
docker pull yuewudocker/pytorchdoubleheads 
```
If you use this docker, you can run the ./cmd_install.sh script for the installation. 

Most experiments are done under the following environments:
```
PyTorch version: 1.0.0
OS: Ubuntu 16.04.3 LTS
Python version: 3.6
CUDA runtime version: 9.0.176
Nvidia driver version: 410.78
GPU: 4x Tesla P100-PCIE-16GB 
```


### Models
Results on the COCO 2017 validation set:

| Models         | AP |  AP_0.5 | AP_0.7 | AP_s | AP_m | AP_l | Link |
| -------------- | ------ | ---- |  ---- |  ---- |  ---- |  ---- |  ---- | 
| ResNet-50-FPN  | 40.3 | 60.3 | 44.2 | 22.4 | 43.3 | 54.3 | [model](https://drive.google.com/open?id=1KnRoyJQjS9rQUTCFEm54AIsUy2qbZTK_) |
| ResNet-101-FPN | 41.9 | 62.4 | 45.9 | 23.9 | 45.2 | 55.8 | [model](https://drive.google.com/open?id=18CMdq4U9TZOCz7SSj-3c27xkfqO_gvwP) |

Results on COCO 2017 test-dev:

| Models         | AP |  AP_0.5 | AP_0.7 | AP_s | AP_m | AP_l | Link |
| -------------- | ------ | ---- |  ---- |  ---- |  ---- |  ---- |  ---- | 
| ResNet-101-FPN | 42.3 | 62.8 | 46.3 | 23.9 | 44.9 | 54.3 | [bbox](https://drive.google.com/open?id=1jBQ2S_eDUyEJplZtofAAHOvkqo4UmlpA) |



### Running
Use config files in ./configs/double_heads/ for Training and Testing.
#### Run Inference
Download models to the ./models directory. Then use the following script:
```
sh cmd_test.sh
```

You need modify the data path:
```
export DATA_DIR=/path/to/datafolder/
```

#### Run Training
You can use the ./cmd_train.sh script to train with 4 gpus. 

You have to modify following paths:
```
export OUTPUT_DIR=/path/to/modelfolder/
export PRETRAIN_MODEL=/path/to/pretrained/model
export DATA_DIR=/path/to/datafolder/
```




