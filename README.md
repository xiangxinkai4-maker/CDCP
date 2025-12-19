# CDCP# 

## Overview
>In this work, we propose CDCP, a unified framework integrating a camera-aware Jaccard distance metric with a dynamic clustering adjustment strategy. Our approach reduces label noise through part-based pseudo-label refinement and improves cross-camera similarity estimation. Extensive experiments on Market-1501, DukeMTMC-reID, MSMT17, and VeRi-776 demonstrate that CDCP consistently outperforms baseline methods, achieving significant gains in mAP and Rank-1 accuracy.
Our CDCP learns discriminative representations with rich local contexts. Also, it operates in a self-ensemble manner without auxiliary teacher networks, which is computationally efficient.
>**This is the official implementation of the manuscript:**  
**"Enhancing Unsupervised Person Re-Identification via Camera-Aware Jaccard Distance and Adaptive Dynamic Clustering"**  
*Submitted to The Visual Computer.*  
**If you find this code useful in your research, please consider citing our paper.**


## Getting Started
### Installation
```shell
git clone https://github.com/yoonkicho/PPLR
cd PPLR
python setup.py develop
```
### Preparing Datasets
```shell
cd examples && mkdir data
```
Download the object re-ID datasets [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](https://arxiv.org/abs/1711.08565), and [VeRi-776](https://github.com/JDAI-CV/VeRidataset) to `PPLR/examples/data`.
The directory should look like:
```
PPLR/examples/data
├── Market-1501-v15.09.15
├── MSMT17_V1
└── VeRi
```
## Training
We utilize 3 TITAN XP GPUs for training.
We use 384x128 sized images for Market-1501 and MSMT17 and 256x256 sized images for VeRi-776.

### Training without camera labels
For Market-1501:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/train_pplr.py \
-d market1501 --logs-dir $PATH_FOR_LOGS
```
For MSMT17:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/train_pplr.py \
-d msmt17 --logs-dir $PATH_FOR_LOGS
```
For VeRi-776:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/train_pplr.py \
-d veri -n 8 --height 256 --width 256 --eps 0.7 --logs-dir $PATH_FOR_LOGS
```

### Training with camera labels
For Market-1501:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/train_pplr_cam.py \
-d market1501 --eps 0.4 --logs-dir $PATH_FOR_LOGS
```
For MSMT17:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/train_pplr_cam.py \
-d msmt17 --eps 0.6 --lam-cam 1.0 --logs-dir $PATH_FOR_LOGS
```
For VeRi-776:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/train_pplr_cam.py \
-d veri -n 8 --height 256 --width 256 --eps 0.7 --logs-dir $PATH_FOR_LOGS
```

## Testing 
We use a single TITAN RTX GPU for testing.

You can download pre-trained weights from this [link](https://drive.google.com/drive/folders/1m5wDOJG7qk62PjkoOpTspNmk0nhLc4Vi?usp=sharing).

For Market-1501:
```
CUDA_VISIBLE_DEVICES=0\
python examples/test.py \
-d market1501 --resume $PATH_FOR_MODEL
```
For MSMT17:
```
CUDA_VISIBLE_DEVICES=0\
python examples/test.py \
-d msmt17 --resume $PATH_FOR_MODEL
```
For VeRi-776:
```
CUDA_VISIBLE_DEVICES=0\
python examples/test.py \
-d veri --height 256 --width 256 --resume $PATH_FOR_MODEL
```


