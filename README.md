# Semi-MoreGAN

This is the PyTorch implementation for our paper: **Semi-MoreGAN: A Semi-supervised Image Mixture of Rain Removal Network**.

## 1. Requirements

- Python 3.6.0
- PyTorch (I use version 1.8.0. Suggest ≥ 1.2.0.)
- opencv
- numpy
- easydict
- skimage

## 2. Data preparation
Download the RainCityscapes training and validation images from [Cityscapes website](https://www.cityscapes-dataset.com/downloads/).

Organize the downloaded files as follows:
```
Semi-MoreGAN
├── datasets
│   ├── rain
│   ├── gt
│   ├── depth
│   ├── real
```
More details please see train_input.txt, train_gt.txt, train_depth.txt and train_real.txt.
### 3. Main Training
Clone this repository:          
   ```shell
   git clone https://github.com/syy-whu/Semi-MoreGAN.git
   ```
run the supervised train code
```
python supervised_train.py
```
run the semi-supervised train code
```
python semi-supervised_train.py
```
### 4. Evaluation
Test the Semi-MoreGAN:
   ```shell
   python3 infer.py    
   ```
The PSNR and SSIM evaluation codes are from the skimage.
## 5. Acknowledgement
The code is based on [DGNL-Net](https://github.com/xw-hu/DGNL-Net). 
## 6. Citation
If you find this work useful in your research, please consider cite:

```
@article {10.1111:cgf.14690,
journal = {Computer Graphics Forum},
title = {{Semi-MoreGAN: Semi-supervised Generative Adversarial Network for Mixture of Rain Removal}},
author = {Shen, Yiyang and Wang, Yongzhen and Wei, Mingqiang and Chen, Honghua and Xie, Haoran and Cheng, Gary and Wang, Fu Lee},
year = {2022},
publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
ISSN = {1467-8659},
DOI = {10.1111/cgf.14690}
}

@article{hu2021single,
     title={Single-Image Real-Time Rain Removal Based on Depth-Guided Non-Local Features},
     author={Hu, Xiaowei and Zhu, Lei and Wang, Tianyu and Fu, Chi-Wing and Heng, Pheng-Ann},
     journal={IEEE Transactions on Image Processing},
     volume={30},
     pages={1759--1770},
     year={2021},
     publisher={IEEE}
}
```
