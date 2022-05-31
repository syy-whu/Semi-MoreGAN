# Semi-MoreGAN
Semi-MoreGAN: A Semi-supervised Image Mixture of Rain Removal Network

This is the PyTorch implementation for our paper:

Here we provide a supervised version, and more codes, real-world datasets and pre-models will be released after published!

## 1. Requirements

- Python 3.6.0
- PyTorch (I use version 1.8.0. Suggest ≥ 1.2.0.)
- opencv
- numpy
- easydict

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
   git clone 
   ```

```
python train.py
```
