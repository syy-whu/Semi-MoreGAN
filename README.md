# Semi-MoreGAN

This is the PyTorch implementation for our paper:Semi-MoreGAN: A Semi-supervised Image Mixture of Rain Removal Network

Here we provide a supervised version, and more codes, real-world datasets and pre-models will be released after published!

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
python train.py
```
### 4. Evaluation
Test the Semi-MoreGAN:
   ```shell
   python3 infer.py    
   ```
The PSNR and SSIM evaluation codes are from the skimage.
## 5. Citation
If you find this work useful in your research, please consider cite:

```
@article{shen2022semi,
  title={Semi-MoreGAN: A New Semi-supervised Generative Adversarial Network for Mixture of Rain Removal},
  author={Shen, Yiyang and Wang, Yongzhen and Wei, Mingqiang and Chen, Honghua and Xie, Haoran and Cheng, Gary and Wang, Fu Lee},
  journal={arXiv preprint arXiv:2204.13420},
  year={2022}
}
```
