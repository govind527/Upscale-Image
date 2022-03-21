# Upscale-Image
This is the project I have done to upscale the blurry Image.

## ESRGAN (Enhanced SRGAN) [:rocket:]

### Enhanced Super-Resolution Generative Adversarial Networks
By Xintao Wang, [Ke Yu](https://yuke93.github.io/), Shixiang Wu, [Jinjin Gu](http://www.jasongt.com/), Yihao Liu, [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=en), [Yu Qiao](http://mmlab.siat.ac.cn/yuqiao/), [Chen Change Loy](http://personal.ie.cuhk.edu.hk/~ccloy/)



#### BibTeX

    @InProceedings{wang2018esrgan,
        author = {Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Loy, Chen Change},
        title = {ESRGAN: Enhanced super-resolution generative adversarial networks},
        booktitle = {The European Conference on Computer Vision Workshops (ECCVW)},
        month = {September},
        year = {2018}
    }


#### Dependencies
- Python 3
- [PyTorch >= 1.0](https://pytorch.org/) (CUDA version >= 7.5 if installing with CUDA. [More details](https://pytorch.org/get-started/previous-versions/))
- Python packages:  `pip install numpy opencv-python`
- OpenCV glob2: `pip install opencv-python glob2` 

### Test models
1. Clone this github repo.
```
git clone https://github.com/xinntao/ESRGAN
cd ESRGAN
```
2. Place your own **low-resolution images** in `./LR` folder. (There are two sample images - baboon and comic).
3. Download pretrained models from [Google Drive](https://drive.google.com/drive/u/0/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY) . Place the models in `./models`. We provide two models with high perceptual quality and high PSNR performance (see [model list](https://github.com/xinntao/ESRGAN/tree/master/models)).
4. Run test. We provide ESRGAN model and RRDB_PSNR model and you can config in the `test.py`.
```
python test.py
```
5. The results are in `./results` folder.



## ESRGAN
We improve the [SRGAN](https://arxiv.org/abs/1609.04802) from three aspects:
1. adopt a deeper model using Residual-in-Residual Dense Block (RRDB) without batch normalization layers.
2. employ [Relativistic average GAN](https://ajolicoeur.wordpress.com/relativisticgan/) instead of the vanilla GAN.
3. improve the perceptual loss by using the features before activation.

In contrast to SRGAN, which claimed that **deeper models are increasingly difficult to train**, our deeper ESRGAN model shows its superior performance with easy training.

<p align="center">
  <img height="120" src="figures/architecture.jpg">
</p>
<p align="center">
  <img height="180" src="figures/RRDB.png">
</p>

## Network Interpolation
We propose the **network interpolation strategy** to balance the visual quality and PSNR.

<p align="center">
  <img height="500" src="figures/net_interp.jpg">
</p>

We show the smooth animation with the interpolation parameters changing from 0 to 1.
Interestingly, it is observed that the network interpolation strategy provides a smooth control of the RRDB_PSNR model and the fine-tuned ESRGAN model.

<p align="center">
  <img height="480" src="figures/81.gif">
  &nbsp &nbsp
  <img height="480" src="figures/102061.gif">
</p>

## Qualitative Results
PSNR (evaluated on the Y channel) and the perceptual index used in the PIRM-SR challenge are also provided for reference.

<p align="center">
  <img src="figures/qualitative_cmp_01.jpg">
</p>
<p align="center">
  <img src="figures/qualitative_cmp_02.jpg">
</p>
<p align="center">
  <img src="figures/qualitative_cmp_03.jpg">
</p>
<p align="center">
  <img src="figures/qualitative_cmp_04.jpg">
</p>

## Ablation Study
Overall visual comparisons for showing the effects of each component in
ESRGAN. Each column represents a model with its configurations in the top.
The red sign indicates the main improvement compared with the previous model.
<p align="center">
  <img src="figures/abalation_study.png">
</p>

## BN artifacts
We empirically observe that BN layers tend to bring artifacts. These artifacts,
namely BN artifacts, occasionally appear among iterations and different settings,
violating the needs for a stable performance over training. We find that
the network depth, BN position, training dataset and training loss
have impact on the occurrence of BN artifacts.
<p align="center">
  <img src="figures/BN_artifacts.jpg">
</p>

## Useful techniques to train a very deep network
We find that residual scaling and smaller initialization can help to train a very deep network. More details are in the Supplementary File attached in our [paper](https://arxiv.org/abs/1809.00219).

<p align="center">
  <img height="250" src="figures/train_deeper_neta.png">
  <img height="250" src="figures/train_deeper_netb.png">
</p>

## The influence of training patch size
We observe that training a deeper network benefits from a larger patch size. Moreover, the deeper model achieves more improvement (∼0.12dB) than the shallower one (∼0.04dB) since larger model capacity is capable of taking full advantage of
larger training patch size. (Evaluated on Set5 dataset with RGB channels.)
<p align="center">
  <img height="250" src="figures/patch_a.png">
  <img height="250" src="figures/patch_b.png">
</p>
