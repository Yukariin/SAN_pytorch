# SAN_pytorch
Second-order Attention Network for Single Image Super-resolution

## Pre-trained Models

### Data
Final dataset consists of 1000/100 (train/val) images.
Original images are all PNG images at least 2K x 2K. Images downsampled to 2K by LANCZOS, that is they have 2K pixels on at least one of the axes (vertical or horizontal), and then cropped to multiple of 12 pixels on both axes.
All images splitted into 96x96/192x192 (x2/x4) HR and 48x48 LR (with jpeg noise) overlapping patches. All HR patches filtered by it's gradient and variance, and stored in SQLite database.

Image noise are from JPEG format only. Same as for [waifu2x](https://github.com/yu45020/Waifu2x).
13
Noise level 1 means quality ranges uniformly from [75, 95]; level 2 means quality ranges uniformly from [50, 75].

### Scores
Scores calculated on validation dataset which consists of ~14K HR/LR patches for scale factor of 2.
​
| Model | Noise level | PSNR(+) | SSIM(+) |
| ----- | ----------- | ------- | ------- |
| SAN   | 0           | 43.6532 | 0.9965  |
| SAN   | 1           | 37.1034 | 0.9866  |

### Models
[SAN](https://drive.google.com/file/d/1bzRmFD7Xi8f38poKpzM8k_OSlKkBTaxi/view?usp=sharing) - Scale factor x2 - Noise level 0

[SAN](https://drive.google.com/file/d/10d_bnCVuxMrfkrk-djaXw3uPcWajksEo/view?usp=sharing) - Scale factor x2 - Noise level 1

## Citation
    @InProceedings{Dai_2019_CVPR,
        author = {Dai, Tao and Cai, Jianrui and Zhang, Yongbing and Xia, Shu-Tao and Zhang, Lei},
        title = {Second-Order Attention Network for Single Image Super-Resolution},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2019}
    }

## Acknowledgments
Model [codes](https://github.com/Yukariin/SAN_pytorch/blob/master/model.py) mainly based on reference [implementation](https://github.com/daitao/SAN).

[ImageSplitter](https://github.com/Yukariin/SAN_pytorch/blob/master/utils.py) and [SQLite-based](https://github.com/Yukariin/SAN_pytorch/blob/master/data.py) dataset are based on [yu45020](https://github.com/yu45020)'s waifu2x [re-implementation](https://github.com/yu45020/Waifu2x).

[SQLite-based](https://github.com/Yukariin/SAN_pytorch/blob/master/gen_data.py) data generator based on NatSR [implementation](https://github.com/JWSoh/NatSR).
