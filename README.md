# SAN_pytorch
Second-order Attention Network for Single Image Super-resolution

## Pre-trained Models

### Data
All models trained on the same anime-themed datased based on [Danbooru2019](https://www.gwern.net/Danbooru2019).
Final dataset consists of 1000/100 (train/val) images.
Original images are all PNG images at least 2K x 2K. Images downsampled to 2K by LANCZOS, that is they have 2K pixels on at least one of the axes (vertical or horizontal), and then cropped to multiple of 12 pixels on both axes.
All images splitted into 96x96/192x192 (x2/x4) HR and 48x48 LR (with jpeg noise) overlapping patches. All HR patches filtered by it's gradient and variance, and stored in SQLite database.

Image noise are from JPEG format only. Same as for [waifu2x](https://github.com/yu45020/Waifu2x).
Noise level 1 means quality ranges uniformly from [75, 95]; level 2 means quality ranges uniformly from [50, 75].

**Note:** Dataset may contain some NSFW data.

Anime - Scale factor x2 - Noise level 0 - [train](https://drive.google.com/file/d/1d-U6O8BGixNd0vESi19ymRt-1-4-0x43/view?usp=sharing) / [val](https://drive.google.com/file/d/1qWbFJSCBGryIFf5n8d8aQMr7xVgJ_14J/view?usp=sharing)

Anime - Scale factor x2 - Noise level 1 - [train](https://drive.google.com/file/d/1PxLqqttxECnh6yj-KtvUBRBDUtyfrm7Q/view?usp=sharing) / [val](https://drive.google.com/file/d/1awEailakz0TJXyPdB5L4TCW7iztftyFK/view?usp=sharing)

Anime - Scale factor x4 - Noise level 0 - [train](https://drive.google.com/file/d/16X3fpqVB6Uusgv9LaO2Alytu-knd90ZM/view?usp=sharing) / [val](https://drive.google.com/file/d/1sD8fHYEbqA0_-kDnSoVhDp9vPsatahMm/view?usp=sharing)

Anime - Scale factor x4 - Noise level 1 - [train](https://drive.google.com/file/d/1w4krazoUOW8Sg06UHS4zHEayCZJsYzNO/view?usp=sharing) / [val](https://drive.google.com/file/d/1gEqyNtE1LZOMsqqGSXPN2yFW5r2mEMtB/view?usp=sharing)

### Scores
Scores calculated on validation dataset which consists of ~14K HR/LR patches for scale factor of 2.
​
| Model | Scale factor |Noise level | PSNR(+) | SSIM(+) |
| ----- | ------------ | ---------- | ------- | ------- |
| SAN   | 2            | 0          | 43.7363 | 0.9965  |
| SAN   | 2            | 1          | 37.1034 | 0.9866  |
| SAN   | 4            | 0          | 34.6585 | 0.9790  |

### Models
[SAN](https://drive.google.com/file/d/1i64McPLgLn2WTOxVgevTahuslme_C_2D/view?usp=sharing) - Scale factor x2 - Noise level 0

[SAN](https://drive.google.com/file/d/10d_bnCVuxMrfkrk-djaXw3uPcWajksEo/view?usp=sharing) - Scale factor x2 - Noise level 1

[SAN](https://drive.google.com/file/d/16St4_8NlcFTfKrLR0hOlHrMfGVHn9eZS/view?usp=sharing) - Scale factor x4 - Noise level 0

### Usage
**Note:** SAN is scale-depedent model so you need to use corresponding model for specified scale factor. Using x2 model to do a x3/x4/x8 upscale produces **VERY** noisy samples.

    python test.py --scale 2 --checkpoint path/to/model.pth --image path/to/image.jpg --output SR_output_x2.png

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
