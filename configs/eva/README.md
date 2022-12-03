# EVA

> [EVA: Exploring the Limits of Masked Visual Representation Learning at Scale](https://arxiv.org/abs/2211.07636)

<!-- [ALGORITHM] -->

## Abstract

We launch EVA, a vision-centric foundation model to explore the limits of visual representation at scale using only publicly accessible data. EVA is a vanilla ViT pre-trained to reconstruct the masked out image-text aligned vision features conditioned on visible image patches. Via this pretext task, we can efficiently scale up EVA to one billion parameters, and sets new records on a broad range of representative vision downstream tasks, such as image recognition, video action recognition, object detection, instance segmentation and semantic segmentation without heavy supervised training. Moreover, we observe quantitative changes in scaling EVA result in qualitative changes in transfer learning performance that are not present in other models. For instance, EVA takes a great leap in the challenging large vocabulary instance segmentation task: our model achieves almost the same state-of-the-art performance on LVISv1.0 dataset with over a thousand categories and COCO dataset with only eighty categories. Beyond a pure vision encoder, EVA can also serve as a vision-centric, multi-modal pivot to connect images and text. We find initializing the vision tower of a giant CLIP from EVA can greatly stabilize the training and outperform the training from scratch counterpart with much fewer samples and less compute, providing a new direction for scaling up and accelerating the costly training of multi-modal foundation models.

<div align="center">
<img src="https://user-images.githubusercontent.com/24734142/205410193-f1164e56-c117-4165-86f5-4cbfd797bc87.png" width="70%"/>
</div>

## Results and models

### merged-30M

The pre-trained models on merged-30M are used to fine-tune, and therefore don't have evaluation results.

| Model | resolution | patch size |  Download   |
| :---: | :--------: | :--------: | :---------: |
| EVA-G |  224x224   |     14     | [model](<>) |
| EVA-G |  224x224   |  14 to 16  | [model](<>) |

### ImageNet-21k

The pre-trained models on ImageNet-21k are used to fine-tune, and therefore don't have evaluation results.

| Model |  Pretrain  | resolution |  Download   |
| :---: | :--------: | :--------: | :---------: |
| EVA-G | merged-30M |  224x224   | [model](<>) |

### ImageNet-1k

| Model |         Pretrain          | resolution | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |               Config                |  Download   |
| :---: | :-----------------------: | :--------: | :-------: | :------: | :-------: | :-------: | :---------------------------------: | :---------: |
| EVA-G | merged-30M + ImageNet-21k |  336x336   |  1013.01  |  620.64  |   89.61   |   98.93   | [config](./eva-g-336_8xb16_in1k.py) | [model](<>) |
| EVA-G | merged-30M + ImageNet-21k |  560x560   |  1014.45  | 1906.76  |   89.71   |   98.96   | [config](./eva-g-560_8xb16_in1k.py) | [model](<>) |

*Models with * are converted from the [official repo](https://github.com/baaivision/EVA). The config files of these models are only for inference.*

## Citation

```bibtex
@article{EVA,
  title={EVA: Exploring the Limits of Masked Visual Representation Learning at Scale},
  author={Fang, Yuxin and Wang, Wen and Xie, Binhui and Sun, Quan and Wu, Ledell and Wang, Xinggang and Huang, Tiejun and Wang, Xinlong and Cao, Yue},
  journal={arXiv preprint arXiv:2211.07636},
  year={2022}
}
```
