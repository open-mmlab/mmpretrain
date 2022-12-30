# CLIP

> [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on. We release our code and pre-trained model weights at this https URL.

<!-- [IMAGE] -->

<div align=center>
<img src="https://raw.githubusercontent.com/Scarecrow0/figures_cache/main/clip_main_fig.png" width="100%"/>
</div>

## Results and models

### ImageNet-1k

|                     Model                      |        Pretrain         | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                     Config                      |                     Download                      |
| :--------------------------------------------: | :---------------------: | :-------: | :------: | :-------: | :-------: | :---------------------------------------------: | :-----------------------------------------------: |
| clip-vit-base-p32_laion2b-in12k-pre_3rdparty_in1k\* | LAION-2B & ImageNet-12k |   88.22   |   4.36   |   83.06   |   96.49   |   [config](./vit-base-p32_pt-64xb64_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/clip/clip-vit-base-p32_laion2b-in12k-pre_3rdparty_in1k_20221220-b384e830.pth) |
| clip-vit-base-p32_laion2b-pre_3rdparty_in1k\*  |        LAION-2B         |   88.22   |   4.36   |   82.46   |   96.12   |   [config](./vit-base-p32_pt-64xb64_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/clip/clip-vit-base-p32_laion2b-pre_3rdparty_in1k_20221220-194df57f.pth) |
|  clip-vit-base-p32_openai-pre_3rdparty_in1k\*  |         OpenAI          |   88.22   |   4.36   |   81.77   |   95.89   |   [config](./vit-base-p32_pt-64xb64_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/clip/clip-vit-base-p32_openai-pre_3rdparty_in1k_20221220-a0182ba9.pth) |
| clip-vit-base-p32_laion2b-in12k-pre_3rdparty_in1k-384px\* | LAION-2B & ImageNet-12k |   88.22   |  12.66   |   85.39   |   97.67   | [config](./vit-base-p32_pt-64xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/clip/clip-vit-base-p32_laion2b-in12k-pre_3rdparty_in1k-384px_20221220-c7757552.pth) |
| clip-vit-base-p32_openai-in12k-pre_3rdparty_in1k-384px\* |  OpenAI & ImageNet-12k  |   88.22   |  12.66   |   85.13   |   97.42   | [config](./vit-base-p32_pt-64xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/clip/clip-vit-base-p32_openai-in12k-pre_3rdparty_in1k-384px_20221220-dc2e49ea.pth) |
| clip-vit-base-p16_laion2b-in12k-pre_3rdparty_in1k\* | LAION-2B & ImageNet-12k |   86.57   |  16.86   |   86.02   |   97.76   |   [config](./vit-base-p16_pt-64xb64_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/clip/clip-vit-base-p16_laion2b-in12k-pre_3rdparty_in1k_20221220-a5e31f8c.pth) |
| clip-vit-base-p16_laion2b-pre_3rdparty_in1k\*  |        LAION-2B         |   86.57   |  16.86   |   85.49   |   97.59   |   [config](./vit-base-p16_pt-64xb64_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/clip/clip-vit-base-p16_laion2b-pre_3rdparty_in1k_20221220-5e24ff58.pth) |
| clip-vit-base-p16_openai-in12k-pre_3rdparty_in1k\* |  OpenAI & ImageNet-12k  |   86.57   |  16.86   |   85.99   |   97.72   |   [config](./vit-base-p16_pt-64xb64_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/clip/clip-vit-base-p16_openai-in12k-pre_3rdparty_in1k_20221220-90d930a8.pth) |
|  clip-vit-base-p16_openai-pre_3rdparty_in1k\*  |         OpenAI          |   86.57   |  16.86   |   85.30   |   97.50   |   [config](./vit-base-p16_pt-64xb64_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/clip/clip-vit-base-p16_openai-pre_3rdparty_in1k_20221220-c7d9c899.pth) |
| clip-vit-base-p32_laion2b-in12k-pre_3rdparty_in1k-448px\* | LAION-2B & ImageNet-12k |   88.22   |  17.20   |   85.76   |   97.63   | [config](./vit-base-p32_pt-64xb64_in1k-448px.py) | [model](https://download.openmmlab.com/mmclassification/v0/clip/clip-vit-base-p32_laion2b-in12k-pre_3rdparty_in1k-448px_20221220-ca404a7d.pth) |
| clip-vit-base-p16_laion2b-in12k-pre_3rdparty_in1k-384px\* | LAION-2B & ImageNet-12k |   86.57   |  49.37   |   87.17   |   98.02   | [config](./vit-base-p16_pt-64xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/clip/clip-vit-base-p16_laion2b-in12k-pre_3rdparty_in1k-384px_20221220-84ed0cc0.pth) |
| clip-vit-base-p16_laion2b-pre_3rdparty_in1k-384px\* |        LAION-2B         |   86.57   |  49.37   |   86.52   |   97.97   | [config](./vit-base-p16_pt-64xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/clip/clip-vit-base-p16_laion2b-pre_3rdparty_in1k-384px_20221220-558ed826.pth) |
| clip-vit-base-p16_openai-in12k-pre_3rdparty_in1k-384px\* |  OpenAI & ImageNet-12k  |   86.57   |  49.37   |   86.87   |   98.05   | [config](./vit-base-p16_pt-64xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/clip/clip-vit-base-p16_openai-in12k-pre_3rdparty_in1k-384px_20221220-8df86b74.pth) |
| clip-vit-base-p16_openai-pre_3rdparty_in1k-384px\* |         OpenAI          |   86.57   |  49.37   |   86.25   |   97.90   | [config](./vit-base-p16_pt-64xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/clip/clip-vit-base-p16_openai-pre_3rdparty_in1k-384px_20221220-eb012e87.pth) |

*Models with * are converted from the [official repo](https://github.com/rwightman/pytorch-image-models). The config files of these models are only for inference. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

## Citation

```bibtex
@InProceedings{pmlr-v139-radford21a,
title = {Learning Transferable Visual Models From Natural Language Supervision},
author = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
booktitle = {Proceedings of the 38th International Conference on Machine Learning},
year = {2021},
series = {Proceedings of Machine Learning Research},
publisher = {PMLR},
}
```
