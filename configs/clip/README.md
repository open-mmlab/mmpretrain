# ConvMixer

> [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on. We release our code and pre-trained model weights at this https URL. 

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/42952108/156284977-abf2245e-d9ba-4e0d-8e10-c0664a20f4c8.png" width="100%"/>
</div>

## Results and models

### ImageNet-1k

|        Model        | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                                  Config                                  |                                  Download                                  |
| :-----------------: | :-------: | :------: | :-------: | :-------: | :----------------------------------------------------------------------: | :------------------------------------------------------------------------: |
| ConvMixer-768/32\*  |   21.11   |  19.62   |   80.16   |   95.08   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convmixer/convmixer-768-32_10xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convmixer/convmixer-768-32_3rdparty_10xb64_in1k_20220323-bca1f7b8.pth) |
| ConvMixer-1024/20\* |   24.38   |   5.55   |   76.94   |   93.36   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convmixer/convmixer-1024-20_10xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convmixer/convmixer-1024-20_3rdparty_10xb64_in1k_20220323-48f8aeba.pth) |
| ConvMixer-1536/20\* |   51.63   |  48.71   |   81.37   |   95.61   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convmixer/convmixer-1536-20_10xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convmixer/convmixer-1536_20_3rdparty_10xb64_in1k_20220323-ea5786f3.pth) |

*Models with * are converted from the [official repo](https://github.com/locuslab/convmixer). The config files of these models are only for inference. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

## Citation

```bibtex
@misc{https://doi.org/10.48550/arxiv.2103.00020,
  doi = {10.48550/ARXIV.2103.00020},
  
  url = {https://arxiv.org/abs/2103.00020},
  
  author = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Learning Transferable Visual Models From Natural Language Supervision},
  
  publisher = {arXiv},
  
  year = {2021},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
