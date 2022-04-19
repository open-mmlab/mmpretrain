# ConvMixer

> [Patches Are All You Need?](https://arxiv.org/abs/2201.09792)
<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->
Although convolutional networks have been the dominant architecture for vision tasks for many years, recent experiments have shown that Transformer-based models, most notably the Vision Transformer (ViT), may exceed their performance in some settings. However, due to the quadratic runtime of the self-attention layers in Transformers, ViTs require the use of patch embeddings, which group together small regions of the image into single input features, in order to be applied to larger image sizes. This raises a question: Is the performance of ViTs due to the inherently-more-powerful Transformer architecture, or is it at least partly due to using patches as the input representation? In this paper, we present some evidence for the latter: specifically, we propose the ConvMixer, an extremely simple model that is similar in spirit to the ViT and the even-more-basic MLP-Mixer in that it operates directly on patches as input, separates the mixing of spatial and channel dimensions, and maintains equal size and resolution throughout the network. In contrast, however, the ConvMixer uses only standard convolutions to achieve the mixing steps. Despite its simplicity, we show that the ConvMixer outperforms the ViT, MLP-Mixer, and some of their variants for similar parameter counts and data set sizes, in addition to outperforming classical vision models such as the ResNet.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/42952108/156284977-abf2245e-d9ba-4e0d-8e10-c0664a20f4c8.png" width="100%"/>
</div>

## Results and models

### ImageNet-1k

|      Model      | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------------:|:---------:|:--------:|:---------:|:---------:|:------:|:--------:|
| ConvMixer-768/32\*    | 21.11 | 19.62 | 80.16 | 95.08  | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convmixer/convmixer-768-32_10xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convmixer/convmixer-768-32_3rdparty_10xb64_in1k_20220323-bca1f7b8.pth) |
| ConvMixer-1024/20\*    | 24.38 | 5.55 | 76.94 | 93.36  | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convmixer/convmixer-1024-20_10xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convmixer/convmixer-1024-20_3rdparty_10xb64_in1k_20220323-48f8aeba.pth) |
| ConvMixer-1536/20\*    | 51.63 | 48.71 | 81.37 | 95.61 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convmixer/convmixer-1536-20_10xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convmixer/convmixer-1536_20_3rdparty_10xb64_in1k_20220323-ea5786f3.pth) |


*Models with \* are converted from the [official repo](https://github.com/locuslab/convmixer). The config files of these models are only for inference. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*


## Citation

```bibtex
@misc{trockman2022patches,
      title={Patches Are All You Need?},
      author={Asher Trockman and J. Zico Kolter},
      year={2022},
      eprint={2201.09792},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
