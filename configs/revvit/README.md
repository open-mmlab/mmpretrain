# Reversible Vision Transformers

> [Reversible Vision Tranformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Mangalam_Reversible_Vision_Transformers_CVPR_2022_paper.pdf)

<!-- [ALGORITHM] -->

## Abstract

We present Reversible Vision Transformers, a memory efficient architecture design for visual recognition. By decoupling the GPU memory footprint from the depth of the model, Reversible Vision Transformers enable memory efficient scaling of transformer architectures. We adapt two popular models, namely Vision Transformer and Multiscale Vision Transformers, to reversible variants and benchmark extensively across both model sizes and tasks of image classification, object detection and video classification. Reversible Vision Transformers achieve a reduced memory footprint of up to 15.5× at identical model complexity, parameters and accuracy, demonstrating the promise of reversible vision transformers as an efficient backbone for resource limited training regimes. Finally, we find that the additional computational burden of recomputing activations is more than overcome for deeper models, where throughput can increase up to 3.9× over their non-reversible counterparts.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/facebookresearch/SlowFast/raw/main/projects/rev/teaser.png" width="70%"/>
</div>

## Results and models

|   Model    |   Pretrain   | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                Config                |                               Download                                |
| :--------: | :----------: | :-------: | :------: | :-------: | :-------: | :----------------------------------: | :-------------------------------------------------------------------: |
| RevViT-S\* | From scratch |   28.59   |   4.46   |   82.05   |   95.86   | [config](./revvit-small_8xb256_in1k) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/) |
| RevViT-B\* | From scratch |   50.22   |   8.69   |   83.13   |   96.44   | [config](./revvit-base_8xb256_in1k)  | [model](https://download.openmmlab.com/mmclassification/v0/convnext/) |

*Models with * are converted from the [official repo](https://github.com/facebookresearch/ConvNeXt). The config files of these models are only for inference. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*
