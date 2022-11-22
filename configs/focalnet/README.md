# Focal Modulation Networks

> [Focal Modulation Networks](https://arxiv.org/abs/2203.11926)

<!-- [ALGORITHM] -->

## Abstract

We propose focal modulation networks (FocalNets in short), where self-attention (SA) is completely replaced by a focal modulation mechanism for modeling token interactions in vision. Focal modulation comprises three components: (i) hierarchical contextualization, implemented using a stack of depth-wise convolutional layers, to encode visual contexts from short to long ranges, (ii) gated aggregation to selectively gather contexts for each query token based on its content, and (iii) element-wise modulation or affine transformation to inject the aggregated context into the query. Extensive experiments show FocalNets outperform the state-of-the-art SA counterparts (e.g., Swin and Focal Transformers) with similar computational costs on the tasks of image classification, object detection, and segmentation. Specifically, FocalNets with tiny and base size achieve 82.3% and 83.9% top-1 accuracy on ImageNet-1K. After pretrained on ImageNet-22K in 224 resolution, it attains 86.5% and 87.3% top-1 accuracy when finetuned with resolution 224 and 384, respectively. When transferred to downstream tasks, FocalNets exhibit clear superiority. For object detection with Mask R-CNN, FocalNet base trained with 1\times outperforms the Swin counterpart by 2.1 points and already surpasses Swin trained with 3\times schedule (49.0 v.s. 48.5). For semantic segmentation with UPerNet, FocalNet base at single-scale outperforms Swin by 2.4, and beats Swin at multi-scale (50.5 v.s. 49.7). Using large FocalNet and Mask2former, we achieve 58.5 mIoU for ADE20K semantic segmentation, and 57.9 PQ for COCO Panoptic Segmentation. Using huge FocalNet and DINO, we achieved 64.3 and 64.4 mAP on COCO minival and test-dev, respectively, establishing new SoTA on top of much larger attention-based models like Swinv2-G and BEIT-3.

<div align=center>
<img src="https://user-images.githubusercontent.com/24734142/203196289-87773b63-0bd0-4ab5-88e8-850a6abe1d57.png" width="80%"/>
</div>

## Results and models

### ImageNet-1k

|      Model       |   Pretrain   | resolution | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                                                        Config                                                        |  Download   |
| :--------------: | :----------: | :--------: | :-------: | :------: | :-------: | :-------: | :------------------------------------------------------------------------------------------------------------------: | :---------: |
| FocalNet-T-SRF\* | From scratch |  224x224   |   28.43   |   4.42   |   82.05   |   95.95   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/focalnet/focalnet-tiny-srf_8xb64_in1k.py) | [model](<>) |
| FocalNet-T-LRF\* | From scratch |  224x224   |   28.64   |   4.49   |   82.07   |   95.88   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/focalnet/focalnet-tiny-lrf_8xb64_in1k.py) | [model](<>) |
| FocalNet-S-SRF\* | From scratch |  224x224   |   48.49   |   8.62   |   83.37   |   96.44   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/focalnet/focalnet-small-srf_8xb64_in1k.py) | [model](<>) |
| FocalNet-S-LRF\* | From scratch |  224x224   |   50.34   |   8.74   |   83.52   |   96.47   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/focalnet/focalnet-small-lrf_8xb64_in1k.py) | [model](<>) |
| FocalNet-B-SRF\* | From scratch |  224x224   |   88.15   |  15.28   |   83.70   |   96.61   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/focalnet/focalnet-base-srf_8xb64_in1k.py) | [model](<>) |
| FocalNet-B-LRF\* | From scratch |  224x224   |   88.75   |  15.43   |   83.75   |   96.60   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/focalnet/focalnet-base-lrf_8xb64_in1k.py) | [model](<>) |

\*Models with * are converted from [the official repo](https://github.com/microsoft/FocalNet). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.

### Pre-trained Models

The pre-trained models on ImageNet-21k are used to fine-tune on the downstream tasks.

|       Model       |   Pretrain   | resolution | Params(M) | Flops(G) |  Download   |
| :---------------: | :----------: | :--------: | :-------: | :------: | :---------: |
| FocalNet-L-FL3\*  | ImageNet-21k |  384x384   |           |          | [model](<>) |
| FocalNet-L-FL4\*  | ImageNet-21k |  384x384   |           |          | [model](<>) |
| FocalNet-XL-FL3\* | ImageNet-21k |  384x384   |           |          | [model](<>) |
| FocalNet-XL-FL4\* | ImageNet-21k |  384x384   |           |          | [model](<>) |
| FocalNet-H-FL3\*  | ImageNet-21k |  224x224   |           |          | [model](<>) |
| FocalNet-H-FL4\*  | ImageNet-21k |  224x224   |           |          | [model](<>) |

\*Models with * are converted from [the official repo](https://github.com/microsoft/FocalNet).

## Citation

```
@misc{yang2022focal,
      title={Focal Modulation Networks},
      author={Jianwei Yang and Chunyuan Li and Xiyang Dai and Jianfeng Gao},
      journal={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2022}
}
```
