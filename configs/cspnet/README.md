# CSPNet

> [CSPNet: A New Backbone that can Enhance Learning Capability of CNN](https://arxiv.org/abs/1911.11929)
<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->
Neural networks have enabled state-of-the-art approaches to achieve incredible results on computer vision tasks such as object detection. However, such success greatly relies on costly computation resources, which hinders people with cheap devices from appreciating the advanced technology. In this paper, we propose Cross Stage Partial Network (CSPNet) to mitigate the problem that previous works require heavy inference computations from the network architecture perspective. We attribute the problem to the duplicate gradient information within network optimization. The proposed networks respect the variability of the gradients by integrating feature maps from the beginning and the end of a network stage, which, in our experiments, reduces computations by 20% with equivalent or even superior accuracy on the ImageNet dataset, and significantly outperforms state-of-the-art approaches in terms of AP50 on the MS COCO object detection dataset. The CSPNet is easy to implement and general enough to cope with architectures based on ResNet, ResNeXt, and DenseNet. Source code is at this https URL.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/18586273/159420842-6147c687-a488-460c-8bb2-4ea5276c26c7.png" width="60%"/>
</div>

## Results and models

### ImageNet-1k

|      Model      |   Pretrain   | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------------:|:------------:|:---------:|:--------:|:---------:|:---------:|:------:|:--------:|
| CSPDarkNet50\*  | From scratch | 27.64 | 5.04 | 80.05 | 95.07  | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/cspnet/cspdarknet50_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/cspnet/cspdarknet50_3rdparty_8xb32_in1k_20220329-bd275287.pth) |
|  CSPResNet50\*  | From scratch | 21.62 | 3.48 | 79.55 | 94.68  | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/cspnet/cspresnet50_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/cspnet/cspresnet50_3rdparty_8xb32_in1k_20220329-dd6dddfb.pth) |
|  CSPResNeXt50\* | From scratch | 20.57 | 3.11 | 79.96 | 94.96 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/cspnet/cspresnext50_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/cspnet/cspresnext50_3rdparty_8xb32_in1k_20220329-2cc84d21.pth) |


*Models with \* are converted from the [timm repo](https://github.com/rwightman/pytorch-image-models). The config files of these models are only for inference. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

## Citation

```bibtex
@inproceedings{wang2020cspnet,
  title={CSPNet: A new backbone that can enhance learning capability of CNN},
  author={Wang, Chien-Yao and Liao, Hong-Yuan Mark and Wu, Yueh-Hua and Chen, Ping-Yang and Hsieh, Jun-Wei and Yeh, I-Hau},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops},
  pages={390--391},
  year={2020}
}
```
