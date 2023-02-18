# VIG

> [Vision GNN: An Image is Worth Graph of Nodes](https://arxiv.org/abs/2206.00272)

<!-- [ALGORITHM] -->

## Abstract

Network architecture plays a key role in the deep learning-based computer vision system. The widely-used convolutional neural network and transformer treat the image as a grid or sequence structure, which is not flexible to capture irregular and complex objects. In this paper, we propose to represent the image as a graph structure and introduce a new Vision GNN (ViG) architecture to extract graph-level feature for visual tasks. We first split the image to a number of patches which are viewed as nodes, and construct a graph by connecting the nearest neighbors. Based on the graph representation of images, we build our ViG model to transform and exchange information among all the nodes. ViG consists of two basic modules: Grapher module with graph convolution for aggregating and updating graph information, and FFN module with two linear layers for node feature transformation. Both isotropic and pyramid architectures of ViG are built with different model sizes. Extensive experiments on image recognition and object detection tasks demonstrate the superiority of our ViG architecture. We hope this pioneering study of GNN on general visual tasks will provide useful inspiration and experience for future research.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/212789461-f085e4da-9ce9-435f-93c0-e1b84d10b79f.png" width="50%"/>
</div>

## Results and models

### ImageNet-1k

|            Model            |   Pretrain   | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                 Config                 |                                         Download                                         |
| :-------------------------: | :----------: | :-------: | :------: | :-------: | :-------: | :------------------------------------: | :--------------------------------------------------------------------------------------: |
|  vig-tiny_3rdparty_in1k\*   | From scratch |   7.18    |   1.31   |   74.40   |   92.34   |  [config](./vig-tiny_8xb128_in1k.py)   | [model](https://download.openmmlab.com/mmclassification/v0/vig/vig-tiny_3rdparty_in1k_20230117-6414c684.pth) |
|  vig-small_3rdparty_in1k\*  | From scratch |   22.75   |   4.54   |   80.61   |   95.28   |  [config](./vig-small_8xb128_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/vig/vig-small_3rdparty_in1k_20230117-5338bf3b.pth) |
|  vig-base_3rdparty_in1k\*   | From scratch |   20.68   |  17.68   |   82.64   |   96.04   |  [config](./vig-base_8xb128_in1k.py)   | [model](https://download.openmmlab.com/mmclassification/v0/vig/vig-base_3rdparty_in1k_20230117-92f6f12f.pth) |
|  pvig-tiny_3rdparty_in1k\*  | From scratch |   9.46    |   1.71   |   78.38   |   94.38   |  [config](./pvig-tiny_8xb128_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/vig/pvig-tiny_3rdparty_in1k_20230117-eb77347d.pth) |
| pvig-small_3rdparty_in1k\*  | From scratch |   29.02   |   4.57   |   82.00   |   95.97   | [config](./pvig-small_8xb128_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/vig/pvig-small_3rdparty_in1k_20230117-9433dc96.pth) |
| pvig-medium_3rdparty_in1k\* | From scratch |   51.68   |   8.89   |   83.12   |   96.35   | [config](./pvig-medium_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/vig/pvig-medium_3rdparty_in1k_20230117-21057a6d.pth) |
|  pvig-base_3rdparty_in1k\*  | From scratch |   95.21   |  16.86   |   83.59   |   96.52   |  [config](./pvig-base_8xb128_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/vig/pvig-base_3rdparty_in1k_20230117-dbab3c85.pth) |

*Models with * are converted from the [official repo](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch). The config files of these models are only for inference. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

## Citation

```bibtex
@inproceedings{han2022vig,
  title={Vision GNN: An Image is Worth Graph of Nodes},
  author={Kai Han and Yunhe Wang and Jianyuan Guo and Yehui Tang and Enhua Wu},
  booktitle={NeurIPS},
  year={2022}
}
```
