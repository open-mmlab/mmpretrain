# VIG

> [Vision GNN: An Image is Worth Graph of Nodes](https://arxiv.org/abs/2206.00272)

<!-- [ALGORITHM] -->

## Abstract

Network architecture plays a key role in the deep learning-based computer vision system. The widely-used convolutional neural network and transformer treat the image as a grid or sequence structure, which is not flexible to capture irregular and complex objects. In this paper, we propose to represent the image as a graph structure and introduce a new Vision GNN (ViG) architecture to extract graph-level feature for visual tasks. We first split the image to a number of patches which are viewed as nodes, and construct a graph by connecting the nearest neighbors. Based on the graph representation of images, we build our ViG model to transform and exchange information among all the nodes. ViG consists of two basic modules: Grapher module with graph convolution for aggregating and updating graph information, and FFN module with two linear layers for node feature transformation. Both isotropic and pyramid architectures of ViG are built with different model sizes. Extensive experiments on image recognition and object detection tasks demonstrate the superiority of our ViG architecture. We hope this pioneering study of GNN on general visual tasks will provide useful inspiration and experience for future research. The PyTorch code is available at this https URL and the MindSpore code is available at this https URL.

## Results and models

### ImageNet-1k

|       Model        | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                Config                 |  Download   |
| :----------------: | :-------: | :------: | :-------: | :-------: | :-----------------------------------: | :---------: |
|      VIG-tiny      |   7.19    |  1.301   |   74.31   |   92.32   |  [config](./vig_tiny_8xb32_in1k.py)   | [model](<>) |
|     VIG-small      |   22.75   |   4.54   |   80.57   |   95.23   |  [config](./vig_small_8xb32_in1k.py)  | [model](<>) |
|      VIG-base      |   20.69   |  17.68   |   82.51   |   96.04   |  [config](./vig_base_8xb32_in1k.py)   | [model](<>) |
|  Pyramid-Vig-tiny  |   9.46    |   1.71   |   78.44   |   94.36   |  [config](./pvig_tiny_8xb32_in1k.py)  | [model](<>) |
| Pyramid-Vig-small  |   29.02   |   4.57   |   81.94   |   95.98   | [config](./pvig_small_8xb32_in1k.py)  | [model](<>) |
| Pyramid-Vig-medium |   51.68   |   8.89   |   82.96   |   96.36   | [config](./pvig_medium_8xb32_in1k.py) | [model](<>) |
|  Pyramid-Vig-base  |   95.21   |  16.86   |   83.49   |   96.56   |  [config](./pvig_base_8xb32_in1k.py)  | [model](<>) |

## Citation

```
@inproceedings{han2022vig,
  title={Vision GNN: An Image is Worth Graph of Nodes},
  author={Kai Han and Yunhe Wang and Jianyuan Guo and Yehui Tang and Enhua Wu},
  booktitle={NeurIPS},
  year={2022}
}
```
