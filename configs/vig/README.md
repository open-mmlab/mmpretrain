# VIG

> [Vision GNN: An Image is Worth Graph of Nodes](https://arxiv.org/abs/2206.00272)

<!-- [ALGORITHM] -->

## Abstract

Network architecture plays a key role in the deep learning-based computer vision system. The widely-used convolutional neural network and transformer treat the image as a grid or sequence structure, which is not flexible to capture irregular and complex objects. In this paper, we propose to represent the image as a graph structure and introduce a new Vision GNN (ViG) architecture to extract graph-level feature for visual tasks. We first split the image to a number of patches which are viewed as nodes, and construct a graph by connecting the nearest neighbors. Based on the graph representation of images, we build our ViG model to transform and exchange information among all the nodes. ViG consists of two basic modules: Grapher module with graph convolution for aggregating and updating graph information, and FFN module with two linear layers for node feature transformation. Both isotropic and pyramid architectures of ViG are built with different model sizes. Extensive experiments on image recognition and object detection tasks demonstrate the superiority of our ViG architecture. We hope this pioneering study of GNN on general visual tasks will provide useful inspiration and experience for future research. The PyTorch code is available at this https URL and the MindSpore code is available at this https URL.

<div align=center>
<img src=
</div>

## Results and models

### ImageNet-1k

|       Model        | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |    Config    |         Download         |
| :----------------: | :-------: | :------: | :-------: | :-------: | :----------: | :----------------------: |
|      VIG-tiny      |           |          |           |           | [config](<>) | [model](<>) \| [log](<>) |
|     VIG-small      |           |          |           |           | [config](<>) | [model](<>) \| [log](<>) |
|      VIG-base      |           |          |           |           | [config](<>) | [model](<>) \| [log](<>) |
|  Pyramid-Vig-tiny  |           |          |           |           | [config](<>) | [model](<>) \| [log](<>) |
| Pyramid-Vig-small  |           |          |           |           | [config](<>) | [model](<>) \| [log](<>) |
| Pyramid-Vig-medium |           |          |           |           | [config](<>) | [model](<>) \| [log](<>) |
|  Pyramid-Vig-base  |           |          |           |           | [config](<>) | [model](<>) \| [log](<>) |

## Citation

```
@inproceedings{han2022vig,
  title={Vision GNN: An Image is Worth Graph of Nodes},
  author={Kai Han and Yunhe Wang and Jianyuan Guo and Yehui Tang and Enhua Wu},
  booktitle={NeurIPS},
  year={2022}
}
```
