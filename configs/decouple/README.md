# Decouple

> [Decoupling Representation and Classifier for Long-Tailed Recognition](https://arxiv.org/abs/1910.09217)

<!-- [ALGORITHM] -->

## Abstract

The long-tail distribution of the visual world poses great challenges for deep
learning based classification models on how to handle the class imbalance problem. Existing solutions usually involve class-balancing strategies, e.g. by loss
re-weighting, data re-sampling, or transfer learning from head- to tail-classes, but
most of them adhere to the scheme of jointly learning representations and classifiers. In this work, we decouple the learning procedure into representation learning and classification, and systematically explore how different balancing strategies affect them for long-tailed recognition. The findings are surprising: (1) data
imbalance might not be an issue in learning high-quality representations; (2) with
representations learned with the simplest instance-balanced (natural) sampling, it
is also possible to achieve strong long-tailed recognition ability by adjusting only
the classifier. We conduct extensive experiments and set new state-of-the-art performance on common long-tailed benchmarks like ImageNet-LT, Places-LT and
iNaturalist, showing that it is possible to outperform carefully designed losses,
sampling strategies, even complex modules with memory, by using a straightforward approach that decouples representation and classification. Our code is avail-
able at https://github.com/facebookresearch/classifier-balancing.

## Results and models

### ImageNet-LT

|    Model     | Params(M) | Flops(G) | Top-1 (%) |                                        Config                                         |  Download   |
| :----------: | :-------: | :------: | :-------: | :-----------------------------------------------------------------------------------: | :---------: |
|    Joint     |   25.03   |   5.56   |   46.15   | [config](./resnext50_decouple-representation_8xb128-instance-balanced_imagenet-lt.py) | [model](<>) |
|     cRT      |   25.03   |   5.56   |   48.40   |          [config](./resnext50_decouple-classifier_8xb128-crt_imagenet-lt.py)          | [model](<>) |
| τ-normalized |   25.03   |   5.56   |   50.29   |          [config](./resnext50_decouple-classifier_1xb512-tau_imagenet-lt.py)          | [model](<>) |
|     LWS      |   25.03   |   5.56   |   48.47   |          [config](./resnext50_decouple-classifier_8xb128-lws_imagenet-lt.py)          | [model](<>) |

Note: cRT、LWS Inference accuracy is a bit lower than paper result Due to the difference between our implementation and the paper's implementation of class-balanced sampling.

## Citation

```
@inproceedings{kang2019decoupling,
  title={Decoupling representation and classifier for long-tailed recognition},
  author={Kang, Bingyi and Xie, Saining and Rohrbach, Marcus and Yan, Zhicheng
          and Gordo, Albert and Feng, Jiashi and Kalantidis, Yannis},
  booktitle={Eighth International Conference on Learning Representations (ICLR)},
  year={2020}
}
```
