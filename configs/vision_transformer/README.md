# Vision Transformer

> [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)

<!-- [ALGORITHM] -->

## Abstract

While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142579081-b5718032-6581-472b-8037-ea66aaa9e278.png" width="70%"/>
</div>

## Results and models

The training step of Vision Transformers is divided into two steps. The first
step is training the model on a large dataset, like ImageNet-21k, and get the
pre-trained model. And the second step is training the model on the target
dataset, like ImageNet-1k, and get the fine-tuned model. Here, we provide both
pre-trained models and fine-tuned models.

### ImageNet-21k

The pre-trained models on ImageNet-21k are used to fine-tune, and therefore don't have evaluation results.

|   Model   | resolution | Params(M) | Flops(G) |                                                                 Download                                                                 |
| :-------: | :--------: | :-------: | :------: | :--------------------------------------------------------------------------------------------------------------------------------------: |
| ViT-B16\* |  224x224   |   86.86   |  33.03   | [model](https://download.openmmlab.com/mmclassification/v0/vit/pretrain/vit-base-p16_3rdparty_pt-64xb64_in1k-224_20210928-02284250.pth)  |
| ViT-B32\* |  224x224   |   88.30   |   8.56   | [model](https://download.openmmlab.com/mmclassification/v0/vit/pretrain/vit-base-p32_3rdparty_pt-64xb64_in1k-224_20210928-eee25dd4.pth)  |
| ViT-L16\* |  224x224   |  304.72   |  116.68  | [model](https://download.openmmlab.com/mmclassification/v0/vit/pretrain/vit-large-p16_3rdparty_pt-64xb64_in1k-224_20210928-0001f9a1.pth) |

*Models with * are converted from the [official repo](https://github.com/google-research/vision_transformer#available-vit-models).*

### ImageNet-1k

|     Model     |   Pretrain   | resolution | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                              Config                              |                              Download                              |
| :-----------: | :----------: | :--------: | :-------: | :------: | :-------: | :-------: | :--------------------------------------------------------------: | :----------------------------------------------------------------: |
|   ViT-B16\*   | ImageNet-21k |  384x384   |   86.86   |  33.03   |   85.43   |   97.77   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vision_transformer/vit-base-p16_ft-64xb64_in1k-384.py) | [model](https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth) |
|   ViT-B32\*   | ImageNet-21k |  384x384   |   88.30   |   8.56   |   84.01   |   97.08   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vision_transformer/vit-base-p32_ft-64xb64_in1k-384.py) | [model](https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth) |
|   ViT-L16\*   | ImageNet-21k |  384x384   |  304.72   |  116.68  |   85.63   |   97.63   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vision_transformer/vit-large-p16_ft-64xb64_in1k-384.py) | [model](https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-large-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-b20ba619.pth) |
| ViT-B16 (IPU) | ImageNet-21k |  224x224   |   86.86   |  33.03   |   81.22   |   95.56   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vision_transformer/vit-base-p16_ft-4xb544-ipu_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/vit/vit-base-p16_ft-4xb544-ipu_in1k_20220603-c215811a.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/vit/vit-base-p16_ft-4xb544-ipu_in1k.log) |

*Models with * are converted from the [official repo](https://github.com/google-research/vision_transformer#available-vit-models). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

## Citation

```
@inproceedings{
  dosovitskiy2021an,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=YicbFdNTTy}
}
```
