# EfficientFormer

> [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)

<!-- [ALGORITHM] -->

## Abstract

Vision Transformers (ViT) have shown rapid progress in computer vision tasks, achieving promising results on various benchmarks. However, due to the massive number of parameters and model design, e.g., attention mechanism, ViT-based models are generally times slower than lightweight convolutional networks. Therefore, the deployment of ViT for real-time applications is particularly challenging, especially on resource-constrained hardware such as mobile devices. Recent efforts try to reduce the computation complexity of ViT through network architecture search or hybrid design with MobileNet block, yet the inference speed is still unsatisfactory. This leads to an important question: can transformers run as fast as MobileNet while obtaining high performance? To answer this, we first revisit the network architecture and operators used in ViT-based models and identify inefficient designs. Then we introduce a dimension-consistent pure transformer (without MobileNet blocks) as a design paradigm.  Finally, we perform latency-driven slimming to get a series of final models dubbed EfficientFormer. Extensive experiments show the superiority of EfficientFormer in performance and speed on mobile devices. Our fastest model, EfficientFormer-L1, achieves 79.2% top-1 accuracy on ImageNet-1K with only 1.6 ms inference latency on iPhone 12 (compiled with CoreML), which runs as fast as MobileNetV2×1.4 (1.6 ms, 74.7% top-1), and our largest model, EfficientFormer-L7, obtains 83.3% accuracy with only 7.0 ms latency. Our work proves that properly designed transformers can reach extremely low latency on mobile devices while maintaining high performance.

<div align=center>
<img src="https://user-images.githubusercontent.com/18586273/180713426-9d3d77e3-3584-42d8-9098-625b4170d796.png" width="100%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('efficientformer-l1_3rdparty_8xb128_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('efficientformer-l1_3rdparty_8xb128_in1k', pretrained=True)
inputs = torch.rand(1, 3, 224, 224)
out = model(inputs)
print(type(out))
# To extract features.
feats = model.extract_feat(inputs)
print(type(feats))
```

**Test Command**

Prepare your dataset according to the [docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset).

Test:

```shell
python tools/test.py configs/efficientformer/efficientformer-l1_8xb128_in1k.py https://download.openmmlab.com/mmclassification/v0/efficientformer/efficientformer-l1_3rdparty_in1k_20220915-cc3e1ac6.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                                       |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                   Config                    |                             Download                              |
| :------------------------------------------ | :----------: | :--------: | :-------: | :-------: | :-------: | :-----------------------------------------: | :---------------------------------------------------------------: |
| `efficientformer-l1_3rdparty_8xb128_in1k`\* | From scratch |   12.28    |   1.30    |   80.46   |   94.99   | [config](efficientformer-l1_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientformer/efficientformer-l1_3rdparty_in1k_20220915-cc3e1ac6.pth) |
| `efficientformer-l3_3rdparty_8xb128_in1k`\* | From scratch |   31.41    |   3.74    |   82.45   |   96.18   | [config](efficientformer-l3_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientformer/efficientformer-l3_3rdparty_in1k_20220915-466793d6.pth) |
| `efficientformer-l7_3rdparty_8xb128_in1k`\* | From scratch |   82.23    |   10.16   |   83.40   |   96.60   | [config](efficientformer-l7_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientformer/efficientformer-l7_3rdparty_in1k_20220915-185e30af.pth) |

*Models with * are converted from the [official repo](https://github.com/snap-research/EfficientFormer). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@misc{https://doi.org/10.48550/arxiv.2206.01191,
  doi = {10.48550/ARXIV.2206.01191},

  url = {https://arxiv.org/abs/2206.01191},

  author = {Li, Yanyu and Yuan, Geng and Wen, Yang and Hu, Eric and Evangelidis, Georgios and Tulyakov, Sergey and Wang, Yanzhi and Ren, Jian},

  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},

  title = {EfficientFormer: Vision Transformers at MobileNet Speed},

  publisher = {arXiv},

  year = {2022},

  copyright = {Creative Commons Attribution 4.0 International}
}
```
