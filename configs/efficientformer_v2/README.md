# EfficientFormerV2

> [EfficientFormerV2: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2212.08059)

<!-- [ALGORITHM] -->

## Abstract

Vision Transformers (ViT) have shown rapid progress in computer vision tasks, achieving promising results on various benchmarks. However, due to the massive number of parameters and model design, \textit{e.g.}, attention mechanism, ViT-based models are generally times slower than lightweight convolutional networks. Therefore, the deployment of ViT for real-time applications is particularly challenging, especially on resource-constrained hardware such as mobile devices. Recent efforts try to reduce the computation complexity of ViT through network architecture search or hybrid design with MobileNet block, yet the inference speed is still unsatisfactory. This leads to an important question: can transformers run as fast as MobileNet while obtaining high performance? To answer this, we first revisit the network architecture and operators used in ViT-based models and identify inefficient designs. Then we introduce a dimension-consistent pure transformer (without MobileNet blocks) as a design paradigm. Finally, we perform latency-driven slimming to get a series of final models dubbed EfficientFormer. Extensive experiments show the superiority of EfficientFormer in performance and speed on mobile devices. Our fastest model, EfficientFormer-L1, achieves 79.2% top-1 accuracy on ImageNet-1K with only 1.6 ms inference latency on iPhone 12 (compiled with CoreML), which runs as fast as MobileNetV2×1.4 (1.6 ms, 74.7% top-1), and our largest model, EfficientFormer-L7, obtains 83.3% accuracy with only 7.0 ms latency. Our work proves that properly designed transformers can reach extremely low latency on mobile devices while maintaining high performance.

### TODO: 这里需要改成EfficientFormerV2的模型图
<div align=center>
<img src=/>
</div>

## Results and models

### ImageNet-1k

|        Model         | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                    Config                     |                                               Download                                               |
| :------------------: | :-------: | :------: | :-------: | :-------: | :-------------------------------------------: | :--------------------------------------------------------------------------------------------------: |
| EfficientFormerV2-s0\* |         |         |           |           | [config](./efficientformer_v2-s0_8xb128_in1k.py) |   |
| EfficientFormerV2-s1\* |         |         |           |           | [config](./efficientformer_v2-s1_8xb128_in1k.py) |   |
| EfficientFormerV2-s3\* |         |         |           |           | [config](./efficientformer_v2-s2_8xb128_in1k.py) |   |
| EfficientFormerV2-l\* |         |         |           |           | [config](./efficientformer_v2-l_8xb128_in1k.py) |   |

*Models with * are converted from the [official repo](https://github.com/snap-research/EfficientFormer). The config files of these models are only for inference. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

## Citation

```bibtex
@article{li2022rethinking,
  title={Rethinking Vision Transformers for MobileNet Size and Speed},
  author={Li, Yanyu and Hu, Ju and Wen, Yang and Evangelidis, Georgios and Salahi, Kamyar and Wang, Yanzhi and Tulyakov, Sergey and Ren, Jian},
  journal={arXiv preprint arXiv:2212.08059},
  year={2022}
}
```
