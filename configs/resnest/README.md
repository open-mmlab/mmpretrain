# ResNeSt

> [ResNeSt: Split-Attention Networks](https://arxiv.org/abs/2004.08955)

<!-- [ALGORITHM] -->

## Abstract

It is well known that featuremap attention and multi-path representation are important for visual recognition. In this paper, we present a modularized architecture, which applies the channel-wise attention on different network branches to leverage their success in capturing cross-feature interactions and learning diverse representations. Our design results in a simple and unified computation block, which can be parameterized using only a few variables. Our model, named ResNeSt, outperforms EfficientNet in accuracy and latency trade-off on image classification. In addition, ResNeSt has achieved superior transfer learning results on several public benchmarks serving as the backbone, and has been adopted by the winning entries of COCO-LVIS challenge. The source code for complete system and pretrained models are publicly available.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142573827-a8189607-614b-4385-b579-b0db148b3db7.png" width="60%"/>
</div>

## Citation

```
@misc{zhang2020resnest,
      title={ResNeSt: Split-Attention Networks},
      author={Hang Zhang and Chongruo Wu and Zhongyue Zhang and Yi Zhu and Haibin Lin and Zhi Zhang and Yue Sun and Tong He and Jonas Mueller and R. Manmatha and Mu Li and Alexander Smola},
      year={2020},
      eprint={2004.08955},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
