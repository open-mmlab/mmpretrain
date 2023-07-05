# DINOv2

> [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)

<!-- [ALGORITHM] -->

## Abstract

The recent breakthroughs in natural language processing for model pretraining on large quantities of data have opened the way for similar foundation models in computer vision. These models could greatly simplify the use of images in any system by producing allpurpose visual features, i.e., features that work across image distributions and tasks without finetuning. This work shows that existing pretraining methods, especially self-supervised methods, can produce such features if trained on enough curated data from diverse sources. We revisit existing approaches and combine different techniques to scale our pretraining in terms of data and model size. Most of the technical contributions aim at accelerating and stabilizing the training at scale. In terms of data, we propose an automatic pipeline to build a dedicated, diverse, and curated image dataset instead of uncurated data, as typically done in the self-supervised literature. In terms of models, we train a ViT model (Dosovitskiy et al., 2020) with 1B parameters and distill it into a series of smaller models that surpass the best available all-purpose features, OpenCLIP (Ilharco et al., 2021) on most of the benchmarks at image and pixel levels.

<div align=center>
<img src="https://user-images.githubusercontent.com/36138628/234560516-b495795c-c75c-444c-a712-bb61a3de444e.png" width="70%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('vit-small-p14_dinov2-pre_3rdparty', pretrained=True)
inputs = torch.rand(1, 3, 224, 224)
out = model(inputs)
print(type(out))
# To extract features.
feats = model.extract_feat(inputs)
print(type(feats))
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                 | Params (M) | Flops (G) |                     Config                     |                                              Download                                              |
| :------------------------------------ | :--------: | :-------: | :--------------------------------------------: | :------------------------------------------------------------------------------------------------: |
| `vit-small-p14_dinov2-pre_3rdparty`\* |   22.06    |   46.76   | [config](vit-small-p14_dinov2-pre_headless.py) | [model](https://download.openmmlab.com/mmpretrain/v1.0/dinov2/vit-small-p14_dinov2-pre_3rdparty_20230426-5641ca5a.pth) |
| `vit-base-p14_dinov2-pre_3rdparty`\*  |   86.58    |  152.00   | [config](vit-base-p14_dinov2-pre_headless.py)  | [model](https://download.openmmlab.com/mmpretrain/v1.0/dinov2/vit-base-p14_dinov2-pre_3rdparty_20230426-ba246503.pth) |
| `vit-large-p14_dinov2-pre_3rdparty`\* |   304.00   |  507.00   | [config](vit-large-p14_dinov2-pre_headless.py) | [model](https://download.openmmlab.com/mmpretrain/v1.0/dinov2/vit-large-p14_dinov2-pre_3rdparty_20230426-f3302d9e.pth) |
| `vit-giant-p14_dinov2-pre_3rdparty`\* |  1136.00   |  1784.00  | [config](vit-giant-p14_dinov2-pre_headless.py) | [model](https://download.openmmlab.com/mmpretrain/v1.0/dinov2/vit-giant-p14_dinov2-pre_3rdparty_20230426-2934a630.pth) |

*Models with * are converted from the [official repo](https://github.com/facebookresearch/dinov2). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```
