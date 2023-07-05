# SAM

> [Segment Anything](https://arxiv.org/abs/2304.02643)

<!-- [ALGORITHM] -->

## Abstract

We introduce the Segment Anything (SA) project: a new task, model, and dataset for image segmentation. Using our efficient model in a data collection loop, we built the largest segmentation dataset to date (by far), with over 1 billionmasks on 11M licensed and privacy respecting images. The model is designed and trained to be promptable, so it can transfer zero-shot to new image distributions and tasks. We evaluate its capabilities on numerous tasks and find that its zero-shot performance is impressive – often competitive with or even superior to prior fully supervised results. We are releasing the Segment Anything Model (SAM) and corresponding dataset (SA-1B) of 1B masks and 11M images at https://segment-anything.com to foster research into foundation models for computer vision.

<div align=center>
<img src="https://user-images.githubusercontent.com/36138628/231106092-261ff035-dd3b-4a8b-b2e7-e91f195090a1.png" width="100%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('vit-base-p16_sam-pre_3rdparty_sa1b-1024px', pretrained=True)
inputs = torch.rand(1, 3, 1024, 1024)
out = model(inputs)
print(type(out))
# To extract features.
feats = model.extract_feat(inputs)
print(type(feats))
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                          | Params (M) | Flops (G) |                 Config                  |                                             Download                                             |
| :--------------------------------------------- | :--------: | :-------: | :-------------------------------------: | :----------------------------------------------------------------------------------------------: |
| `vit-base-p16_sam-pre_3rdparty_sa1b-1024px`\*  |   89.67    |  486.00   | [config](vit-base-p16_sam_headless.py)  | [model](https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-base-p16_sam-pre_3rdparty_sa1b-1024px_20230411-2320f9cc.pth) |
| `vit-large-p16_sam-pre_3rdparty_sa1b-1024px`\* |   308.00   |  1494.00  | [config](vit-large-p16_sam_headless.py) | [model](https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-large-p16_sam-pre_3rdparty_sa1b-1024px_20230411-595feafd.pth) |
| `vit-huge-p16_sam-pre_3rdparty_sa1b-1024px`\*  |   637.00   |  2982.00  | [config](vit-huge-p16_sam_headless.py)  | [model](https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-huge-p16_sam-pre_3rdparty_sa1b-1024px_20230411-3f13c653.pth) |

*Models with * are converted from the [official repo](https://github.com/facebookresearch/segment-anything/). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
