# ChineseCLIP

> [Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese](https://arxiv.org/abs/2211.01335)

<!-- [ALGORITHM] -->

## Abstract

The tremendous success of CLIP (Radford et al., 2021) has promoted the research and application of contrastive learning for vision-language pretraining. In this work, we construct a large-scale dataset of image-text pairs in Chinese, where most data are retrieved from publicly available datasets, and we pretrain Chinese CLIP models on the new dataset. We develop 5 Chinese CLIP models of multiple sizes, spanning from 77 to 958 million parameters. Furthermore, we propose a two-stage pretraining method, where the model is first trained with the image encoder frozen and then trained with all parameters being optimized, to achieve enhanced model performance. Our comprehensive experiments demonstrate that Chinese CLIP can achieve the state-of-the-art performance on MUGE, Flickr30K-CN, and COCO-CN in the setups of zero-shot learning and finetuning, and it is able to achieve competitive performance in zero-shot image classification based on the evaluation on the ELEVATER benchmark (Li et al., 2022). We have released our codes, models, and demos in https://github.com/OFA-Sys/Chinese-CLIP

<div align=center>
<img src="https://github.com/open-mmlab/mmpretrain/assets/36138628/4d05e51f-d834-4ef5-bbf0-0e2f80fea461" width="80%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Use the model for zero-shot classification**

```python
from mmpretrain import ImageClassificationInferencer

inferencer = ImageClassificationInferencer(
    'cn-clip_resnet50_zeroshot-cls_cifar100',
    pretrained=True,
    classes=['鸟', '狗', '猫', '蛇'],
    text_prototype=['鸟', '狗', '猫', '蛇'],
)

prediction = inferencer('./demo/bird.JPEG')[0]
print('Results:', prediction['pred_class'])
```

**Train/Test Command**

Prepare your dataset according to the [docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset).

Test:

```shell
python tools/test.py configs/chinese_clip/cn-clip_resnet50_zeroshot-cls_cifar100.py https://download.openmmlab.com/mmpretrain/v1.0/chinese_clip/cn-clip_resnet50_3rdparty_20230519-6a2b3eb2.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on CIFAR100

| Model                                           | Params (M) | Top-1 (%) |                          Config                          |                                    Download                                    |
| :---------------------------------------------- | :--------: | :-------: | :------------------------------------------------------: | :----------------------------------------------------------------------------: |
| `cn-clip_resnet50_zeroshot-cls_cifar100`\*      |   77.00    |   40.70   |   [config](cn-clip_resnet50_zeroshot-cls_cifar100.py)    | [model](https://download.openmmlab.com/mmpretrain/v1.0/chinese_clip/cn-clip_resnet50_3rdparty_20230519-6a2b3eb2.pth) |
| `cn-clip_vit-base-p16_zeroshot-cls_cifar100`\*  |   188.00   |   64.50   | [config](cn-clip_vit-base-p16_zeroshot-cls_cifar100.py)  | [model](https://download.openmmlab.com/mmpretrain/v1.0/chinese_clip/cn-clip_vit-base-p16_3rdparty_20230519-37fbc59e.pth) |
| `cn-clip_vit-large-p14_zeroshot-cls_cifar100`\* |   406.00   |   74.80   | [config](cn-clip_vit-large-p14_zeroshot-cls_cifar100.py) | [model](https://download.openmmlab.com/mmpretrain/v1.0/chinese_clip/cn-clip_vit-large-p14_3rdparty_20230519-3f844503.pth) |
| `cn-clip_vit-huge-p14_zeroshot-cls_cifar100`\*  |   958.00   |   79.10   | [config](cn-clip_vit-huge-p14_zeroshot-cls_cifar100.py)  | [model](https://download.openmmlab.com/mmpretrain/v1.0/chinese_clip/cn-clip_vit-huge-p14_3rdparty_20230519-e4f49b00.pth) |

*Models with * are converted from the [official repo](https://github.com/OFA-Sys/Chinese-CLIP). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@article{chinese-clip,
  title={Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese},
  author={Yang, An and Pan, Junshu and Lin, Junyang and Men, Rui and Zhang, Yichang and Zhou, Jingren and Zhou, Chang},
  journal={arXiv preprint arXiv:2211.01335},
  year={2022}
}
```
