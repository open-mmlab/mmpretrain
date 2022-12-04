# 1st Place Solution for Webly-supervised Fine-grained Recognition

For the competition in https://www.cvmart.net/race/10412/base

Mainly done by [Ezra-Yu](https://github.com/Ezra-Yu), [Yuan Liu](https://github.com/YuanLiuuuuuu) and [Songyang Zhang](https://github.com/tonysy), base on [**MMClassifiion**](https://github.com/open-mmlab/mmclassification) 与 [**MMSelfSup**](https://github.com/open-mmlab/mmselfsup). Wlcome to use them and star, flork.

## Result

<details>

<summary>Show the result</summary>

<br>

**LB A**

![LB-A](https://user-images.githubusercontent.com/18586273/205498131-5728e470-b4f6-43b7-82a5-5f8e3bd5168e.png)

**LB B**

![LB-B](https://user-images.githubusercontent.com/18586273/205498171-5a3a3055-370a-4a8b-9779-b686254ebc94.png)

</br>

</details>

## Reproduce / 复现

**复现精度请点击[这里](./Reproduce.md)(Only Chinese Doc)**

## Description

### Overview Flow

![image](https://user-images.githubusercontent.com/18586273/205498371-31dbc1f4-5814-44bc-904a-f0d32515c7dd.png)

### Model Select

- ViT(MAE-pt)   # Pretrained from [**MMSelfSup**](https://github.com/open-mmlab/mmselfsup).
- Swin(21kpt)   # From [MMCls-swin_transformer_v2](https://github.com/open-mmlab/mmclassification/tree/dev-1.x/configs/swin_transformer_v2).

**Main Arch**

- ViT + CE-loss + post-LongTail-Adjusment
- ViT + SubCenterArcFaceWithAdvMargin(CE)
- Swin-B + SubCenterArcFaceWithAdvMargin(SoftMax-EQL)
- Swin-L + SubCenterArcFaceWithAdvMargin(SoftMAx-EQL)

## bag of tricks paper and code

- [MAE](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/mae) |  [Config](./configs/vit/)
- [Swinv2](https://github.com/open-mmlab/mmclassification/tree/dev-1.x/configs/swin_transformer_v2) | [Config](./configs/swin/)
- [ArcFace](https://arxiv.org/abs/1801.07698)   |   [Code](./src/models/arcface_head.py)
- [SubCenterArcFaceWithAdvMargin](https://paperswithcode.com/paper/sub-center-arcface-boosting-face-recognition)   |   [Code](./src/models/arcface_head.py)
- [Post-LT-adjusment](https://paperswithcode.com/paper/long-tail-learning-via-logit-adjustment)   |   [Code](./src/models/linear_head_lt.py)
- [SoftMaxEQL](https://paperswithcode.com/paper/the-equalization-losses-gradient-driven)   |   [Code](./src/models/eql.py)
- FlipTTA [Code](./src/models/tta_classifier.py)
- clean dataset
- self-emsemble: [Uniform-model-soup](https://arxiv.org/abs/2203.05482) | [code](./tools/model_soup.py)
- [pseudo](https://lilianweng.github.io/posts/2021-12-05-semi-supervised/)  | [Code](./tools/creat_pseudo.py)
- bagging-emsemble [Code](./tools/emsemble.py),
- post-process: [re-distribute-label](./tools/re-distribute-label.py);

![Overview](https://user-images.githubusercontent.com/18586273/205498258-e5720d83-7006-4aea-86b5-aab1a8998c6c.png)

![image](https://user-images.githubusercontent.com/18586273/205498027-def99b0d-a99a-470b-b292-8d5fc83111fc.png)

#### Used but no improvement

1. Retrieval Paradigm
2. EfficientNetv2

#### Not used but worth to do

1. DiVE for Long tail Problem;
2. Use SimMIM to train a swinv2 SelfSup pretrined model;
3. improve re-distribute-label tool
