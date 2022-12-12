# Solution of FGIA ACCV 2022(1st Place)

This is fine-tuning part of the 1st Place Solution for Webly-supervised Fine-grained Recognition, refer to the ACCV workshop competition in https://www.cvmart.net/race/10412/base.

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

For detailed self-supervised pretrain code, please refer to [MMSelfSup](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/projects/fgia_accv2022_1st).
For detailed finetuning and inference code, please refer to [this repo](https://github.com/Ezra-Yu/ACCV2022_FGIA_1st).

## Description

### Overview of Our Solution

![image](https://user-images.githubusercontent.com/18586273/205498371-31dbc1f4-5814-44bc-904a-f0d32515c7dd.png)

### Our Model

- ViT(MAE-pre-train)   # Pretrained from [**MMSelfSup**](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/projects/fgia_accv2022_1st).
- Swin-v2(SimMIM-pre-train)   # From [MMCls-swin_transformer_v2](https://github.com/open-mmlab/mmclassification/tree/dev-1.x/configs/swin_transformer_v2).

\*\*The architectures we use \*\*

- ViT + CE-loss + post-LongTail-Adjusment
- ViT + SubCenterArcFaceWithAdvMargin(CE)
- Swin-B + SubCenterArcFaceWithAdvMargin(SoftMax-EQL)
- Swin-L + SubCenterArcFaceWithAdvMargin(SoftMAx-EQL)

## bag of tricks paper and code

- [MAE](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/mae) |  [Config](https://github.com/Ezra-Yu/ACCV_workshop/tree/master/configs/vit)
- [Swinv2](https://github.com/open-mmlab/mmclassification/tree/dev-1.x/configs/swin_transformer_v2) | [Config](https://github.com/Ezra-Yu/ACCV_workshop/tree/master/configs/swin)
- [ArcFace](https://arxiv.org/abs/1801.07698)   |   [Code](https://github.com/Ezra-Yu/ACCV_workshop/blob/master/src/models/arcface_head.py)
- [SubCenterArcFaceWithAdvMargin](https://paperswithcode.com/paper/sub-center-arcface-boosting-face-recognition)   |   [Code](https://github.com/Ezra-Yu/ACCV_workshop/blob/master/src/models/arcface_head.py)
- [Post-LT-adjusment](https://paperswithcode.com/paper/long-tail-learning-via-logit-adjustment)   |   [Code](https://github.com/Ezra-Yu/ACCV_workshop/blob/master/src/models/linear_head_lt.py)
- [SoftMaxEQL](https://paperswithcode.com/paper/the-equalization-losses-gradient-driven)   |   [Code](https://github.com/Ezra-Yu/ACCV_workshop/blob/master/src/models/eql.py)
- FlipTTA [Code](https://github.com/Ezra-Yu/ACCV_workshop/blob/master/src/models/tta_classifier.py)
- clean dataset
- self-emsemble: [Uniform-model-soup](https://arxiv.org/abs/2203.05482) | [code](https://github.com/Ezra-Yu/ACCV_workshop/blob/master/tools/model_soup.py)
- [pseudo](https://lilianweng.github.io/posts/2021-12-05-semi-supervised/)  | [Code](https://github.com/Ezra-Yu/ACCV_workshop/blob/master/tools/creat_pseudo.py)
- bagging-emsemble [Code](https://github.com/Ezra-Yu/ACCV_workshop/blob/master/tools/emsemble.py),
- post-process: [re-distribute-label](https://github.com/Ezra-Yu/ACCV_workshop/blob/master/tools/re-distribute-label.py);

![Overview](https://user-images.githubusercontent.com/18586273/205498258-e5720d83-7006-4aea-86b5-aab1a8998c6c.png)

![image](https://user-images.githubusercontent.com/18586273/205498027-def99b0d-a99a-470b-b292-8d5fc83111fc.png)

#### Used but no improvements

1. Using retrieval paradigm to solve this classification task;
2. Using EfficientNetv2 backbone.

#### Not used but worth to do

1. Try [DiVE](https://arxiv.org/abs/2103.15042) algorithm to improve performance in long tail dataset;
2. Use SimMIM to pre-train Swin-v2 on the competition dataset;
3. refine the re-distribute-label tool.
