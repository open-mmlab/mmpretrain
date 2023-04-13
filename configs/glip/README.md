# GLIP

> [Grounded Language-Image Pre-training](https://arxiv.org/abs/2112.03857)

<!-- [ALGORITHM] -->

## Abstract

This paper presents a grounded language-image pre-training (GLIP) model for learning object-level, language-aware, and semantic-rich visual representations. GLIP unifies object detection and phrase grounding for pre-training. The unification brings two benefits: 1) it allows GLIP to learn from both detection and grounding data to improve both tasks and bootstrap a good grounding model; 2) GLIP can leverage massive image-text pairs by generating grounding boxes in a self-training fashion, making the learned representation semantic-rich. In our experiments, we pre-train GLIP on 27M grounding data, including 3M human-annotated and 24M web-crawled image-text pairs. The learned representations demonstrate strong zero-shot and few-shot transferability to various object-level recognition tasks. 1) When directly evaluated on COCO and LVIS (without seeing any images in COCO during pre-training), GLIP achieves 49.8 AP and 26.9 AP, respectively, surpassing many supervised baselines. 2) After fine-tuned on COCO, GLIP achieves 60.8 AP on val and 61.5 AP on test-dev, surpassing prior SoTA. 3) When transferred to 13 downstream object detection tasks, a 1-shot GLIP rivals with a fully-supervised Dynamic Head.

<div align="center">
<img src="https://github.com/microsoft/GLIP/blob/main/docs/lead.png" width="70%"/>
</div>

## Results and models

### Pre-trained models

The pre-trained models are used to fine-tune, and therefore don't have evaluation results.

| Model                                   |          Pretrain          | resolution |  Download   |
| :-------------------------------------- | :------------------------: | :--------: | :---------: |
| GLIP-T (`glip-t_3rdparty_o365`)\*       |            O365            |  224x224   | [model](<>) |
| GLIP-T (`glip-t_3rdparty_o365_goldg`)\* |         O365,GoldG         |  224x224   | [model](<>) |
| GLIP-T (`glip-t_3rdparty_merged`)\*     |    O365,GoldG,CC3M,SBU     |  224x224   | [model](<>) |
| GLIP-L (`glip-l_3rdparty_merged`)\*     | FourODs,GoldG,CC3M+12M,SBU |  384x384   | [model](<>) |

*Models with * are converted from the [official repo](https://github.com/microsoft/GLIP).*

## Citation

```bibtex
@inproceedings{li2021grounded,
      title={Grounded Language-Image Pre-training},
      author={Liunian Harold Li* and Pengchuan Zhang* and Haotian Zhang* and Jianwei Yang and Chunyuan Li and Yiwu Zhong and Lijuan Wang and Lu Yuan and Lei Zhang and Jenq-Neng Hwang and Kai-Wei Chang and Jianfeng Gao},
      year={2022},
      booktitle={CVPR},
}
```
