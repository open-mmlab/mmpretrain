# Logit-Adjustment (Long-tail)

> [Long-tail Learning via Logit Adjustment](https://arxiv.org/abs/2007.07314)

<!-- [ALGORITHM] -->

## Abstract

Real-world classification problems typically exhibit an imbalanced or long-tailed label distribution, where in many labels are associated with only a few samples. This poses a challenge for generalisation on such labels, and also makes naïve learning biased towards dominant labels. In this paper, we present two simple modifications of standard softmax cross-entropy training to cope with these challenges. Our techniques revisit the classic idea of logit adjustment based on the label frequencies, either applied post-hoc to a trained model, or enforced in the loss during training. Such adjustment encourages a large relative margin between logits of rare versus dominant labels. These techniques unify and generalise several recent proposals in the literature, while possessing firmer statistical grounding and empirical performance.

<div align=center>
<img src="https://user-images.githubusercontent.com/18586273/210783601-bc99b0a8-9568-44bd-a550-3f80f3f60e3f.png" width="40%"/>
</div>

## Results

### Cifar10-LT

|  Imbalance-ratio   |  200  |  100  |  50   |  10   |                                                           Config (imb-ratio=10)                                                           |
| :----------------: | :---: | :---: | :---: | :---: | :---------------------------------------------------------------------------------------------------------------------------------------: |
|      Baseline      | 72.56 | 77.49 | 81.57 | 89.36 |     [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/logit_adjustment/resnet34_8xb16_cifar10-lt-rho10.py)      |
| Posthoc-adjustment | 76.49 | 80.25 | 83.90 | 89.89 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/logit_adjustment/resnet34-loss-adj_8xb16_cifar10-lt-rho10.py) |
|  Loss-adjustment   | 76.20 | 80.24 | 84.24 | 90.55 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/logit_adjustment/resnet34-loss-adj_8xb16_cifar10-lt-rho10.py) |

### Cifar100-LT

|  Imbalance-ratio   |  200  |  100  |  50   |  10   |                                                           Config (imb-ratio=10)                                                            |
| :----------------: | :---: | :---: | :---: | :---: | :----------------------------------------------------------------------------------------------------------------------------------------: |
|      Baseline      | 40.48 | 44.77 | 51.11 | 64.14 |     [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/logit_adjustment/resnet34_8xb16_cifar100-lt-rho10.py)      |
| Posthoc-adjustment | 43.82 | 48.43 | 54.19 | 65.28 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/logit_adjustment/resnet34-loss-adj_8xb16_cifar100-lt-rho10.py) |
|  Loss-adjustment   | 44.17 | 48.92 | 53.69 | 65.85 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/logit_adjustment/resnet34-loss-adj_8xb16_cifar100-lt-rho10.py) |

All `imb_ratio` in the given configs are 10, if you want to modify it, you can modify in config file or add `--cfg-option train_dataloader.dataset.imb_ratio={$imb-ratio}` in the end of your command lines.

```{note}
When using post-hoc logit adjustemnt, retraining from scratch is not necessary. You can just use the post-hoc configs to load checkpoints trained from non-logit-adjustemnt.
```

## Citation

```
@article{krishna2020long,
  title={Long-tail learning via logit adjustment},
  author={Krishna Menon, Aditya and Jayasumana, Sadeep and Singh Rawat, Ankit and Jain, Himanshu and Veit, Andreas and Kumar, Sanjiv},
  journal={arXiv e-prints},
  pages={arXiv--2007},
  year={2020}
}
```
