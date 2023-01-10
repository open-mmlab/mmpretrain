# Long-tail Logit Adjustment

> [Long-tail Learning via Logit Adjustment](https://arxiv.org/abs/2007.07314)

<!-- [ALGORITHM] -->

## Abstract

Real-world classification problems typically exhibit an imbalanced or long-tailed label distribution, where in many labels are associated with only a few samples. This poses a challenge for generalisation on such labels, and also makes naïve learning biased towards dominant labels. In this paper, we present two simple modifications of standard softmax cross-entropy training to cope with these challenges. Our techniques revisit the classic idea of logit adjustment based on the label frequencies, either applied post-hoc to a trained model, or enforced in the loss during training. Such adjustment encourages a large relative margin between logits of rare versus dominant labels. These techniques unify and generalise several recent proposals in the literature, while possessing firmer statistical grounding and empirical performance.

<div align=center>
<img src="https://user-images.githubusercontent.com/18586273/210783601-bc99b0a8-9568-44bd-a550-3f80f3f60e3f.png" width="40%"/>
</div>

## Results

### Cifar10-LT

|  Imbalance-ratio   |  200  |  100  |  50   |  10   |                              Config (imb-ratio=10)                               |
| :----------------: | :---: | :---: | :---: | :---: | :------------------------------------------------------------------------------: |
|      Baseline      | 73.08 | 77.54 | 81.84 | 89.36 |     [config](./configs/logit_adjustment/resnet34_8xb16_cifar10-lt-rho10.py)      |
| Posthoc-adjustment | 76.59 | 80.67 | 83.98 | 90.01 | [config](./configs/logit_adjustment/resnet34-loss-adj_8xb16_cifar10-lt-rho10.py) |
|  Loss-adjustment   | 76.19 | 80.26 | 83.41 | 90.93 | [config](./configs/logit_adjustment/resnet34-loss-adj_8xb16_cifar10-lt-rho10.py) |

### Cifar100-LT

|  Imbalance-ratio   |  200  |  100  |  50   |  10   |                               Config (imb-ratio=10)                               |
| :----------------: | :---: | :---: | :---: | :---: | :-------------------------------------------------------------------------------: |
|      Baseline      | 41.22 | 45.98 | 51.87 | 64.75 |     [config](./configs/logit_adjustment/resnet34_8xb16_cifar100-lt-rho10.py)      |
| Posthoc-adjustment | 44.64 | 49.92 | 54.91 | 65.99 | [config](./configs/logit_adjustment/resnet34-loss-adj_8xb16_cifar100-lt-rho10.py) |
|  Loss-adjustment   | 44.14 | 48.53 | 54.35 | 65.75 | [config](./configs/logit_adjustment/resnet34-loss-adj_8xb16_cifar100-lt-rho10.py) |

All `imb_ratio` in the given configs are 10, if you want to modify it, you can modify in config file or add `--cfg-option train_dataloader.dataset.imb_ratio={$YOUR-IMB-RATIO}` in the end of your command lines.

```{note}
When using post-hoc logit adjustemnt, re-training from scratch is not necessary. You can just use the post-hoc configs to load checkpoints trained from non-logit-adjustemnt(baseline).
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
