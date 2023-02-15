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

|      Imbalance      |  200  |  100  |  50   |  20   |  10   |  1   |
| :-----------------: | :---: | :---: | :---: | :---: | :---: | :--: |
|      Baseline       | 35.67 | 29.71 | 22.91 | 16.04 | 13.26 | 6.83 |
| post-hoc-adjustment | 27.23 | 20.17 | 16.80 | 12.76 | 10.71 | 6.29 |
|   loss-adjustment   | 27.23 | 20.17 | 16.80 | 12.76 | 10.71 | 6.29 |

### Cifar100-LT

|      Imbalance      |  200  |  100  |  50   |  20   |  10   |  1   |
| :-----------------: | :---: | :---: | :---: | :---: | :---: | :--: |
|      Baseline       | 35.67 | 29.71 | 22.91 | 16.04 | 13.26 | 6.83 |
| post-hoc-adjustment | 27.23 | 20.17 | 16.80 | 12.76 | 10.71 | 6.29 |
|   loss-adjustment   | 27.23 | 20.17 | 16.80 | 12.76 | 10.71 | 6.29 |

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
