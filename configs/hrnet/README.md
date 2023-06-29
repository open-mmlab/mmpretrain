# HRNet

> [Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/abs/1908.07919v2)

<!-- [ALGORITHM] -->

## Abstract

High-resolution representations are essential for position-sensitive vision problems, such as human pose estimation, semantic segmentation, and object detection. Existing state-of-the-art frameworks first encode the input image as a low-resolution representation through a subnetwork that is formed by connecting high-to-low resolution convolutions *in series* (e.g., ResNet, VGGNet), and then recover the high-resolution representation from the encoded low-resolution representation. Instead, our proposed network, named as High-Resolution Network (HRNet), maintains high-resolution representations through the whole process. There are two key characteristics: (i) Connect the high-to-low resolution convolution streams *in parallel*; (ii) Repeatedly exchange the information across resolutions. The benefit is that the resulting representation is semantically richer and spatially more precise. We show the superiority of the proposed HRNet in a wide range of applications, including human pose estimation, semantic segmentation, and object detection, suggesting that the HRNet is a stronger backbone for computer vision problems.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/149920446-cbe05670-989d-4fe6-accc-df20ae2984eb.png" width="100%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('hrnet-w18_3rdparty_8xb32_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('hrnet-w18_3rdparty_8xb32_in1k', pretrained=True)
inputs = torch.rand(1, 3, 224, 224)
out = model(inputs)
print(type(out))
# To extract features.
feats = model.extract_feat(inputs)
print(type(feats))
```

**Test Command**

Prepare your dataset according to the [docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset).

Test:

```shell
python tools/test.py configs/hrnet/hrnet-w18_4xb32_in1k.py https://download.openmmlab.com/mmclassification/v0/hrnet/hrnet-w18_3rdparty_8xb32_in1k_20220120-0c10b180.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                                  |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |              Config               |                                     Download                                     |
| :------------------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :-------------------------------: | :------------------------------------------------------------------------------: |
| `hrnet-w18_3rdparty_8xb32_in1k`\*      | From scratch |   21.30    |   4.33    |   76.75   |   93.44   | [config](hrnet-w18_4xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/hrnet/hrnet-w18_3rdparty_8xb32_in1k_20220120-0c10b180.pth) |
| `hrnet-w30_3rdparty_8xb32_in1k`\*      | From scratch |   37.71    |   8.17    |   78.19   |   94.22   | [config](hrnet-w30_4xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/hrnet/hrnet-w30_3rdparty_8xb32_in1k_20220120-8aa3832f.pth) |
| `hrnet-w32_3rdparty_8xb32_in1k`\*      | From scratch |   41.23    |   8.99    |   78.44   |   94.19   | [config](hrnet-w32_4xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/hrnet/hrnet-w32_3rdparty_8xb32_in1k_20220120-c394f1ab.pth) |
| `hrnet-w40_3rdparty_8xb32_in1k`\*      | From scratch |   57.55    |   12.77   |   78.94   |   94.47   | [config](hrnet-w40_4xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/hrnet/hrnet-w40_3rdparty_8xb32_in1k_20220120-9a2dbfc5.pth) |
| `hrnet-w44_3rdparty_8xb32_in1k`\*      | From scratch |   67.06    |   14.96   |   78.88   |   94.37   | [config](hrnet-w44_4xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/hrnet/hrnet-w44_3rdparty_8xb32_in1k_20220120-35d07f73.pth) |
| `hrnet-w48_3rdparty_8xb32_in1k`\*      | From scratch |   77.47    |   17.36   |   79.32   |   94.52   | [config](hrnet-w48_4xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/hrnet/hrnet-w48_3rdparty_8xb32_in1k_20220120-e555ef50.pth) |
| `hrnet-w64_3rdparty_8xb32_in1k`\*      | From scratch |   128.06   |   29.00   |   79.46   |   94.65   | [config](hrnet-w64_4xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/hrnet/hrnet-w64_3rdparty_8xb32_in1k_20220120-19126642.pth) |
| `hrnet-w18_3rdparty_8xb32-ssld_in1k`\* | From scratch |   21.30    |   4.33    |   81.06   |   95.70   | [config](hrnet-w18_4xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/hrnet/hrnet-w18_3rdparty_8xb32-ssld_in1k_20220120-455f69ea.pth) |
| `hrnet-w48_3rdparty_8xb32-ssld_in1k`\* | From scratch |   77.47    |   17.36   |   83.63   |   96.79   | [config](hrnet-w48_4xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/hrnet/hrnet-w48_3rdparty_8xb32-ssld_in1k_20220120-d0459c38.pth) |

*Models with * are converted from the [official repo](https://github.com/HRNet/HRNet-Image-Classification). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI},
  year={2019}
}
```
