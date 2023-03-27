# 下游任务

## 检测

我们使用 MMDetection 进行图像检测。首先确保您已经安装了 [MIM](https://github.com/open-mmlab/mim)，这也是 OpenMMLab 的一个项目。

```shell
pip install openmim
mim install 'mmdet>=3.0.0rc0'
```

此外，请参考 MMDetection 的[安装](https://mmdetection.readthedocs.io/en/dev-3.x/get_started.html)和[数据准备](https://mmdetection.readthedocs.io/en/dev-3.x/user_guides/dataset_prepare.html)

### 训练

安装完后，您可以使用如下的简单命令运行 MMDetection。

```shell
# distributed version
bash tools/benchmarks/mmdetection/mim_dist_train_c4.sh ${CONFIG} ${PRETRAIN} ${GPUS}
bash tools/benchmarks/mmdetection/mim_dist_train_fpn.sh ${CONFIG} ${PRETRAIN} ${GPUS}

# slurm version
bash tools/benchmarks/mmdetection/mim_slurm_train_c4.sh ${PARTITION} ${CONFIG} ${PRETRAIN}
bash tools/benchmarks/mmdetection/mim_slurm_train_fpn.sh ${PARTITION} ${CONFIG} ${PRETRAIN}
```

- `${CONFIG}`：直接用 MMDetection 中的配置文件路径即可。对于一些算法，我们有一些修改过的配置文件，
  可以在相应算法文件夹下的 `benchmarks` 文件夹中找到。另外，您也可以从头开始编写配置文件。
- `${PRETRAIN}`：预训练模型文件
- `${GPUS}`：使用多少 GPU 进行训练，对于检测任务，我们默认使用 8 个 GPU。

例子：

```shell
bash ./tools/benchmarks/mmdetection/mim_dist_train_c4.sh \
  configs/byol/benchmarks/mask-rcnn_r50-c4_ms-1x_coco.py \
  https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth 8
```

### 测试

在训练之后，您可以运行如下命令测试您的模型。

```shell
# distributed version
bash tools/benchmarks/mmdetection/mim_dist_test.sh ${CONFIG} ${CHECKPOINT} ${GPUS}

# slurm version
bash tools/benchmarks/mmdetection/mim_slurm_test.sh ${PARTITION} ${CONFIG} ${CHECKPOINT}
```

备注：

- `${CHECKPOINT}`：您想测试的训练好的检测模型。

例子：

```shell
bash ./tools/benchmarks/mmdetection/mim_dist_test.sh \
configs/benchmarks/mmdetection/coco/mask-rcnn_r50_fpn_ms-1x_coco.py \
https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth 8
```

## 分割

我们使用 MMSegmentation 进行图像分割。首先确保您已经安装了 [MIM](https://github.com/open-mmlab/mim)，这也是 OpenMMLab 的一个项目。

```shell
pip install openmim
mim install 'mmsegmentation>=1.0.0rc0'
```

此外，请参考 MMSegmentation 的[安装](https://mmsegmentation.readthedocs.io/en/dev-1.x/get_started.html)和[数据准备](https://mmsegmentation.readthedocs.io/en/dev-1.x/user_guides/2_dataset_prepare.html)。

### 训练

在安装完后，可以使用如下简单命令运行 MMSegmentation。

```shell
# distributed version
bash tools/benchmarks/mmsegmentation/mim_dist_train.sh ${CONFIG} ${PRETRAIN} ${GPUS}

# slurm version
bash tools/benchmarks/mmsegmentation/mim_slurm_train.sh ${PARTITION} ${CONFIG} ${PRETRAIN}
```

备注：

- `${CONFIG}`：直接用 MMSegmentation 中的配置文件路径即可。对于一些算法，我们有一些修改过的配置文件，
  可以在相应算法文件夹下的 `benchmarks` 文件夹中找到。另外，您也可以从头开始编写配置文件。
- `${PRETRAIN}`：预训练模型文件
- `${GPUS}`：使用多少 GPU 进行训练，对于检测任务，我们默认使用 8 个 GPU。

例子：

```shell
bash ./tools/benchmarks/mmsegmentation/mim_dist_train.sh \
configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_4xb4-20k_voc12aug-512x512.py \
https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth 4
```

### 测试

在训练之后，您可以运行如下命令测试您的模型。

```shell
# distributed version
bash tools/benchmarks/mmsegmentation/mim_dist_test.sh ${CONFIG} ${CHECKPOINT} ${GPUS}

# slurm version
bash tools/benchmarks/mmsegmentation/mim_slurm_test.sh ${PARTITION} ${CONFIG} ${CHECKPOINT}
```

备注：

- `${CHECKPOINT}`：您想测试的训练好的分割模型。

例子：

```shell
bash ./tools/benchmarks/mmsegmentation/mim_dist_test.sh \
configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_4xb4-20k_voc12aug-512x512.py \
https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth 4
```
