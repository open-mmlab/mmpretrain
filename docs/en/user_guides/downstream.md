# Downstream tasks

## Detection

For detection tasks, please use MMDetection. First, make sure you have installed [MIM](https://github.com/open-mmlab/mim), which is also a project of OpenMMLab.

```shell
pip install openmim
mim install 'mmdet>=3.0.0rc0'
```

Besides, please refer to MMDet for [installation](https://mmdetection.readthedocs.io/en/dev-3.x/get_started.html) and [data preparation](https://mmdetection.readthedocs.io/en/dev-3.x/user_guides/dataset_prepare.html)

### Train

After installation, you can run MMDetection with simple command.

```shell
# distributed version
bash tools/benchmarks/mmdetection/mim_dist_train_c4.sh ${CONFIG} ${PRETRAIN} ${GPUS}
bash tools/benchmarks/mmdetection/mim_dist_train_fpn.sh ${CONFIG} ${PRETRAIN} ${GPUS}

# slurm version
bash tools/benchmarks/mmdetection/mim_slurm_train_c4.sh ${PARTITION} ${CONFIG} ${PRETRAIN}
bash tools/benchmarks/mmdetection/mim_slurm_train_fpn.sh ${PARTITION} ${CONFIG} ${PRETRAIN}
```

- `${CONFIG}`: Use config file path in MMDetection directly. And for some algorithms, we also have some
  modified config files which can be found in the `benchmarks` folder under the correspondding algorithm
  folder. You can also writing your config file from scratch.
- `${PRETRAIN}`: the pre-trained model file.
- `${GPUS}`: The number of GPUs that you want to use to train. We adopt 8 GPUs for detection tasks by default.

Example:

```shell
bash ./tools/benchmarks/mmdetection/mim_dist_train_c4.sh \
  configs/byol/benchmarks/mask-rcnn_r50-c4_ms-1x_coco.py \
  https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth 8
```

### Test

After training, you can also run the command below to test your model.

```shell
# distributed version
bash tools/benchmarks/mmdetection/mim_dist_test.sh ${CONFIG} ${CHECKPOINT} ${GPUS}

# slurm version
bash tools/benchmarks/mmdetection/mim_slurm_test.sh ${PARTITION} ${CONFIG} ${CHECKPOINT}
```

- `${CONFIG}`: Use config file name in MMDetection directly. And for some algorithms, we also have some
  modified config files which can be found in the `benchmarks` folder under the correspondding algorithm
  folder. You can also writing your config file from scratch.
- `${CHECKPOINT}`: The fine-tuned detection model that you want to test.
- `${GPUS}`: The number of GPUs that you want to use to test. We adopt 8 GPUs for detection tasks by default.

Example:

```shell
bash ./tools/benchmarks/mmdetection/mim_dist_test.sh \
configs/byol/benchmarks/mask-rcnn_r50_fpn_ms-1x_coco.py \
https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth 8
```

## Segmentation

For semantic segmentation task, we use MMSegmentation. First, make sure you have installed [MIM](https://github.com/open-mmlab/mim), which is also a project of OpenMMLab.

```shell
pip install openmim
mim install 'mmsegmentation>=1.0.0rc0'
```

Besides, please refer to MMSegmentation for [installation](https://mmsegmentation.readthedocs.io/en/dev-1.x/get_started.html) and [data preparation](https://mmsegmentation.readthedocs.io/en/dev-1.x/user_guides/2_dataset_prepare.html).

### Train

After installation, you can run MMSegmentation with simple command.

```shell
# distributed version
bash tools/benchmarks/mmsegmentation/mim_dist_train.sh ${CONFIG} ${PRETRAIN} ${GPUS}

# slurm version
bash tools/benchmarks/mmsegmentation/mim_slurm_train.sh ${PARTITION} ${CONFIG} ${PRETRAIN}
```

- `${CONFIG}`: Use config file path in MMSegmentation directly. And for some algorithms, we also have some
  modified config files which can be found in the `benchmarks` folder under the correspondding algorithm
  folder. You can also writing your config file from scratch.
- `${PRETRAIN}`: the pre-trained model file.
- `${GPUS}`: The number of GPUs that you want to use to train. We adopt 4 GPUs for segmentation tasks by default.

Example:

```shell
bash ./tools/benchmarks/mmsegmentation/mim_dist_train.sh \
configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_4xb4-20k_voc12aug-512x512.py \
https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth 4
```

### Test

After training, you can also run the command below to test your model.

```shell
# distributed version
bash tools/benchmarks/mmsegmentation/mim_dist_test.sh ${CONFIG} ${CHECKPOINT} ${GPUS}

# slurm version
bash tools/benchmarks/mmsegmentation/mim_slurm_test.sh ${PARTITION} ${CONFIG} ${CHECKPOINT}
```

- `${CONFIG}`: Use config file name in MMSegmentation directly. And for some algorithms, we also have some
  modified config files which can be found in the `benchmarks` folder under the correspondding algorithm
  folder. You can also writing your config file from scratch.
- `${CHECKPOINT}`: The fine-tuned segmentation model that you want to test.
- `${GPUS}`: The number of GPUs that you want to use to test. We adopt 4 GPUs for segmentation tasks by default.

Example:

```shell
bash ./tools/benchmarks/mmsegmentation/mim_dist_test.sh  fcn_r50-d8_4xb4-20k_voc12aug-512x512.py \
https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth 4
```
