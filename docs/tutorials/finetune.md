# Tutorial 1: Finetuning Models

Classification models pre-trained on the ImageNet dataset has been demonstrated to be effective for other datasets and other downstream tasks.
This tutorial provides instruction for users to use the models provided in the [Model Zoo](../model_zoo.md) for other datasets to obtain better performance.

There are two steps to finetune a model on a new dataset.

- Add support for the new dataset following [Tutorial 2: Adding New Dataset](new_dataset.md).
- Modify the configs as will be discussed in this tutorial.

Take the finetuning on CIFAR10 Dataset as an example, the users need to modify five parts in the config.

## Inherit base configs

To reuse the common parts among different configs, we support inheriting configs from multiple existing configs. To finetune a ResNet-50 model, the new config needs to inherit
`_base_/models/resnet50.py` to build the basic structure of the model. To use the CIFAR10 Dataset, the new config can also simply inherit `_base_/datasets/cifar10.py`. For runtime settings such as training schedules, the new config needs to inherit `_base_/default_runtime.py`.

```python
_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/cifar10.py', '../_base_/default_runtime.py'
]
```

Besides, users can also choose to write the whole contents rather than use inheritance, e.g. `configs/mnist/lenet5.py`.

## Modify head

Then the new config needs to modify the head according to the class numbers of the new datasets. By only changing `num_classes` in the head, the weights of the pre-trained models are mostly reused except the final prediction head.

```python
_base_ = ['./resnet50.py']
model = dict(
    pretrained=None,
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
```

## Modify dataset

The users may also need to prepare the dataset and write the configs about dataset. We currently support MNIST, CIFAR and ImageNet Dataset.
For fintuning on CIFAR10, its original input size is 32 and thus we should resize it to 224, to fit the input size of models pretrained on ImageNet.

```python
_base_ = ['./cifar10.py']
img_norm_cfg = dict(
     mean=[125.307, 122.961, 113.8575],
     std=[51.5865, 50.847, 51.255],
     to_rgb=True)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Resize', size=224)
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
 ]
 test_pipeline = [
    dict(type='Resize', size=224)
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
 ]
```

## Modify training schedule

The finetuning hyperparameters vary from the default schedule. It usually requires smaller learning rate and less training epochs.

```python
# optimizer
# lr is set for a batch size of 128
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    step=[15])
runner = dict(type='EpochBasedRunner', max_epochs=200)
log_config = dict(interval=100)
```

## Use pre-trained model

To use the pre-trained model, the new config add the link of pre-trained models in the `load_from`. The users might need to download the model weights before training to avoid the download time during training.

```python
load_from = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmclassification/models/tbd.pth'  # noqa
```
