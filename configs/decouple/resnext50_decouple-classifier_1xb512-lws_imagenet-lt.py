_base_ = [
    '../_base_/models/resnext50_32x4d.py',
    '../_base_/datasets/imagenet-lt_bs512_decouple.py',
    '../_base_/schedules/imagenet-lt_bs512_coslr_90e.py',
    '../_base_/default_runtime.py'
]

train_dataloader = dict(
    dataset=dict(type='ClassBalancedDataset',
                 oversample_thr=0.005,
                 dataset={{_base_.train_dataset}}),
)

train_epoch = 10

model = dict(
    backbone=dict(
        frozen_stages=4,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='.pth',
            prefix='backbone',
        )),
    head=dict(type='LWSHead',
              num_classes=1000,
              in_channels=2048,
              loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
              topk=(1, 5),
              ))
