# Finetune ViT-B at resolution 384x384:
_base_ = [
    '../_base_/models/deit3/deit3-base-p16-384.py',
    '../_base_/datasets/imagenet_bs64_deit3_384.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    head=dict(
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            mode='original',
            _delete_=True,
        )))

# schedule settings
optim_wrapper = dict(optimizer=dict(lr=1e-5, weight_decay=0.1))

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=0., by_epoch=True, begin=5)
]

train_cfg = dict(max_epochs=20)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=512)
