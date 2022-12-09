_base_ = [
    '../_base_/models/convmixer/convmixer-1536-20.py',
    '../_base_/datasets/imagenet_bs64_convmixer_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

# schedule setting
optim_wrapper = dict(
    optimizer=dict(lr=0.01),
    clip_grad=dict(max_norm=5.0),
)

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        begin=0,
        end=20,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=130,
        eta_min=1e-5,
        by_epoch=True,
        begin=20,
        end=150)
]

train_cfg = dict(by_epoch=True, max_epochs=150)

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (10 GPUs) x (64 samples per GPU)
auto_scale_lr = dict(base_batch_size=640)
