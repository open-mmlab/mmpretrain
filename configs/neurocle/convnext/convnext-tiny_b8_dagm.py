_base_ = [
    '../_base_/models/convnext/convnext-tiny.py',
    '../_base_/datasets/dagm_bs8.py',
    '../_base_/schedules/dagm_swin.py',
    '../_base_/default_runtime.py',
]

# dataset setting
train_dataloader = dict(batch_size=32)

# schedule setting
optim_wrapper = dict(
    optimizer=dict(lr=4e-3),
    clip_grad=None,
)

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (32 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=32)

model = dict(
    head=dict(
        num_classes={{_base_.data_preprocessor.num_classes}}
    )
)
