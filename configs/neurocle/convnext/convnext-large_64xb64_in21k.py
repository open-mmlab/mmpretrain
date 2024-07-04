_base_ = [
    '../_base_/models/convnext/convnext-base.py',
    '../_base_/datasets/imagenet21k_bs128.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

# model setting
model = dict(head=dict(num_classes=21841))

# dataset setting
data_preprocessor = dict(num_classes=21841)
train_dataloader = dict(batch_size=64)

# schedule setting
optim_wrapper = dict(
    optimizer=dict(lr=4e-3),
    clip_grad=dict(max_norm=5.0),
)

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (32 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=4096)
