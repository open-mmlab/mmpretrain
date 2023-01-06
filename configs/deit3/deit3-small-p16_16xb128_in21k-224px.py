# Training on ImageNet-21k
_base_ = [
    '../_base_/models/deit3/deit3-small-p16-224.py',
    '../_base_/datasets/imagenet21k_bs64_deit3.py',
    '../_base_/schedules/imagenet_bs2048_deit3.py',
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

# dataset settings
train_dataloader = dict(batch_size=128)

# schedule settings
lr = 0.001
optim_wrapper = dict(optimizer=dict(lr=lr, weight_decay=0.02))
param_scheduler = _base_['param_scheduler']
param_scheduler[0]['start_factor'] = 1e-6 / lr

train_cfg = dict(max_epochs=240)
val_cfg = None
test_cfg = None
