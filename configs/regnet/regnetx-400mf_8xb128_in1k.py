_base_ = [
    '../_base_/models/regnet/regnetx_400mf.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs1024_coslr.py',
    '../_base_/default_runtime.py'
]

# dataset settings
data_preprocessor = dict(
    # BGR format normalization parameters
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    to_rgb=False,  # The checkpoints from PyCls requires BGR format inputs.
)

# lighting params, in order of BGR, from repo. pycls
EIGVAL = [0.2175, 0.0188, 0.0045]
EIGVEC = [
    [-0.5836, -0.6948, 0.4203],
    [-0.5808, -0.0045, -0.814],
    [-0.5675, 0.7192, 0.4009],
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='Lighting',
        eigval=EIGVAL,
        eigvec=EIGVEC,
        alphastd=25.5,  # because the value range of images is [0,255]
        to_rgb=False),
    dict(type='PackInputs'),
]

train_dataloader = dict(batch_size=128, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=128)
test_dataloader = dict(batch_size=128)

# schedule settings

# sgd with nesterov, base ls is 0.8 for batch_size 1024,
optim_wrapper = dict(optimizer=dict(lr=0.8, nesterov=True))

# runtime settings

# Precise BN hook will update the bn stats, so this hook should be executed
# before CheckpointHook(priority of 'VERY_LOW') and
# EMAHook(priority of 'NORMAL') So set the priority of PreciseBNHook to
# 'ABOVENORMAL' here.
custom_hooks = [
    dict(
        type='PreciseBNHook',
        num_samples=8192,
        interval=1,
        priority='ABOVE_NORMAL')
]
