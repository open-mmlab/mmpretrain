_base_ = [
    '../_base_/models/mobilevit/mobilevit_s.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/imagenet_bs256.py',
]

# no normalize for original implements
data_preprocessor = dict(
    # RGB format normalization parameters
    mean=[0, 0, 0],
    std=[255, 255, 255],
    # use bgr directly
    to_rgb=False,
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=288, edge='short'),
    dict(type='CenterCrop', crop_size=256),
    dict(type='PackInputs'),
]

train_dataloader = dict(batch_size=128)

val_dataloader = dict(
    batch_size=128,
    dataset=dict(pipeline=test_pipeline),
)
test_dataloader = val_dataloader
