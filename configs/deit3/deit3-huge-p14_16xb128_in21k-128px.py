_base_ = [
    '../_base_/models/deit3/deit3-huge-p14-224.py',
    '../_base_/datasets/imagenet_bs256_deit3.py',
    '../_base_/schedules/imagenet_bs2048_deit3.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(backbone=dict(img_size=160))

# dataset settings
train_pipeline = _base_['train_pipeline']
train_pipeline[1]['scale'] = 160

test_pipeline = _base_['test_pipeline']
test_pipeline[1]['scale'] = 160
test_pipeline[2]['crop_size'] = 160

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
