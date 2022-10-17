_base_ = [
    '../_base_/models/deit3/deit3-small-p16-224.py',
    '../_base_/datasets/imagenet_bs256_deit3.py',
    '../_base_/schedules/imagenet_bs2048_deit3.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(backbone=dict(drop_path_rate=0.05))

# dataset settings
train_pipeline = _base_['train_pipeline']
train_pipeline[1]['scale'] = 224

test_pipeline = _base_['test_pipeline']
test_pipeline[1]['scale'] = 224

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# schedule settings
optim_wrapper = dict(optimizer=dict(lr=4e-3))

# runtime settings
custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]
