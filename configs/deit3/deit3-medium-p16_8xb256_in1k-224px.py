_base_ = [
    '../_base_/models/deit3/deit3-medium-p16-224.py',
    '../_base_/datasets/imagenet_bs256_deit3.py',
    '../_base_/schedules/imagenet_bs2048_deit3.py',
    '../_base_/default_runtime.py'
]

# dataset settings
train_pipeline = _base_['train_pipeline']
train_pipeline[1]['scale'] = 224

test_pipeline = _base_['test_pipeline']
test_pipeline[1]['scale'] = 224
test_pipeline[2]['crop_size'] = 224

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# schedule settings
lr = 4e-3
optim_wrapper = dict(optimizer=dict(lr=lr))
param_scheduler = _base_['param_scheduler']
param_scheduler[0]['start_factor'] = 1e-6 / lr
