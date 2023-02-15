_base_ = 'mmcls::resnet/resnetv1c50_8xb32_in1k.py'  # can be your own config

test_dataloader = dict(dataset=dict(data_root='data/imagenet-c'))
test_evaluator = dict(type='CorruptionError')
