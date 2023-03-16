_base_ = 'mmcls::resnet/resnetv1c50_8xb32_in1k.py'  # can be your own config

# You can replace imagenet-r with imagenet-a or imagenet-s
test_dataloader = dict(dataset=dict(data_root='data/imagenet-r'))
test_evaluator = dict(ann_file='data/imagenet-r/meta/val.txt')
