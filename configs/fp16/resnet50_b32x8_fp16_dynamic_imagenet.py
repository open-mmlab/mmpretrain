_base_ = ['../resnet/resnet50_b32x8_imagenet.py']

# fp16 settings
fp16 = dict(loss_scale='dynamic')
