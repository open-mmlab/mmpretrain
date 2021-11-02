_base_ = ['../resnet/resnet50_8xb32_in1k.py']

# fp16 settings
fp16 = dict(loss_scale='dynamic')
