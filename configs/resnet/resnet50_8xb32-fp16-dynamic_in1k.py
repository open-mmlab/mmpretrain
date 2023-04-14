_base_ = ['./resnet50_8xb32_in1k.py']

# schedule settings
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
