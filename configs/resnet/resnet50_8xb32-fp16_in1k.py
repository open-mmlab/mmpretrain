_base_ = ['./resnet50_8xb32_in1k.py']

# schedule settings
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale=512.)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (32 samples per GPU)
auto_scale_lr = dict(base_batch_size=256)
