_base_ = './deit-small_pt-4xb256_in1k.py'

# model settings
model = dict(
    backbone=dict(type='DistilledVisionTransformer', arch='deit-small'),
    head=dict(type='DeiTClsHead', in_channels=384),
)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (4 GPUs) x (256 samples per GPU)
auto_scale_lr = dict(base_batch_size=1024)
