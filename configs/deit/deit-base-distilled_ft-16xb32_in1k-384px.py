_base_ = './deit-base_ft-16xb32_in1k-384px.py'

# model settings
model = dict(
    backbone=dict(type='DistilledVisionTransformer'),
    head=dict(type='DeiTClsHead'),
    # Change to the path of the pretrained model
    # init_cfg=dict(type='Pretrained', checkpoint=''),
)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (32 samples per GPU)
auto_scale_lr = dict(base_batch_size=512)
