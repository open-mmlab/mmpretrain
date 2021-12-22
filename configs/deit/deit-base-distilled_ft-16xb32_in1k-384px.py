_base_ = './deit-base_ft-16xb32_in1k-384px.py'

# model settings
model = dict(
    backbone=dict(type='DistilledVisionTransformer'),
    head=dict(type='DeiTClsHead'),
    # Change to the path of the pretrained model
    # init_cfg=dict(type='Pretrained', checkpoint=''),
)
