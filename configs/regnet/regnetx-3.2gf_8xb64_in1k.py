_base_ = ['./regnetx-400mf_8xb128_in1k.py']

# model settings
model = dict(
    backbone=dict(type='RegNet', arch='regnetx_3.2gf'),
    head=dict(in_channels=1008, ))

# dataset settings
train_dataloader = dict(batch_size=64)

# schedule settings
optimizer = dict(lr=0.4)  # for batch_size 512, use lr = 0.4
