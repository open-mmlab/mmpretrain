_base_ = ['./regnetx-400mf_8xb128_in1k.py']

# model settings
model = dict(
    backbone=dict(type='RegNet', arch='regnetx_4.0gf'),
    head=dict(in_channels=1360, ))

# dataset settings
train_dataloader = dict(batch_size=64)

# schedule settings
# for batch_size 512, use lr = 0.4
optim_wrapper = dict(optimizer=dict(lr=0.4))
