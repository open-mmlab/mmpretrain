_base_ = ['./regnetx-400mf_8xb128_in1k.py']

# model settings
model = dict(
    backbone=dict(type='RegNet', arch='regnetx_6.4gf'),
    head=dict(in_channels=1624, ))

# for batch_size 512, use lr = 0.4
optimizer = dict(lr=0.4)

data = dict(samples_per_gpu=64, )
