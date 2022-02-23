_base_ = ['./repmlp-b224_8xb64_in1k.py']

model = dict(backbone=dict(deploy=True))
