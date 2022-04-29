_base_ = ['./repmlp-base_8xb64_in1k.py']

model = dict(backbone=dict(deploy=True))
