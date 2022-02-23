_base_ = ['./repmlp-b256_8xb64_in1k.py']

model = dict(backbone=dict(deploy=True))
