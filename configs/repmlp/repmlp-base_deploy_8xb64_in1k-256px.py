_base_ = ['./repmlp-base_8xb64_in1k-256px.py']

model = dict(backbone=dict(deploy=True))
