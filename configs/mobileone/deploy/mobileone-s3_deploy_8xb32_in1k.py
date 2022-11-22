_base_ = ['../mobileone-s3_8xb32_in1k.py']

model = dict(backbone=dict(deploy=True))
