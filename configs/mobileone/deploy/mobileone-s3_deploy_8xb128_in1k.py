_base_ = ['../mobileone-s3_8xb128_in1k.py']

model = dict(backbone=dict(deploy=True))
