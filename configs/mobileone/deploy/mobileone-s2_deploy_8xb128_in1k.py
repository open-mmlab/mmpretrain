_base_ = ['../mobileone-s2_8xb128_in1k.py']

model = dict(backbone=dict(deploy=True))
