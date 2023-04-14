_base_ = ['../mobileone-s1_8xb32_in1k.py']

model = dict(backbone=dict(deploy=True))
