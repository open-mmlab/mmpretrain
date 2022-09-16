_base_ = ['../mobileone-s4_8xb128_in1k.py']

model = dict(backbone=dict(deploy=True))
