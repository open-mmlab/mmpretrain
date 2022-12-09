_base_ = ['../mobileone-s4_8xb32_in1k.py']

model = dict(backbone=dict(deploy=True))
