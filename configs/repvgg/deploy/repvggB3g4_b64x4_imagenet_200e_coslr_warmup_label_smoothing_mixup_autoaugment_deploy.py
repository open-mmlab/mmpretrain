_base_ = '../repvggB3g4_b64x4_imagenet_200e_coslr_warmup_' \
         'label_smoothing_mixup_autoaugment.py'

model = dict(backbone=dict(deploy=True))
