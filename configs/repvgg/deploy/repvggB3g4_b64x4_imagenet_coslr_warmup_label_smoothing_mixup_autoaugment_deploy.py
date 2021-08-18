_base_ = '../repvggB3g4_b64x4_imagenet_coslr_' \
         'warmup_label_smoothing_mixup_autoaugment.py'

model = dict(backbone=dict(deploy=True))
