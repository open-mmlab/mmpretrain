_base_ = '../repvggD2se_b64x4_imagenet_200e_coslr_' \
         'warmup_label_smoothing_mixup_autoaugment.py'

model = dict(backbone=dict(deploy=True))
