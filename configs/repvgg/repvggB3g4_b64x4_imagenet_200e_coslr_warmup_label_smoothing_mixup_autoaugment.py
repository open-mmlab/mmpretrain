_base_ = './repvggB3_b64x4_imagenet_200e_coslr_' \
         'warmup_label_smoothing_mixup_autoaugment.py'

model = dict(backbone=dict(arch='B3g4'))
