_base_ = '../replknet-XL_32xb64-coslr-300e_320_meg73mto1k.py'

model = dict(backbone=dict(small_kernel_merged=True))
