_base_ = '../replknet-31B_32xb64-coslr-300e_224_in22kto1k.py'

model = dict(backbone=dict(small_kernel_merged=True))