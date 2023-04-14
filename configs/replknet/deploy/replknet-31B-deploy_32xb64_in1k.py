_base_ = '../replknet-31B_32xb64_in1k.py'

model = dict(backbone=dict(small_kernel_merged=True))
