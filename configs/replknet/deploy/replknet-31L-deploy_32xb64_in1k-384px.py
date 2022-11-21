_base_ = '../replknet-31L_32xb64_in1k-384px.py'

model = dict(backbone=dict(small_kernel_merged=True))
