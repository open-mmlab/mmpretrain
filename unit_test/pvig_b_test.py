import timm
import vig_pytorch.pyramid_vig
from func import tensor_test

from mmcls.models import build_classifier

blocks = [2, 2, 18, 2]
channels = [128, 256, 512, 1024]
model_cfg = dict(
    type='ImageClassifier',
    backbone=dict(
        type='pyramid_vig',
        k=9,
        dropout=0,
        use_dilation=True,
        epsilon=0.2,
        use_stochastic=False,
        drop_path=0,
        blocks=blocks,
        channels=channels,
        n_classes=1000,
        emb_dims=1024),
    # neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
model_dst = build_classifier(model_cfg)
model_src = timm.create_model('pvig_b_224_gelu')
tensor_test(model_src, 'vig_checkpoint/pvig_b_83.66.pth.tar', model_dst,
            'vig_checkpoint_covert/pvig_b.pth')
