import timm
import vig_pytorch.pyramid_vig
from func import tensor_test

from mmcls.models import build_classifier

model_cfg = dict(
    type='ImageClassifier',
    backbone=dict(
        type='vig',
        k=9,
        n_classes=1000,
        n_blocks=16,
        epsilon=0.2,
        use_stochastic=False,
        drop_path=0,
        use_dilation=True,
        channels=320,
        dropout=0),
    # neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

model_dst = build_classifier(model_cfg)
model_src = timm.create_model('vig_s_224_gelu')
tensor_test(model_src, 'vig_checkpoint/vig_s_80.6.pth', model_dst,
            'vig_checkpoint_covert/vig_s.pth')
