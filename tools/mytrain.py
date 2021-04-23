import argparse
import os
import warnings

import mmcv
# import torch
from mmcv import DictAction
from mmcv.runner import load_checkpoint

from mmcls.models import build_classifier

# TODO import `wrap_fp16_model` from mmcv and delete them from mmcls
try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('wrap_fp16_model from mmcls will be deprecated.'
                  'Please install mmcv>=1.1.4.')
    from mmcls.core import wrap_fp16_model


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
        '"accuracy", "precision", "recall", "f1_score", "support" for single '
        'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
        'multi-label dataset')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be parsed as a dict metric_options for dataset.evaluate()'
        ' function.')
    parser.add_argument(
        '--show-options',
        nargs='+',
        action=DictAction,
        help='custom options for show_result. key-value pair in xxx=yyy.'
        'Check available options in `model.show_result`.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    # args = parse_args()

    cfg = mmcv.Config.fromfile('../configs/vision_transformer/'
                               'vit_base_patch16_384_finetune_imagenet.py')

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model, '../model_zoo/vit_base_patch16_384.pth', map_location='cpu')
    print(len(checkpoint))
    model.eval()
    print(model.training)
    # backbone = model.backbone
    #
    # imgs = torch.ones(1, 3, 384, 384)
    # label = torch.randint(0, 1000, (1, ))
    # x = imgs
    # B = x.shape[0]
    # x = backbone.patch_embed(x)
    #
    # cls_tokens = backbone.cls_token.expand(
    #     B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    # x = torch.cat((cls_tokens, x), dim=1)
    # x = x + backbone.pos_embed
    # x = backbone.drop_after_pos(x)
    # x = backbone.layers[0](x)
    # print(model.simple_test(imgs))


if __name__ == '__main__':
    main()
