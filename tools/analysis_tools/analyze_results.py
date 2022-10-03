# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from pathlib import Path

import torch
import mmengine
import mmcv
from mmengine import DictAction

from mmcls.datasets import build_dataset
from mmcls.visualization import ClsVisualizer
from mmcls.structures import ClsDataSample


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMCls evaluate prediction success/fail')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('result', help='test result json/pkl file')
    parser.add_argument('--out-dir', help='dir to store output files')
    parser.add_argument(
        '--topk',
        default=20,
        type=int,
        help='Number of images to select for success/fail')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()

    return args


def save_imgs(result_dir, folder_name, results, dataset):
    full_dir = osp.join(result_dir, folder_name)
    vis = ClsVisualizer(
        save_dir=full_dir, vis_backends=[dict(type='LocalVisBackend')]
    )
    vis.dataset_meta = {'classes': dataset.CLASSES}

    # save imgs
    for result in results:
        data_sample = ClsDataSample().set_gt_label(
            result['gt_label']
            ).set_pred_label(
                result['pred_label']
                ).set_pred_score(torch.Tensor(result['pred_scores']))
        img = mmcv.imread(result['img_path'], channel_order='rgb')
        vis.add_datasample(result['filename'], img, data_sample)

    mmengine.dump(results, osp.join(full_dir, folder_name + '.json'))


def main():
    args = parse_args()

    # load test results
    outputs = mmengine.load(args.result)

    cfg = mmengine.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # build the dataloader
    cfg.test_dataloader.dataset.pipeline = []
    dataset = build_dataset(cfg.test_dataloader.dataset)

    outputs_list = list()
    for i in range(len(outputs)):
        output = dict()
        output['img_path'] = outputs[i]['img_path']
        output['filename'] = Path(outputs[i]['img_path']).name
        output['gt_label'] = int(outputs[i]['gt_label']['label'][0])
        output['pred_score'] = float(torch.max(
            outputs[i]['pred_label']['score']).item())
        output['pred_scores'] = outputs[i]['pred_label']['score'].tolist()
        output['pred_label'] = int(outputs[i]['pred_label']['label'][0])
        outputs_list.append(output)

    # sort result
    outputs_list = sorted(outputs_list, key=lambda x: x['pred_score'])

    success = list()
    fail = list()
    for output in outputs_list:
        if output['pred_label'] == output['gt_label']:
            success.append(output)
        else:
            fail.append(output)

    success = success[:args.topk]
    fail = fail[:args.topk]

    save_imgs(args.out_dir, 'success', success, dataset)
    save_imgs(args.out_dir, 'fail', fail, dataset)


if __name__ == '__main__':
    main()
