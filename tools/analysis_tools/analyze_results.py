# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from pathlib import Path

import mmcv
import mmengine
import torch
from mmengine import DictAction

from mmcls.datasets import build_dataset
from mmcls.structures import ClsDataSample
from mmcls.visualization import ClsVisualizer


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
        '--rescale-factor',
        '-r',
        type=float,
        help='image rescale factor, which is useful if the output is too '
        'large or too small.')
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


def save_imgs(result_dir, folder_name, results, dataset, rescale_factor=None):
    full_dir = osp.join(result_dir, folder_name)
    vis = ClsVisualizer()
    vis.dataset_meta = {'classes': dataset.CLASSES}

    # save imgs
    for result in results:
        data_sample = ClsDataSample()\
            .set_gt_label(result['gt_label'])\
            .set_pred_label(result['pred_label'])\
            .set_pred_score(result['pred_scores'])
        data_info = dataset.get_data_info(result['sample_idx'])
        if 'img' in data_info:
            img = data_info['img']
            name = str(result['sample_idx'])
        elif 'img_path' in data_info:
            img = mmcv.imread(data_info['img_path'], channel_order='rgb')
            name = Path(data_info['img_path']).name
        else:
            raise ValueError('Cannot load images from the dataset infos.')
        if rescale_factor is not None:
            img = mmcv.imrescale(img, rescale_factor)
        vis.add_datasample(
            name, img, data_sample, out_file=osp.join(full_dir, name + '.png'))

        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.tolist()

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
        output['sample_idx'] = outputs[i]['sample_idx']
        output['gt_label'] = outputs[i]['gt_label']['label']
        output['pred_score'] = float(
            torch.max(outputs[i]['pred_label']['score']).item())
        output['pred_scores'] = outputs[i]['pred_label']['score']
        output['pred_label'] = outputs[i]['pred_label']['label']
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

    save_imgs(args.out_dir, 'success', success, dataset, args.rescale_factor)
    save_imgs(args.out_dir, 'fail', fail, dataset, args.rescale_factor)


if __name__ == '__main__':
    main()
