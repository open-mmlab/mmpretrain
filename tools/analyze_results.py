import argparse
import os.path as osp

import mmcv
from mmcv import DictAction

from mmcls.datasets import build_dataset
from mmcls.models import build_classifier


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
        help='Number of images to select for sucess/fail')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    args = parser.parse_args()
    return args


def save_imgs(result_dir, folder_name, results, model):
    full_dir = osp.join(result_dir, folder_name)
    mmcv.mkdir_or_exist(full_dir)
    mmcv.dump(results, osp.join(full_dir, folder_name + '.json'))

    # save imgs
    show_keys = ['pred_score', 'pred_class', 'gt_class']
    for result in results:
        result_show = dict((k, v) for k, v in result.items() if k in show_keys)
        outfile = osp.join(full_dir, osp.basename(result['filename']))
        model.show_result(result['filename'], result_show, out_file=outfile)


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    model = build_classifier(cfg.model)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    filenames = list()
    for info in dataset.data_infos:
        if info['img_prefix'] is not None:
            filename = osp.join(info['img_prefix'],
                                info['img_info']['filename'])
        else:
            filename = info['img_info']['filename']
        filenames.append(filename)
    gt_labels = list(dataset.get_gt_labels())
    gt_classes = [dataset.CLASSES[x] for x in gt_labels]

    # load test results
    outputs = mmcv.load(args.result)
    outputs['filename'] = filenames
    outputs['gt_label'] = gt_labels
    outputs['gt_class'] = gt_classes

    outputs_list = list()
    for i in range(len(gt_labels)):
        output = dict()
        for k in outputs.keys():
            output[k] = outputs[k][i]
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

    save_imgs(args.out_dir, 'success', success, model)
    save_imgs(args.out_dir, 'fail', fail, model)


if __name__ == '__main__':
    main()
