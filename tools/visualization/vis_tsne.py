# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import time
from functools import partial
from typing import Optional

import matplotlib.pyplot as plt
import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config, DictAction
from mmengine.dataset import default_collate, worker_init_fn
from mmengine.dist import get_rank
from mmengine.logging import MMLogger
from mmengine.utils import mkdir_or_exist
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from mmpretrain.apis import get_model
from mmpretrain.registry import DATA_SAMPLERS, DATASETS


def parse_args():
    parser = argparse.ArgumentParser(description='t-SNE visualization')
    parser.add_argument('config', help='tsne config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--vis-stage',
        choices=['backbone', 'neck', 'pre_logits'],
        default='backbone',
        help='the visualization stage of the model')
    parser.add_argument(
        '--max-num-class',
        type=int,
        default=20,
        help='the maximum number of classes to apply t-SNE algorithms, now the'
        'function supports maximum 20 classes')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
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
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')

    # t-SNE settings
    parser.add_argument(
        '--n-components', type=int, default=2, help='the dimension of results')
    parser.add_argument(
        '--perplexity',
        type=float,
        default=30.0,
        help='The perplexity is related to the number of nearest neighbors'
        'that is used in other manifold learning algorithms.')
    parser.add_argument(
        '--early-exaggeration',
        type=float,
        default=12.0,
        help='Controls how tight natural clusters in the original space are in'
        'the embedded space and how much space will be between them.')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=200.0,
        help='The learning rate for t-SNE is usually in the range'
        '[10.0, 1000.0]. If the learning rate is too high, the data may look'
        'like a ball with any point approximately equidistant from its nearest'
        'neighbours. If the learning rate is too low, most points may look'
        'compressed in a dense cloud with few outliers.')
    parser.add_argument(
        '--n-iter',
        type=int,
        default=1000,
        help='Maximum number of iterations for the optimization. Should be at'
        'least 250.')
    parser.add_argument(
        '--n-iter-without-progress',
        type=int,
        default=300,
        help='Maximum number of iterations without progress before we abort'
        'the optimization.')
    parser.add_argument(
        '--init', type=str, default='random', help='The init method')
    args = parser.parse_args()
    return args


def post_process():
    pass


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        work_type = args.config.split('/')[1]
        cfg.work_dir = osp.join('./work_dirs', work_type,
                                osp.splitext(osp.basename(args.config))[0])

    # create work_dir
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    tsne_work_dir = osp.join(cfg.work_dir, f'tsne_{timestamp}/')
    mkdir_or_exist(osp.abspath(tsne_work_dir))

    # init the logger before other steps
    log_file = osp.join(tsne_work_dir, 'tsne.log')
    logger = MMLogger.get_instance(
        'mmpretrain',
        logger_name='mmpretrain',
        log_file=log_file,
        log_level=cfg.log_level)

    # build the model from a config file and a checkpoint file
    model = get_model(cfg, args.checkpoint, device=args.device)
    logger.info(f'Model loaded and the output indices of backbone is '
                f'{model.backbone.out_indices}.')

    # build the dataset
    tsne_dataloader_cfg = cfg.get('test_dataloader')
    tsne_dataset_cfg = tsne_dataloader_cfg.pop('dataset')
    if isinstance(tsne_dataset_cfg, dict):
        dataset = DATASETS.build(tsne_dataset_cfg)
        if hasattr(dataset, 'full_init'):
            dataset.full_init()

    # compress dataset, select that the label is less then max_num_class
    subset_idx_list = []
    for i in range(len(dataset)):
        if dataset.get_data_info(i)['gt_label'] < args.max_num_class:
            subset_idx_list.append(i)
    dataset.get_subset_(subset_idx_list)
    logger.info(f'Apply t-SNE to visualize {len(subset_idx_list)} samples.')

    # build sampler
    sampler_cfg = tsne_dataloader_cfg.pop('sampler')
    if isinstance(sampler_cfg, dict):
        sampler = DATA_SAMPLERS.build(
            sampler_cfg, default_args=dict(dataset=dataset, seed=args.seed))

    # build dataloader
    init_fn: Optional[partial]
    if args.seed is not None:
        init_fn = partial(
            worker_init_fn,
            num_workers=tsne_dataloader_cfg.get('num_workers'),
            rank=get_rank(),
            seed=args.seed)
    else:
        init_fn = None

    tsne_dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        collate_fn=default_collate,
        worker_init_fn=init_fn,
        **tsne_dataloader_cfg)

    results = dict()
    features = []
    labels = []
    progress_bar = mmengine.ProgressBar(len(tsne_dataloader))
    for _, data in enumerate(tsne_dataloader):
        with torch.no_grad():
            # preprocess data
            data = model.data_preprocessor(data)
            batch_inputs, batch_data_samples = \
                data['inputs'], data['data_samples']

            # extract backbone features
            batch_features = model.extract_feat(
                batch_inputs, stage=args.vis_stage)

            # post process
            if args.vis_stage == 'backbone':
                if getattr(model.backbone, 'output_cls_token', False) is False:
                    batch_features = [
                        F.adaptive_avg_pool2d(inputs, 1).squeeze()
                        for inputs in batch_features
                    ]
                else:
                    # output_cls_token is True, here t-SNE uses cls_token
                    batch_features = [feat[-1] for feat in batch_features]

            batch_labels = torch.cat([i.gt_label for i in batch_data_samples])

        # save batch features
        features.append(batch_features)
        labels.extend(batch_labels.cpu().numpy())
        progress_bar.update()

    for i in range(len(features[0])):
        key = 'feat_' + str(model.backbone.out_indices[i])
        results[key] = np.concatenate(
            [batch[i].cpu().numpy() for batch in features], axis=0)

    # save features
    for key, val in results.items():
        output_file = f'{tsne_work_dir}{key}.npy'
        np.save(output_file, val)

    # build t-SNE model
    tsne_model = TSNE(
        n_components=args.n_components,
        perplexity=args.perplexity,
        early_exaggeration=args.early_exaggeration,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        n_iter_without_progress=args.n_iter_without_progress,
        init=args.init)

    # run and get results
    logger.info('Running t-SNE.')
    for key, val in results.items():
        result = tsne_model.fit_transform(val)
        res_min, res_max = result.min(0), result.max(0)
        res_norm = (result - res_min) / (res_max - res_min)
        plt.figure(figsize=(10, 10))
        plt.scatter(
            res_norm[:, 0],
            res_norm[:, 1],
            alpha=1.0,
            s=15,
            c=labels,
            cmap='tab20')
        plt.savefig(f'{tsne_work_dir}{key}.png')
    logger.info(f'Save features and results to {tsne_work_dir}')


if __name__ == '__main__':
    main()
