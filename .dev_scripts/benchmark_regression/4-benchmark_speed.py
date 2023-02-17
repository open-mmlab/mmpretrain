import logging
import re
import time
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import OrderedDict

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel.data_parallel import MMDataParallel
from mmcv.parallel.distributed import MMDistributedDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmengine.logging.logger import MMLogger
from modelindex.load_model_index import load
from rich.console import Console
from rich.table import Table

from mmpretrain.datasets.builder import build_dataloader
from mmpretrain.datasets.pipelines import Compose
from mmpretrain.models.builder import build_classifier

console = Console()
MMCLS_ROOT = Path(__file__).absolute().parents[2]
logger = MMLogger(
    name='benchmark',
    logger_name='benchmark',
    log_file='benchmark_speed.log',
    log_level=logging.INFO)


def parse_args():
    parser = ArgumentParser(
        description='Get FPS of all models in model-index.yml')
    parser.add_argument(
        '--checkpoint-root',
        help='Checkpoint file root path. If set, load checkpoint before test.')
    parser.add_argument(
        '--models', nargs='+', help='models name to inference.')
    parser.add_argument(
        '--work-dir',
        type=Path,
        default='work_dirs/benchmark_speed',
        help='the dir to save speed test results')
    parser.add_argument(
        '--max-iter', type=int, default=2048, help='num of max iter')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='The batch size to inference.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--device', default='cuda', help='Device used for inference')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    args = parser.parse_args()
    return args


class ToyDataset:
    """A dummy dataset used to provide images for benchmark."""

    def __init__(self, num, hw) -> None:
        data = []
        for _ in range(num):
            if isinstance(hw, int):
                w = h = hw
            else:
                w, h = hw
            img = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
            data.append({'img': img})
        self.data = data
        self.pipeline = None

    def __getitem__(self, idx):
        return self.pipeline(deepcopy(self.data[idx]))

    def __len__(self):
        return len(self.data)


def measure_fps(config_file, checkpoint, dataset, args, distributed=False):
    cfg = Config.fromfile(config_file)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # build the data pipeline
    if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
        cfg.data.test.pipeline.pop(0)

    dataset.pipeline = Compose(cfg.data.test.pipeline)
    resolution = tuple(dataset[0]['img'].shape[1:])

    # build the dataloader
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.batch_size,
        # Because multiple processes will occupy additional CPU resources,
        # FPS statistics will be more unstable when workers_per_gpu is not 0.
        # It is reasonable to set workers_per_gpu to 0.
        workers_per_gpu=0,
        dist=False if args.launcher == 'none' else True,
        shuffle=False,
        drop_last=True,
        persistent_workers=False)

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location='cpu')

    if not distributed:
        if args.device == 'cpu':
            model = model.cpu()
        else:
            model = MMDataParallel(model, device_ids=[args.gpu_id])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    infer_time = []
    fps = 0

    # forward the model
    result = {'model': config_file.stem, 'resolution': resolution}
    for i, data in enumerate(data_loader):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(return_loss=False, **data)

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start_time) / args.batch_size

        if i >= num_warmup:
            infer_time.append(elapsed)
            if (i + 1) % 8 == 0:
                fps = (i + 1 - num_warmup) / sum(infer_time)
                print(
                    f'Done image [{(i + 1)*args.batch_size:<4}/'
                    f'{args.max_iter}], fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)
    result['fps'] = (len(data_loader) - num_warmup) / sum(infer_time)
    result['time_mean'] = np.mean(infer_time) * 1000
    result['time_std'] = np.std(infer_time) * 1000

    return result


def show_summary(summary_data, args):
    table = Table(title='Speed Benchmark Regression Summary')
    table.add_column('Model')
    table.add_column('Resolution (h, w)')
    table.add_column('FPS (img/s)')
    table.add_column('Inference Time (std) (ms/img)')

    for model_name, summary in summary_data.items():
        row = [model_name]
        row.append(str(summary['resolution']))
        row.append(f"{summary['fps']:.2f}")
        time_mean = f"{summary['time_mean']:.2f}"
        time_std = f"{summary['time_std']:.2f}"
        row.append(f'{time_mean}\t({time_std})'.expandtabs(8))
        table.add_row(*row)

    console.print(table)


# Sample test whether the inference code is correct
def main(args):
    model_index_file = MMCLS_ROOT / 'model-index.yml'
    model_index = load(str(model_index_file))
    model_index.build_models_with_collections()
    models = OrderedDict({model.name: model for model in model_index.models})

    if args.models:
        patterns = [re.compile(pattern) for pattern in args.models]
        filter_models = {}
        for k, v in models.items():
            if any([re.match(pattern, k) for pattern in patterns]):
                filter_models[k] = v
        if len(filter_models) == 0:
            print('No model found, please specify models in:')
            print('\n'.join(models.keys()))
            return
        models = filter_models

    dataset_map = {
        # come from the average size of ImageNet
        'ImageNet-1k': ToyDataset(args.max_iter, (442, 522)),
        'CIFAR-10': ToyDataset(args.max_iter, 32),
        'CIFAR-100': ToyDataset(args.max_iter, 32),
    }

    summary_data = {}
    for model_name, model_info in models.items():

        if model_info.config is None:
            continue

        config = Path(model_info.config)
        assert config.exists(), f'{model_name}: {config} not found.'

        logger.info(f'Processing: {model_name}')

        http_prefix = 'https://download.openmmlab.com/mmclassification/'
        dataset = model_info.results[0].dataset
        if dataset not in dataset_map.keys():
            continue
        if args.checkpoint_root is not None:
            root = args.checkpoint_root
            if 's3://' in args.checkpoint_root:
                from mmcv.fileio import FileClient
                from petrel_client.common.exception import AccessDeniedError
                file_client = FileClient.infer_client(uri=root)
                checkpoint = file_client.join_path(
                    root, model_info.weights[len(http_prefix):])
                try:
                    exists = file_client.exists(checkpoint)
                except AccessDeniedError:
                    exists = False
            else:
                checkpoint = Path(root) / model_info.weights[len(http_prefix):]
                exists = checkpoint.exists()
            if exists:
                checkpoint = str(checkpoint)
            else:
                print(f'WARNING: {model_name}: {checkpoint} not found.')
                checkpoint = None
        else:
            checkpoint = None

        # build the model from a config file and a checkpoint file
        result = measure_fps(MMCLS_ROOT / config, checkpoint,
                             dataset_map[dataset], args)

        summary_data[model_name] = result

    show_summary(summary_data, args)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.work_dir / datetime.now().strftime('%Y-%m-%d.csv')
    with open(out_path, 'w') as f:
        f.write('MODEL,SHAPE,FPS\n')
        for model, summary in summary_data.items():
            f.write(
                f'{model},"{summary["resolution"]}",{summary["fps"]:.2f}\n')


if __name__ == '__main__':
    args = parse_args()
    main(args)
