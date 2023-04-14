import logging
import sys
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
from unittest.mock import Mock

import mmcv
import numpy as np
import torch
from mmengine import DictAction, MMLogger
from mmengine.dataset import Compose, default_collate
from mmengine.device import get_device
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.runner import Runner, load_checkpoint
from rich.console import Console
from rich.table import Table
from utils import substitute_weights

from mmpretrain.apis import ModelHub, get_model, list_models
from mmpretrain.datasets import CIFAR10, CIFAR100, ImageNet
from mmpretrain.utils import register_all_modules
from mmpretrain.visualization import UniversalVisualizer

console = Console()
MMCLS_ROOT = Path(__file__).absolute().parents[2]

classes_map = {
    'ImageNet-1k': ImageNet.CLASSES,
    'CIFAR-10': CIFAR10.CLASSES,
    'CIFAR-100': CIFAR100.CLASSES,
}

logger = MMLogger.get_instance('validation', logger_name='mmpretrain')
logger.handlers[0].stream = sys.stderr
logger.addHandler(logging.FileHandler('benchmark_valid.log', mode='w'))
# Force to use the logger in runners.
Runner.build_logger = Mock(return_value=logger)


def parse_args():
    parser = ArgumentParser(description='Valid all models in model-index.yml')
    parser.add_argument(
        '--checkpoint-root',
        help='Checkpoint file root path. If set, load checkpoint before test.')
    parser.add_argument('--img', default='demo/demo.JPEG', help='Image file')
    parser.add_argument('--models', nargs='+', help='models name to inference')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='the interval of show (s), 0 is block')
    parser.add_argument(
        '--inference-time',
        action='store_true',
        help='Test inference time by run 10 times for each model.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='The batch size during the inference.')
    parser.add_argument(
        '--flops', action='store_true', help='Get Flops and Params of models')
    parser.add_argument(
        '--flops-str',
        action='store_true',
        help='Output FLOPs and params counts in a string form.')
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


def inference(metainfo, checkpoint, work_dir, args, exp_name=None):
    cfg = metainfo.config
    cfg.work_dir = work_dir
    cfg.load_from = checkpoint
    cfg.log_level = 'WARN'
    cfg.experiment_name = exp_name or metainfo.name
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if 'test_dataloader' in cfg:
        # build the data pipeline
        test_dataset = cfg.test_dataloader.dataset
        if test_dataset.pipeline[0]['type'] != 'LoadImageFromFile':
            test_dataset.pipeline.insert(0, dict(type='LoadImageFromFile'))
        if test_dataset.type in ['CIFAR10', 'CIFAR100']:
            # The image shape of CIFAR is (32, 32, 3)
            test_dataset.pipeline.insert(1, dict(type='Resize', scale=32))

        data = Compose(test_dataset.pipeline)({'img_path': args.img})
        data = default_collate([data] * args.batch_size)
        resolution = tuple(data['inputs'].shape[-2:])
        model = Runner.from_cfg(cfg).model
        model = revert_sync_batchnorm(model)
        model.eval()
        forward = model.val_step
    else:
        # For configs without data settings.
        model = get_model(cfg, device=get_device())
        model = revert_sync_batchnorm(model)
        model.eval()
        data = torch.rand(1, 3, 224, 224).to(model.data_preprocessor.device)
        resolution = (224, 224)
        forward = model.extract_feat

    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location='cpu')

    # forward the model
    result = {'model': metainfo.name, 'resolution': resolution}
    with torch.no_grad():
        if args.inference_time:
            time_record = []
            forward(data)  # warmup before profiling
            for _ in range(10):
                torch.cuda.synchronize()
                start = perf_counter()
                forward(data)
                torch.cuda.synchronize()
                time_record.append(
                    (perf_counter() - start) / args.batch_size * 1000)
            result['time_mean'] = np.mean(time_record[1:-1])
            result['time_std'] = np.std(time_record[1:-1])
        else:
            forward(data)

    if args.flops:
        from mmengine.analysis import FlopAnalyzer, parameter_count
        from mmengine.analysis.print_helper import _format_size
        _format_size = _format_size if args.flops_str else lambda x: x
        with torch.no_grad():
            model.forward = model.extract_feat
            model.to('cpu')
            inputs = (torch.randn((1, 3, *resolution)), )
            analyzer = FlopAnalyzer(model, inputs)
            # extract_feat only includes backbone
            analyzer._enable_warn_uncalled_mods = False
            flops = _format_size(analyzer.total())
            params = _format_size(parameter_count(model)[''])
            result['flops'] = flops if args.flops_str else int(flops)
            result['params'] = params if args.flops_str else int(params)

    return result


def show_summary(summary_data, args):
    table = Table(title='Validation Benchmark Regression Summary')
    table.add_column('Model')
    table.add_column('Validation')
    table.add_column('Resolution (h w)')
    if args.inference_time:
        table.add_column('Inference Time (std) (ms/im)')
    if args.flops:
        table.add_column('Flops', justify='right', width=13)
        table.add_column('Params', justify='right', width=11)

    for model_name, summary in summary_data.items():
        row = [model_name]
        valid = summary['valid']
        color = {'PASS': 'green', 'CUDA OOM': 'yellow'}.get(valid, 'red')
        row.append(f'[{color}]{valid}[/{color}]')
        if valid == 'PASS':
            row.append(str(summary['resolution']))
            if args.inference_time:
                time_mean = f"{summary['time_mean']:.2f}"
                time_std = f"{summary['time_std']:.2f}"
                row.append(f'{time_mean}\t({time_std})'.expandtabs(8))
            if args.flops:
                row.append(str(summary['flops']))
                row.append(str(summary['params']))
        table.add_row(*row)


# Sample test whether the inference code is correct
def main(args):
    register_all_modules()

    if args.models:
        models = set()
        for pattern in args.models:
            models.update(list_models(pattern=pattern))
        if len(models) == 0:
            print('No model found, please specify models in:')
            print('\n'.join(list_models()))
            return
    else:
        models = list_models()

    summary_data = {}
    tmpdir = tempfile.TemporaryDirectory()
    for model_name in models:

        model_info = ModelHub.get(model_name)
        if model_info.config is None:
            continue

        logger.info(f'Processing: {model_name}')

        weights = model_info.weights
        if args.checkpoint_root is not None and weights is not None:
            checkpoint = substitute_weights(weights, args.checkpoint_root)
        else:
            checkpoint = None

        try:
            # build the model from a config file and a checkpoint file
            result = inference(model_info, checkpoint, tmpdir.name, args)
            result['valid'] = 'PASS'
        except Exception as e:
            if 'CUDA out of memory' in str(e):
                logger.error(f'"{model_name}" :\nCUDA out of memory')
                result = {'valid': 'CUDA OOM'}
            else:
                import traceback
                logger.error(f'"{model_name}" :\n{traceback.format_exc()}')
                result = {'valid': 'FAIL'}

        summary_data[model_name] = result
        # show the results
        if args.show:
            vis = UniversalVisualizer.get_instance('valid')
            vis.set_image(mmcv.imread(args.img))
            vis.draw_texts(
                texts='\n'.join([f'{k}: {v}' for k, v in result.items()]),
                positions=np.array([(5, 5)]))
            vis.show(wait_time=args.wait_time)

    tmpdir.cleanup()
    show_summary(summary_data, args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
