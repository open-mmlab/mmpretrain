import logging
import re
import tempfile
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path
from time import time

import mmcv
import numpy as np
import torch
from mmengine import Config, DictAction, MMLogger
from mmengine.dataset import Compose, default_collate
from mmengine.fileio import FileClient
from mmengine.runner import Runner, load_checkpoint
from modelindex.load_model_index import load
from rich.console import Console
from rich.table import Table

from mmcls.apis import init_model
from mmcls.datasets import CIFAR10, CIFAR100, ImageNet
from mmcls.utils import register_all_modules
from mmcls.visualization import ClsVisualizer

console = Console()
MMCLS_ROOT = Path(__file__).absolute().parents[2]

classes_map = {
    'ImageNet-1k': ImageNet.CLASSES,
    'CIFAR-10': CIFAR10.CLASSES,
    'CIFAR-100': CIFAR100.CLASSES,
}


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


def inference(config_file, checkpoint, work_dir, args, exp_name):
    cfg = Config.fromfile(config_file)
    cfg.work_dir = work_dir
    cfg.load_from = checkpoint
    cfg.log_level = 'WARN'
    cfg.experiment_name = exp_name
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
        forward = model.val_step
    else:
        # For configs only for get model.
        model = init_model(cfg)
        load_checkpoint(model, checkpoint, map_location='cpu')
        data = torch.empty(1, 3, 224, 224).to(model.data_preprocessor.device)
        resolution = (224, 224)
        forward = model.extract_feat

    # forward the model
    result = {'resolution': resolution}
    with torch.no_grad():
        if args.inference_time:
            time_record = []
            for _ in range(10):
                forward(data)  # warmup before profiling
                torch.cuda.synchronize()
                start = time()
                forward(data)
                torch.cuda.synchronize()
                time_record.append((time() - start) / args.batch_size * 1000)
            result['time_mean'] = np.mean(time_record[1:-1])
            result['time_std'] = np.std(time_record[1:-1])
        else:
            forward(data)

    result['model'] = config_file.stem

    if args.flops:
        from fvcore.nn import FlopCountAnalysis, parameter_count
        from fvcore.nn.print_model_statistics import _format_size
        _format_size = _format_size if args.flops_str else lambda x: x
        with torch.no_grad():
            if hasattr(model, 'extract_feat'):
                model.forward = model.extract_feat
                model.to('cpu')
                inputs = (torch.randn((1, 3, *resolution)), )
                flops = _format_size(FlopCountAnalysis(model, inputs).total())
                params = _format_size(parameter_count(model)[''])
                result['flops'] = flops if args.flops_str else int(flops)
                result['params'] = params if args.flops_str else int(params)
            else:
                result['flops'] = ''
                result['params'] = ''

    return result


def show_summary(summary_data, args):
    table = Table(title='Validation Benchmark Regression Summary')
    table.add_column('Model')
    table.add_column('Validation')
    table.add_column('Resolution (h, w)')
    if args.inference_time:
        table.add_column('Inference Time (std) (ms/im)')
    if args.flops:
        table.add_column('Flops', justify='right', width=13)
        table.add_column('Params', justify='right', width=11)

    for model_name, summary in summary_data.items():
        row = [model_name]
        valid = summary['valid']
        color = 'green' if valid == 'PASS' else 'red'
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

    console.print(table)


# Sample test whether the inference code is correct
def main(args):
    register_all_modules()
    model_index_file = MMCLS_ROOT / 'model-index.yml'
    model_index = load(str(model_index_file))
    model_index.build_models_with_collections()
    models = OrderedDict({model.name: model for model in model_index.models})

    logger = MMLogger(
        'validation',
        logger_name='validation',
        log_file='benchmark_test_image.log',
        log_level=logging.INFO)

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

    summary_data = {}
    tmpdir = tempfile.TemporaryDirectory()
    for model_name, model_info in models.items():

        if model_info.config is None:
            continue

        config = Path(model_info.config)
        assert config.exists(), f'{model_name}: {config} not found.'

        logger.info(f'Processing: {model_name}')

        http_prefix = 'https://download.openmmlab.com/mmclassification/'
        if args.checkpoint_root is not None:
            root = args.checkpoint_root
            if 's3://' in args.checkpoint_root:
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

        try:
            # build the model from a config file and a checkpoint file
            result = inference(MMCLS_ROOT / config, checkpoint, tmpdir.name,
                               args, model_name)
            result['valid'] = 'PASS'
        except Exception:
            import traceback
            logger.error(f'"{config}" :\n{traceback.format_exc()}')
            result = {'valid': 'FAIL'}

        summary_data[model_name] = result
        # show the results
        if args.show:
            vis = ClsVisualizer.get_instance('valid')
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
