import argparse
import fnmatch
import json
import logging
import os
import os.path as osp
import re
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import yaml
from modelindex.load_model_index import load
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from .utils import METRICS_MAP, MMCLS_ROOT

# Avoid to import MMPretrain to accelerate speed to show summary

console = Console()
logger = logging.getLogger('train')
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler('benchmark_train.log', mode='w'))
CYCLE_LEVELS = ['month', 'quarter', 'half-year', 'no-training']


class RangeAction(argparse.Action):

    def __call__(self, _, namespace, values: str, __):
        matches = re.match(r'([><=]*)([-\w]+)', values)
        if matches is None:
            raise ValueError(f'Unavailable range option {values}')
        symbol, range_str = matches.groups()
        assert range_str in CYCLE_LEVELS, \
            f'{range_str} are not in {CYCLE_LEVELS}.'
        level = CYCLE_LEVELS.index(range_str)
        symbol = symbol or '<='
        ranges = set()
        if '=' in symbol:
            ranges.add(level)
        if '>' in symbol:
            ranges.update(range(level + 1, len(CYCLE_LEVELS)))
        if '<' in symbol:
            ranges.update(range(level))
        assert len(ranges) > 0, 'No range are selected.'
        setattr(namespace, self.dest, ranges)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train models (in bench_train.yml) and compare accuracy.')
    parser.add_argument(
        '--local',
        action='store_true',
        help='run at local instead of cluster.')
    parser.add_argument(
        '--models', nargs='+', type=str, help='Specify model names to run.')
    parser.add_argument(
        '--run', action='store_true', help='run script directly')
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Summarize benchmark train results.')
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save the summary and archive log files.')
    parser.add_argument(
        '--non-distributed',
        action='store_true',
        help='Use non-distributed environment (for debug).')
    parser.add_argument(
        '--range',
        type=str,
        default={0},
        action=RangeAction,
        metavar='{month,quarter,half-year,no-training}',
        help='The training benchmark range, "no-training" means all models '
        "including those we haven't trained.")
    parser.add_argument(
        '--work-dir',
        default='work_dirs/benchmark_train',
        help='the dir to save train log')
    parser.add_argument('--port', type=int, default=29666, help='dist port')
    parser.add_argument(
        '--partition',
        type=str,
        default='mm_model',
        help='(for slurm) Cluster partition to use.')
    parser.add_argument(
        '--job-name',
        type=str,
        default='cls-train-benchmark',
        help='(for slurm) Slurm job name prefix')
    parser.add_argument(
        '--quotatype',
        default=None,
        choices=['reserved', 'auto', 'spot'],
        help='(for slurm) Quota type, only available for phoenix-slurm>=0.2')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        type=str,
        default=[],
        help='Config options for all config files.')

    args = parser.parse_args()
    return args


def get_gpu_number(model_info):
    config = osp.basename(model_info.config)
    matches = re.match(r'.*[-_](\d+)xb(\d+).*', config)
    if matches is None:
        raise ValueError(
            'Cannot get gpu numbers from the config name {config}')
    gpus = int(matches.groups()[0])
    return gpus


def create_train_job_batch(model_info, args, port, pretrain_info=None):
    model_name = model_info.name
    config = Path(model_info.config)
    gpus = get_gpu_number(model_info)

    job_name = f'{args.job_name}_{model_name}'
    work_dir = Path(args.work_dir) / model_name
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg_options = deepcopy(args.cfg_options)

    if args.quotatype is not None:
        quota_cfg = f'#SBATCH --quotatype {args.quotatype}'
    else:
        quota_cfg = ''

    if pretrain_info is not None:
        pretrain = Path(args.work_dir) / pretrain_info.name / 'last_checkpoint'
        pretrain_cfg = (f'model.backbone.init_cfg.checkpoint="$(<{pretrain})" '
                        'model.backbone.init_cfg.type="Pretrained" '
                        'model.backbone.init_cfg.prefix="backbone."')
    else:
        pretrain_cfg = ''

    if not args.local:
        launcher = 'slurm'
        runner = 'srun python'
        if gpus > 8:
            gpus = 8
            cfg_options.append('auto_scale_lr.enable=True')
    elif not args.non_distributed:
        launcher = 'pytorch'
        if gpus > 8:
            gpus = 8
            cfg_options.append('auto_scale_lr.enable=True')
        runner = ('torchrun --master_addr="127.0.0.1" '
                  f'--master_port={port} --nproc_per_node={gpus}')
    else:
        launcher = 'none'
        runner = 'python -u'

    job_script = (f'#!/bin/bash\n'
                  f'#SBATCH --output {work_dir}/job.%j.out\n'
                  f'#SBATCH --partition={args.partition}\n'
                  f'#SBATCH --job-name {job_name}\n'
                  f'#SBATCH --gres=gpu:{min(8, gpus)}\n'
                  f'{quota_cfg}\n'
                  f'#SBATCH --ntasks-per-node={min(8, gpus)}\n'
                  f'#SBATCH --ntasks={gpus}\n'
                  f'#SBATCH --cpus-per-task=5\n\n'
                  f'{runner} tools/train.py {config} '
                  f'--work-dir={work_dir} --cfg-option '
                  f'{" ".join(cfg_options)} '
                  f'default_hooks.checkpoint.max_keep_ckpts=2 '
                  f'default_hooks.checkpoint.save_best="auto" '
                  f'{pretrain_cfg} '
                  f'--launcher={launcher}\n')

    with open(work_dir / 'job.sh', 'w') as f:
        f.write(job_script)

    return work_dir / 'job.sh'


def train(models, args):
    port = args.port

    commands = []

    for model_info in models.values():
        script_path = create_train_job_batch(model_info, args, port)
        if hasattr(model_info, 'downstream'):
            downstream_info = model_info.downstream
            downstream_script = create_train_job_batch(
                downstream_info, args, port, pretrain_info=model_info)
        else:
            downstream_script = None

        if args.local:
            command = f'bash {script_path}'
            if downstream_script:
                command += f' && bash {downstream_script}'
        else:
            command = f'JOBID=$(sbatch --parsable {script_path})'
            if downstream_script:
                command += f' && sbatch --dependency=afterok:$JOBID {downstream_script}'  # noqa: E501
        commands.append(command)

        port += 1

    command_str = '\n'.join(commands)

    preview = Table()
    preview.add_column(str(script_path))
    preview.add_column('Shell command preview')
    preview.add_row(
        Syntax.from_path(
            script_path,
            background_color='default',
            line_numbers=True,
            word_wrap=True),
        Syntax(
            command_str,
            'bash',
            background_color='default',
            line_numbers=True,
            word_wrap=True))
    console.print(preview)

    if args.run:
        os.system(command_str)
    else:
        console.print('Please set "--run" to start the job')


def save_summary(summary_data, work_dir):
    date = datetime.now().strftime('%Y%m%d-%H%M%S')
    zip_path = work_dir / f'archive-{date}.zip'
    zip_file = ZipFile(zip_path, 'w')

    summary_path = work_dir / 'benchmark_summary.csv'
    file = open(summary_path, 'w')
    columns = defaultdict(list)
    for model_name, summary in summary_data.items():
        if len(summary) == 0:
            # Skip models without results
            continue
        columns['Name'].append(model_name)

        for metric_key in METRICS_MAP:
            if metric_key in summary:
                metric = summary[metric_key]
                expect = str(round(metric['expect'], 2))
                result = str(round(metric['result'], 2))
                columns[f'{metric_key} (expect)'].append(expect)
                columns[f'{metric_key}'].append(result)
                best = str(round(metric['best'], 2))
                best_epoch = str(int(metric['best_epoch']))
                columns[f'{metric_key} (best)'].append(best)
                columns[f'{metric_key} (best epoch)'].append(best_epoch)
            else:
                columns[f'{metric_key} (expect)'].append('')
                columns[f'{metric_key}'].append('')
                columns[f'{metric_key} (best)'].append('')
                columns[f'{metric_key} (best epoch)'].append('')

        columns['Log'].append(str(summary['log_file'].relative_to(work_dir)))
        zip_file.write(summary['log_file'])

    columns = {
        field: column
        for field, column in columns.items() if ''.join(column)
    }
    file.write(','.join(columns.keys()) + '\n')
    for row in zip(*columns.values()):
        file.write(','.join(row) + '\n')
    file.close()
    zip_file.write(summary_path)
    zip_file.close()
    logger.info('Summary file saved at ' + str(summary_path))
    logger.info('Log files archived at ' + str(zip_path))


def show_summary(summary_data):
    table = Table(title='Train Benchmark Regression Summary')
    table.add_column('Name')
    for metric in METRICS_MAP:
        table.add_column(f'{metric} (expect)')
        table.add_column(f'{metric}')
        table.add_column(f'{metric} (best)')
    table.add_column('Date')

    def set_color(value, expect):
        if value > expect:
            return 'green'
        elif value >= expect - 0.2:
            return 'white'
        else:
            return 'red'

    for model_name, summary in summary_data.items():
        row = [model_name]
        for metric_key in METRICS_MAP:
            if metric_key in summary:
                metric = summary[metric_key]
                expect = round(metric['expect'], 2)
                last = round(metric['last'], 2)
                last_epoch = metric['last_epoch']
                last_color = set_color(last, expect)
                best = metric['best']
                best_color = set_color(best, expect)
                best_epoch = round(metric['best_epoch'], 2)
                row.append(f'{expect:.2f}')
                row.append(
                    f'[{last_color}]{last:.2f}[/{last_color}] ({last_epoch})')
                row.append(
                    f'[{best_color}]{best:.2f}[/{best_color}] ({best_epoch})')
            else:
                row.extend([''] * 3)
        table.add_row(*row)

    # Remove empty columns
    table.columns = [
        column for column in table.columns if ''.join(column._cells)
    ]
    console.print(table)


def summary(models, args):
    work_dir = Path(args.work_dir)
    dir_map = {p.name: p for p in work_dir.iterdir() if p.is_dir()}

    summary_data = {}
    for model_name, model_info in models.items():

        summary_data[model_name] = {}

        if model_name not in dir_map:
            continue
        elif hasattr(model_info, 'downstream'):
            downstream_name = model_info.downstream.name
            if downstream_name not in dir_map:
                continue
            else:
                sub_dir = dir_map[downstream_name]
                model_info = model_info.downstream
        else:
            # Skip if not found any vis_data folder.
            sub_dir = dir_map[model_name]

        log_files = [f for f in sub_dir.glob('*/vis_data/scalars.json')]
        if len(log_files) == 0:
            continue
        log_file = sorted(log_files)[-1]

        # parse train log
        with open(log_file) as f:
            json_logs = [json.loads(s) for s in f.readlines()]
            # TODO: need a better method to extract validate log
            val_logs = [log for log in json_logs if 'loss' not in log]

        if len(val_logs) == 0:
            continue

        expect_metrics = model_info.results[0].metrics

        # extract metrics
        summary = {'log_file': log_file}
        for key_yml, key_res in METRICS_MAP.items():
            if key_yml in expect_metrics:
                assert key_res in val_logs[-1], \
                    f'{model_name}: No metric "{key_res}"'
                expect_result = float(expect_metrics[key_yml])
                last = float(val_logs[-1][key_res])
                best_log, best_epoch = sorted(
                    zip(val_logs, range(len(val_logs))),
                    key=lambda x: x[0][key_res])[-1]
                best = float(best_log[key_res])

                summary[key_yml] = dict(
                    expect=expect_result,
                    last=last,
                    last_epoch=len(val_logs),
                    best=best,
                    best_epoch=best_epoch + 1)
        summary_data[model_name].update(summary)

    show_summary(summary_data)
    if args.save:
        save_summary(summary_data, work_dir)


def main():
    args = parse_args()

    # parse model-index.yml
    model_index_file = MMCLS_ROOT / 'model-index.yml'
    model_index = load(str(model_index_file))
    model_index.build_models_with_collections()
    all_models = {model.name: model for model in model_index.models}

    with open(Path(__file__).parent / 'bench_train.yml', 'r') as f:
        train_items = yaml.safe_load(f)
    models = {}
    for item in train_items:
        name = item['Name']
        cycle = item['Cycle']
        cycle_level = CYCLE_LEVELS.index(cycle)
        if cycle_level in args.range:
            model_info = all_models[name]
            if 'Downstream' in item:
                downstream = item['Downstream']
                setattr(model_info, 'downstream', all_models[downstream])
            models[name] = model_info

    if args.models:
        filter_models = {}
        for pattern in args.models:
            filter_models.update({
                name: models[name]
                for name in fnmatch.filter(models, pattern + '*')
            })
        if len(filter_models) == 0:
            logger.error('No model found, please specify models in:\n' +
                         '\n'.join(models.keys()))
            return
        models = filter_models

    if args.summary:
        summary(models, args)
    else:
        train(models, args)


if __name__ == '__main__':
    main()
