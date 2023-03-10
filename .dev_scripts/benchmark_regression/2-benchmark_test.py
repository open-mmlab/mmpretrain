import argparse
import fnmatch
import logging
import os
import os.path as osp
import pickle
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

from modelindex.load_model_index import load
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from utils import METRICS_MAP, MMCLS_ROOT, substitute_weights

# Avoid to import MMPretrain to accelerate speed to show summary

console = Console()
logger = logging.getLogger('test')
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler('benchmark_test.log', mode='w'))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test all models' accuracy in model-index.yml")
    parser.add_argument('checkpoint_root', help='Checkpoint file root path.')
    parser.add_argument(
        '--local', action='store_true', help='run at local instead of slurm.')
    parser.add_argument(
        '--models', nargs='+', type=str, help='Specify model names to run.')
    parser.add_argument(
        '--run', action='store_true', help='run script directly')
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Summarize benchmark test results.')
    parser.add_argument('--save', action='store_true', help='Save the summary')
    parser.add_argument(
        '--gpus', type=int, default=1, help='How many GPUS to use.')
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Whether to skip models without results record in the metafile.')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/benchmark_test',
        help='the dir to save metric')
    parser.add_argument('--port', type=int, default=29666, help='dist port')
    parser.add_argument(
        '--partition',
        type=str,
        default='mm_model',
        help='(for slurm) Cluster partition to use.')
    parser.add_argument(
        '--job-name',
        type=str,
        default='cls-test-benchmark',
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


def create_test_job_batch(commands, model_info, args, port, script_name):
    model_name = model_info.name
    config = Path(model_info.config)

    if model_info.weights is not None:
        checkpoint = substitute_weights(model_info.weights,
                                        args.checkpoint_root)
        if checkpoint is None:
            logger.warning(f'{model_name}: {checkpoint} not found.')
            return None
    else:
        return None

    job_name = f'{args.job_name}_{model_name}'
    work_dir = Path(args.work_dir) / model_name
    work_dir.mkdir(parents=True, exist_ok=True)
    result_file = work_dir / 'result.pkl'

    if args.quotatype is not None:
        quota_cfg = f'#SBATCH --quotatype {args.quotatype}'
    else:
        quota_cfg = ''

    if not args.local:
        launcher = 'srun python'
        runner = 'slurm'
    elif args.gpus > 1:
        launcher = 'pytorch'
        runner = ('torchrun --master_addr="127.0.0.1" '
                  f'--master_port={port} --nproc_per_node={args.gpus}')
    else:
        launcher = 'none'
        runner = 'python -u'

    job_script = (f'#!/bin/bash\n'
                  f'#SBATCH --output {work_dir}/job.%j.out\n'
                  f'#SBATCH --partition={args.partition}\n'
                  f'#SBATCH --job-name {job_name}\n'
                  f'#SBATCH --gres=gpu:{min(8, args.gpus)}\n'
                  f'{quota_cfg}\n'
                  f'#SBATCH --ntasks-per-node={min(8, args.gpus)}\n'
                  f'#SBATCH --ntasks={args.gpus}\n'
                  f'#SBATCH --cpus-per-task=5\n\n'
                  f'{runner} {script_name} {config} {checkpoint} '
                  f'--work-dir={work_dir} --cfg-option '
                  f'env_cfg.dist_cfg.port={port} '
                  f'{" ".join(args.cfg_options)} '
                  f'--out={result_file} --out-item="metrics" '
                  f'--launcher={launcher}\n')

    with open(work_dir / 'job.sh', 'w') as f:
        f.write(job_script)

    commands.append(f'echo "{config}"')
    if args.local:
        commands.append(f'bash {work_dir}/job.sh')
    else:
        commands.append(f'sbatch {work_dir}/job.sh')

    return work_dir / 'job.sh'


def test(models, args):
    script_name = osp.join('tools', 'test.py')
    port = args.port

    commands = []

    preview_script = ''
    for model_info in models.values():

        if model_info.results is None:
            # Skip pre-train model
            continue

        script_path = create_test_job_batch(commands, model_info, args, port,
                                            script_name)
        preview_script = script_path or preview_script
        port += 1

    command_str = '\n'.join(commands)

    preview = Table()
    preview.add_column(str(preview_script))
    preview.add_column('Shell command preview')
    preview.add_row(
        Syntax.from_path(
            preview_script,
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
    summary_path = work_dir / 'test_benchmark_summary.csv'
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
                expect = round(metric['expect'], 2)
                result = round(metric['result'], 2)
                columns[f'{metric_key} (expect)'].append(str(expect))
                columns[f'{metric_key}'].append(str(result))
            else:
                columns[f'{metric_key} (expect)'].append('')
                columns[f'{metric_key}'].append('')

    columns = {
        field: column
        for field, column in columns.items() if ''.join(column)
    }
    file.write(','.join(columns.keys()) + '\n')
    for row in zip(*columns.values()):
        file.write(','.join(row) + '\n')
    file.close()
    logger.info('Summary file saved at ' + str(summary_path))


def show_summary(summary_data):
    table = Table(title='Test Benchmark Regression Summary')
    table.add_column('Name')
    for metric in METRICS_MAP:
        table.add_column(f'{metric} (expect)')
        table.add_column(f'{metric}')
    table.add_column('Date')

    def set_color(value, expect):
        if value > expect + 0.01:
            return 'green'
        elif value >= expect - 0.01:
            return 'white'
        else:
            return 'red'

    for model_name, summary in summary_data.items():
        row = [model_name]
        for metric_key in METRICS_MAP:
            if metric_key in summary:
                metric = summary[metric_key]
                expect = round(metric['expect'], 2)
                result = round(metric['result'], 2)
                color = set_color(result, expect)
                row.append(f'{expect:.2f}')
                row.append(f'[{color}]{result:.2f}[/{color}]')
            else:
                row.extend([''] * 2)
        if 'date' in summary:
            row.append(summary['date'])
        else:
            row.append('')
        table.add_row(*row)

    # Remove empty columns
    table.columns = [
        column for column in table.columns if ''.join(column._cells)
    ]
    console.print(table)


def summary(models, args):
    work_dir = Path(args.work_dir)

    summary_data = {}
    for model_name, model_info in models.items():

        if model_info.results is None and not args.no_skip:
            continue

        # Skip if not found result file.
        result_file = work_dir / model_name / 'result.pkl'
        if not result_file.exists():
            summary_data[model_name] = {}
            continue

        with open(result_file, 'rb') as file:
            results = pickle.load(file)
        date = datetime.fromtimestamp(result_file.lstat().st_mtime)

        expect_metrics = model_info.results[0].metrics

        # extract metrics
        summary = {'date': date.strftime('%Y-%m-%d')}
        for key_yml, key_res in METRICS_MAP.items():
            if key_yml in expect_metrics and key_res in results:
                expect_result = float(expect_metrics[key_yml])
                result = float(results[key_res])
                summary[key_yml] = dict(expect=expect_result, result=result)

        summary_data[model_name] = summary

    show_summary(summary_data)
    if args.save:
        save_summary(summary_data, work_dir)


def main():
    args = parse_args()

    # parse model-index.yml
    model_index_file = MMCLS_ROOT / 'model-index.yml'
    model_index = load(str(model_index_file))
    model_index.build_models_with_collections()
    models = OrderedDict({model.name: model for model in model_index.models})

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
        test(models, args)


if __name__ == '__main__':
    main()
