import argparse
import os
import os.path as osp
import pickle
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from modelindex.load_model_index import load
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

console = Console()
MMCLS_ROOT = Path(__file__).absolute().parents[2]
METRICS_MAP = {
    'Top 1 Accuracy': 'accuracy_top-1',
    'Top 5 Accuracy': 'accuracy_top-5'
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test all models' accuracy in model-index.yml")
    parser.add_argument(
        'partition', type=str, help='Cluster partition to use.')
    parser.add_argument('checkpoint_root', help='Checkpoint file root path.')
    parser.add_argument(
        '--job-name',
        type=str,
        default='cls-test-benchmark',
        help='Slurm job name prefix')
    parser.add_argument('--port', type=int, default=29666, help='dist port')
    parser.add_argument(
        '--models', nargs='+', type=str, help='Specify model names to run.')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/benchmark_test',
        help='the dir to save metric')
    parser.add_argument(
        '--run', action='store_true', help='run script directly')
    parser.add_argument(
        '--local',
        action='store_true',
        help='run at local instead of cluster.')
    parser.add_argument(
        '--mail', type=str, help='Mail address to watch test status.')
    parser.add_argument(
        '--mail-type',
        nargs='+',
        default=['BEGIN'],
        choices=['NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'],
        help='Mail address to watch test status.')
    parser.add_argument(
        '--quotatype',
        default=None,
        choices=['reserved', 'auto', 'spot'],
        help='Quota type, only available for phoenix-slurm>=0.2')
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Summarize benchmark test results.')
    parser.add_argument('--save', action='store_true', help='Save the summary')

    args = parser.parse_args()
    return args


def create_test_job_batch(commands, model_info, args, port, script_name):

    fname = model_info.name

    config = Path(model_info.config)
    assert config.exists(), f'{fname}: {config} not found.'

    http_prefix = 'https://download.openmmlab.com/mmclassification/'
    if 's3://' in args.checkpoint_root:
        from mmcv.fileio import FileClient
        from petrel_client.common.exception import AccessDeniedError
        file_client = FileClient.infer_client(uri=args.checkpoint_root)
        checkpoint = file_client.join_path(
            args.checkpoint_root, model_info.weights[len(http_prefix):])
        try:
            exists = file_client.exists(checkpoint)
        except AccessDeniedError:
            exists = False
    else:
        checkpoint_root = Path(args.checkpoint_root)
        checkpoint = checkpoint_root / model_info.weights[len(http_prefix):]
        exists = checkpoint.exists()
    if not exists:
        print(f'WARNING: {fname}: {checkpoint} not found.')
        return None

    job_name = f'{args.job_name}_{fname}'
    work_dir = Path(args.work_dir) / fname
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.mail is not None and 'NONE' not in args.mail_type:
        mail_cfg = (f'#SBATCH --mail {args.mail}\n'
                    f'#SBATCH --mail-type {args.mail_type}\n')
    else:
        mail_cfg = ''

    if args.quotatype is not None:
        quota_cfg = f'#SBATCH --quotatype {args.quotatype}\n'
    else:
        quota_cfg = ''

    launcher = 'none' if args.local else 'slurm'
    runner = 'python' if args.local else 'srun python'

    job_script = (f'#!/bin/bash\n'
                  f'#SBATCH --output {work_dir}/job.%j.out\n'
                  f'#SBATCH --partition={args.partition}\n'
                  f'#SBATCH --job-name {job_name}\n'
                  f'#SBATCH --gres=gpu:8\n'
                  f'{mail_cfg}{quota_cfg}'
                  f'#SBATCH --ntasks-per-node=8\n'
                  f'#SBATCH --ntasks=8\n'
                  f'#SBATCH --cpus-per-task=5\n\n'
                  f'{runner} -u {script_name} {config} {checkpoint} '
                  f'--out={work_dir / "result.pkl"} --metrics accuracy '
                  f'--out-items=none '
                  f'--cfg-option dist_params.port={port} '
                  f'--launcher={launcher}\n')

    with open(work_dir / 'job.sh', 'w') as f:
        f.write(job_script)

    commands.append(f'echo "{config}"')
    if args.local:
        commands.append(f'bash {work_dir}/job.sh')
    else:
        commands.append(f'sbatch {work_dir}/job.sh')

    return work_dir / 'job.sh'


def test(args):
    # parse model-index.yml
    model_index_file = MMCLS_ROOT / 'model-index.yml'
    model_index = load(str(model_index_file))
    model_index.build_models_with_collections()
    models = OrderedDict({model.name: model for model in model_index.models})

    script_name = osp.join('tools', 'test.py')
    port = args.port

    commands = []
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

    preview_script = ''
    for model_info in models.values():

        if model_info.results is None:
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


def save_summary(summary_data, models_map, work_dir):
    summary_path = work_dir / 'test_benchmark_summary.md'
    file = open(summary_path, 'w')
    headers = [
        'Model', 'Top-1 Expected(%)', 'Top-1 (%)', 'Top-5 Expected (%)',
        'Top-5 (%)', 'Config'
    ]
    file.write('# Test Benchmark Regression Summary\n')
    file.write('| ' + ' | '.join(headers) + ' |\n')
    file.write('|:' + ':|:'.join(['---'] * len(headers)) + ':|\n')
    for model_name, summary in summary_data.items():
        if len(summary) == 0:
            # Skip models without results
            continue
        row = [model_name]
        if 'Top 1 Accuracy' in summary:
            metric = summary['Top 1 Accuracy']
            row.append(f"{metric['expect']:.2f}")
            row.append(f"{metric['result']:.2f}")
        else:
            row.extend([''] * 2)
        if 'Top 5 Accuracy' in summary:
            metric = summary['Top 5 Accuracy']
            row.append(f"{metric['expect']:.2f}")
            row.append(f"{metric['result']:.2f}")
        else:
            row.extend([''] * 2)

        model_info = models_map[model_name]
        row.append(model_info.config)
        file.write('| ' + ' | '.join(row) + ' |\n')
    file.close()
    print('Summary file saved at ' + str(summary_path))


def show_summary(summary_data):
    table = Table(title='Test Benchmark Regression Summary')
    table.add_column('Model')
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
                expect = metric['expect']
                result = metric['result']
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

    console.print(table)


def summary(args):
    model_index_file = MMCLS_ROOT / 'model-index.yml'
    model_index = load(str(model_index_file))
    model_index.build_models_with_collections()
    models = OrderedDict({model.name: model for model in model_index.models})

    work_dir = Path(args.work_dir)

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
    for model_name, model_info in models.items():

        if model_info.results is None:
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
            if key_yml in expect_metrics:
                assert key_res in results, \
                    f'{model_name}: No metric "{key_res}"'
                expect_result = float(expect_metrics[key_yml])
                result = float(results[key_res])
                summary[key_yml] = dict(expect=expect_result, result=result)

        summary_data[model_name] = summary

    show_summary(summary_data)
    if args.save:
        save_summary(summary_data, models, work_dir)


def main():
    args = parse_args()

    if args.summary:
        summary(args)
    else:
        test(args)


if __name__ == '__main__':
    main()
