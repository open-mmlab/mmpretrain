# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from mmcls.utils import load_json_log

TEST_METRICS = ('precision', 'recall', 'f1_score', 'support', 'mAP', 'CP',
                'CR', 'CF1', 'OP', 'OR', 'OF1', 'accuracy')


def cal_train_time(log_dicts, args):
    """Compute the average time per training iteration."""
    for i, log_dict in enumerate(log_dicts):
        print(f'{"-" * 5}Analyze train time of {args.json_logs[i]}{"-" * 5}')
        all_times = []
        for epoch in log_dict.keys():
            if args.include_outliers:
                all_times.append(log_dict[epoch]['time'])
            else:
                all_times.append(log_dict[epoch]['time'][1:])
        all_times = np.array(all_times)
        epoch_ave_time = all_times.mean(-1)
        slowest_epoch = epoch_ave_time.argmax()
        fastest_epoch = epoch_ave_time.argmin()
        std_over_epoch = epoch_ave_time.std()
        print(f'slowest epoch {slowest_epoch + 1}, '
              f'average time is {epoch_ave_time[slowest_epoch]:.4f}')
        print(f'fastest epoch {fastest_epoch + 1}, '
              f'average time is {epoch_ave_time[fastest_epoch]:.4f}')
        print(f'time std over epochs is {std_over_epoch:.4f}')
        print(f'average iter time: {np.mean(all_times):.4f} s/iter')
        print()


def get_legends(args):
    """if legend is None, use {filename}_{key} as legend."""
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                # remove '.json' in the end of log names
                basename = os.path.basename(json_log)[:-5]
                if basename.endswith('.log'):
                    basename = basename[:-4]
                legend.append(f'{basename}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    return legend


def plot_phase_train(metric, log_dict, epochs, curve_label, json_log):
    """plot phase of train cruve."""
    if metric not in log_dict[epochs[0]]:
        raise KeyError(f'{json_log} does not contain metric {metric}'
                       f' in train mode')
    xs, ys = [], []
    for epoch in epochs:
        iters = log_dict[epoch]['iter']
        if log_dict[epoch]['mode'][-1] == 'val':
            iters = iters[:-1]
        num_iters_per_epoch = iters[-1]
        assert len(iters) > 0, (
            'The training log is empty, please try to reduce the '
            'interval of log in config file.')
        xs.append(np.array(iters) / num_iters_per_epoch + (epoch - 1))
        ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    plt.xlabel('Epochs')
    plt.plot(xs, ys, label=curve_label, linewidth=0.75)


def plot_phase_val(metric, log_dict, epochs, curve_label, json_log):
    """plot phase of val cruves."""
    # some epoch may not have evaluation. as [(train, 5),(val, 1)]
    xs = [e for e in epochs if metric in log_dict[e]]
    ys = [log_dict[e][metric] for e in xs if metric in log_dict[e]]
    assert len(xs) > 0, (f'{json_log} does not contain metric {metric}')
    plt.xlabel('Epochs')
    plt.plot(xs, ys, label=curve_label, linewidth=0.75)


def plot_curve_helper(log_dicts, metrics, args, legend):
    """plot curves from log_dicts by metrics."""
    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            json_log = args.json_logs[i]
            print(f'plot curve of {json_log}, metric is {metric}')
            curve_label = legend[i * num_metrics + j]
            if any(m in metric for m in TEST_METRICS):
                plot_phase_val(metric, log_dict, epochs, curve_label, json_log)
            else:
                plot_phase_train(metric, log_dict, epochs, curve_label,
                                 json_log)
            plt.legend()


def plot_curve(log_dicts, args):
    """Plot train metric-iter graph."""
    # set backend and style
    if args.backend is not None:
        plt.switch_backend(args.backend)
    try:
        import seaborn as sns
        sns.set_style(args.style)
    except ImportError:
        print("Attention: The plot style won't be applied because 'seaborn' "
              'package is not installed, please install it if you want better '
              'show style.')

    # set plot window size
    wind_w, wind_h = args.window_size.split('*')
    wind_w, wind_h = int(wind_w), int(wind_h)
    plt.figure(figsize=(wind_w, wind_h))

    # get legends and metrics
    legends = get_legends(args)
    metrics = args.keys

    # plot curves from log_dicts by metrics
    plot_curve_helper(log_dicts, metrics, args, legends)

    # set title and show or save
    if args.title is not None:
        plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser(
        'plot_curve', help='parser for plotting curves')
    parser_plt.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_plt.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['loss'],
        help='the metric that you want to plot')
    parser_plt.add_argument('--title', type=str, help='title of figure')
    parser_plt.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser_plt.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser_plt.add_argument(
        '--style', type=str, default='whitegrid', help='style of plt')
    parser_plt.add_argument('--out', type=str, default=None)
    parser_plt.add_argument(
        '--window-size',
        default='12*7',
        help='size of the window to display images, in format of "$W*$H".')


def add_time_parser(subparsers):
    parser_time = subparsers.add_parser(
        'cal_train_time',
        help='parser for computing the average time per training iteration')
    parser_time.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_time.add_argument(
        '--include-outliers',
        action='store_true',
        help='include the first value of every epoch when computing '
        'the average time')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    subparsers = parser.add_subparsers(dest='task', help='task parser')
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    args = parser.parse_args()

    if hasattr(args, 'window_size') and args.window_size != '':
        assert re.match(r'\d+\*\d+', args.window_size), \
            "'window-size' must be in format 'W*H'."
    return args


def main():
    args = parse_args()

    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')

    log_dicts = [load_json_log(json_log) for json_log in json_logs]

    eval(args.task)(log_dicts, args)


if __name__ == '__main__':
    main()
