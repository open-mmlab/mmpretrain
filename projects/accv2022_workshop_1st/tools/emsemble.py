import argparse
import csv
import os
import pickle

import numpy as np

CLASSES = [f'{i:0>4d}' for i in range(5000)]


def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble Learning')
    parser.add_argument(
        '--pkls', nargs='+', default=[], help='Ensemble results')
    parser.add_argument('--pkls-dir', default=None, help='Ensemble results')
    parser.add_argument(
        '--nopkls', nargs='+', default=[], help='Ensemble results')
    parser.add_argument(
        '--out', default='pred_results.csv', help='output path')
    parser.add_argument('--dump', default=None, help='dump to list results.')
    parser.add_argument(
        '--dump-all', default=None, help='dump to all dict results.')
    parser.add_argument(
        '--factor', default=1.0, type=float, help='The factor to scale acc')
    parser.add_argument('--scale', action='store_true')
    args = parser.parse_args()
    assert len(args.pkls) != 0 or args.pkls_dir is not None
    return args


def loda_pkl(pkl_path, data_dict, num_models, acc=1.0):
    print(f'Process {pkl_path}.......')
    with open(pkl_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    for item in data:
        if item['filename'] in data_dict:
            data_dict[item['filename']] += item['scores'] / num_models * acc
        else:
            data_dict[item['filename']] = item['scores'] / num_models * acc
    return data


def post_process(data_dict):
    result_list = []
    for filename, scores in data_dict.items():
        pred_label = np.argmax(scores)
        pred_class = CLASSES[pred_label]
        result_list.append((filename, pred_class, scores))
    return result_list


def main():
    args = parse_args()
    pkls = []
    if len(args.pkls) > 0:
        pkls += args.pkls
    if args.pkls_dir is not None:
        pkls += [
            os.path.join(args.pkls_dir, p) for p in os.listdir(args.pkls_dir)
            if p.endswith('.pkl')
        ]
    if len(args.nopkls) > 0:
        for nopkl in args.nopkls:
            if nopkl.endswith('.pkl'):
                pkls.remove(nopkl)

    num_models = len(pkls)
    print(f'Number of .pkls is {num_models}....')
    assert num_models > 0

    data_dict = dict()
    accs = [float(pkl.split('.')[0].split('-')[-1]) for pkl in pkls]
    if args.scale:
        min_adjust_accs = [pow(acc / min(accs), args.factor) for acc in accs]
    else:
        min_adjust_accs = [1 for _ in accs]

    print(f'Adjusted factor is: {min_adjust_accs}')
    for pkl, acc in zip(pkls, min_adjust_accs):
        loda_pkl(pkl, data_dict, num_models, acc)

    result_list = post_process(data_dict)

    if args.dump:
        assert args.dump.endswith('.pkl')
        with open(args.dump, 'wb') as dumpfile:
            import pickle
            pickle.dump(result_list, dumpfile)

    if args.dump_all:
        assert args.dump_all.endswith('.pkl')
        with open(args.dump_all, 'wb') as dumpfile:
            import pickle
            pickle.dump(data_dict, dumpfile)

    assert args.out and args.out.endswith('.csv')
    with open(args.out, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for result in result_list:
            writer.writerow(result[:2])


if __name__ == '__main__':
    main()
