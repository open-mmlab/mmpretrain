import argparse
import pickle

import numpy as np

CLASSES = [f'{i:0>4d}' for i in range(5000)]


def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble Learning')
    parser.add_argument('pkl', help='Ensemble list results')
    parser.add_argument('--thr', default=0, type=float, help='threshold')
    parser.add_argument('--out', default='pseudo.txt', help='output path')
    parser.add_argument('--testb', action='store_true', help='testa or testsb')
    args = parser.parse_args()
    return args


def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data


def generate_pseudo_label(data, thr):
    pseudo_list = []
    for filename, classname, scores in data:
        pred_score = np.max(scores)
        if pred_score > thr:
            pseudo_list.append((filename, classname))
    return pseudo_list


def main():
    args = parse_args()
    data = load_pkl(args.pkl)
    print(f'{len(data)} samples have been found....')

    pseudo_list = generate_pseudo_label(data, args.thr)
    print(f'Get {len(pseudo_list)} pseudo samples....')

    assert args.out and args.out.endswith('.txt')
    with open(args.out, 'w') as outfile:
        for filename, label in pseudo_list:
            test = 'testb' if args.testb else 'testa'
            outfile.write(f'{test}/{filename} {label}\n')


if __name__ == '__main__':
    main()
