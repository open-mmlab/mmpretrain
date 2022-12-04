import argparse
import csv
import pickle
from collections import defaultdict

import numpy as np

CLASSES = [f'{i:0>4d}' for i in range(5000)]


def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble Learning')
    parser.add_argument('pkl', help='Ensemble results')
    parser.add_argument('--K', type=int, help='Ensemble results')
    parser.add_argument(
        '--out', default='pred_results.csv', help='output path')
    args = parser.parse_args()
    return args


def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data


def post_process(data_dict):
    result_list = []
    for filename, scores in data_dict.items():
        pred_label = np.argmax(scores)
        pred_class = CLASSES[pred_label]
        result_list.append([filename, pred_class, scores])
    return result_list


def plot_labels2(data_list, K):
    data_dict = defaultdict(list)
    for i, (filename, classname, score) in enumerate(data_list):
        data_dict[classname].append(i)

    max_counts = 0
    for classname in CLASSES:
        max_counts = max(max_counts, len(data_dict[classname]))

    counts = list(range(max_counts + 1))
    less_count_classes = defaultdict(list)
    count_dict = defaultdict(int)
    for i in counts:
        count_dict[i] = 0
    for classname in CLASSES:
        count_dict[len(data_dict[classname])] += 1
        if len(data_dict[classname]) < K:
            less_count_classes[len(data_dict[classname])].append(classname)

    numbers = list(count_dict.values())
    import matplotlib.pyplot as plt

    plt.bar(counts, numbers)
    plt.savefig('target_label_after.jpg')
    plt.show()


def plot_labels(data, K):
    data_list = post_process(data)

    print(f'{len(data_list)} samples have been found....')

    data_dict = defaultdict(list)
    for i, (filename, classname, score) in enumerate(data_list):
        data_dict[classname].append(i)

    max_counts = 0
    for classname in CLASSES:
        max_counts = max(max_counts, len(data_dict[classname]))

    counts = list(range(max_counts + 1))
    less_count_classes = defaultdict(list)
    count_dict = defaultdict(int)
    for i in counts:
        count_dict[i] = 0
    for classname in CLASSES:
        count_dict[len(data_dict[classname])] += 1
        if len(data_dict[classname]) < K:
            less_count_classes[len(data_dict[classname])].append(classname)

    numbers = list(count_dict.values())
    import matplotlib.pyplot as plt

    plt.bar(counts, numbers)
    plt.savefig('target_label_before.jpg')
    plt.show()

    return data_list, less_count_classes


def main():
    args = parse_args()
    K = args.K

    data_dict = load_pkl(args.pkl)
    result_list, less_count_classes = plot_labels(data_dict, K)
    pred_labels = np.array([int(r[1]) for r in result_list])
    print(pred_labels.shape)

    all_soreces = np.stack([r[2] for r in result_list], axis=0)

    for count, classname_list in less_count_classes.items():
        print(count)
        for classname in classname_list:
            class_idx = int(classname)
            soreces = all_soreces[:, class_idx]
            soreces[pred_labels == class_idx] = 0
            topk = K - count
            indxs = np.argpartition(soreces, -topk)[-topk:]
            for ind in indxs:
                result_list[int(ind)][1] = classname

    assert args.out and args.out.endswith('.csv')

    plot_labels2(result_list, K)

    with open(args.out, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for result in result_list:
            writer.writerow(result[:2])


main()
