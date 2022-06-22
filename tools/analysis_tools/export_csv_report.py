#!/usr/bin/env python
# -*- coding=utf-8 -*-

import argparse
import os.path as osp

import mmcv
from mmcv import DictAction
import csv
import numpy as np
import os

from mmcls.core.evaluation import (
    calculate_confusion_matrix,
    precision_recall_f1,
)
from mmcls.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="MMCls export test report")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("result", help="test result json/pkl file")
    parser.add_argument(
        "--out-dir",
        help="dir to store report file, if not specified, will save into same path as test result json/pkl file",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    args = parser.parse_args()

    return args


def convert_float(sub_v):
    return "nan" if (sub_v is None) or sub_v == -1 else "{:.4f}".format(sub_v)


def convert_float_seq(seq):
    return map(convert_float, seq)


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    xls_path = "test-report.csv"
    if args.out_dir is None:
        save_dir = os.path.join(os.path.dirname(args.result), xls_path)
    else:
        save_dir = os.path.join(args.out_dir, xls_path)

    dataset = build_dataset(cfg.data.test)
    # load test results
    result = mmcv.load(args.result)
    pred = result["class_scores"]
    matrix = calculate_confusion_matrix(pred, dataset.get_gt_labels())

    csv_file = open(save_dir, "w")
    csv_writer = csv.writer(csv_file)

    report = dict()
    report["Confusion_Matrix"] = matrix.numpy()

    csv_writer.writerow(["Confusion_Matrix"])
    head = ["Class"]
    head.extend(dataset.CLASSES)
    csv_writer.writerow(head)
    for i in range(len(dataset.CLASSES)):
        row = [dataset.CLASSES[i]]
        row.extend(report["Confusion_Matrix"][i])
        csv_writer.writerow(row)

    csv_writer.writerow([])
    csv_writer.writerow([])

    csv_writer.writerow(["Metrics for Each Class"])
    precisions, recalls, f1_scores = precision_recall_f1(
        pred, dataset.get_gt_labels(), average_mode="none"
    )

    csv_writer.writerow(["Class", "Precision", "Recall", "F1"])
    for i, label in enumerate(dataset.CLASSES):
        csv_writer.writerow(
            [
                label,
                convert_float(precisions[i]),
                convert_float(recalls[i]),
                convert_float(f1_scores[i]),
            ]
        )

    csv_writer.writerow([])
    csv_writer.writerow([])

    for k in [
        "accuracy_top-1",
        "accuracy_top-5",
        "support",
        "precision",
        "recall",
        "f1_score",
    ]:
        if k in result:
            csv_writer.writerow([k, convert_float(result[k])])

    csv_file.close()


if __name__ == "__main__":
    main()
