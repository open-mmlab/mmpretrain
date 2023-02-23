# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpretrain.structures import (batch_label_to_onehot, cat_batch_labels,
                                   tensor_split)


class TestStructureUtils(TestCase):

    def test_tensor_split(self):
        tensor = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        split_indices = [0, 2, 6, 6]
        outs = tensor_split(tensor, split_indices)
        self.assertEqual(len(outs), len(split_indices) + 1)
        self.assertEqual(outs[0].size(0), 0)
        self.assertEqual(outs[1].size(0), 2)
        self.assertEqual(outs[2].size(0), 4)
        self.assertEqual(outs[3].size(0), 0)
        self.assertEqual(outs[4].size(0), 1)

        tensor = torch.tensor([])
        split_indices = [0, 0, 0, 0]
        outs = tensor_split(tensor, split_indices)
        self.assertEqual(len(outs), len(split_indices) + 1)

    def test_cat_batch_labels(self):
        labels = [
            torch.tensor([1]),
            torch.tensor([3, 2]),
            torch.tensor([0, 1, 4]),
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
        ]

        batch_label, split_indices = cat_batch_labels(labels)
        self.assertEqual(split_indices, [1, 3, 6, 6])
        self.assertEqual(len(batch_label), 6)
        labels = tensor_split(batch_label, split_indices)
        self.assertEqual(labels[0].tolist(), [1])
        self.assertEqual(labels[1].tolist(), [3, 2])
        self.assertEqual(labels[2].tolist(), [0, 1, 4])
        self.assertEqual(labels[3].tolist(), [])
        self.assertEqual(labels[4].tolist(), [])

    def test_batch_label_to_onehot(self):
        labels = [
            torch.tensor([1]),
            torch.tensor([3, 2]),
            torch.tensor([0, 1, 4]),
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
        ]

        batch_label, split_indices = cat_batch_labels(labels)
        batch_score = batch_label_to_onehot(
            batch_label, split_indices, num_classes=5)
        self.assertEqual(batch_score[0].tolist(), [0, 1, 0, 0, 0])
        self.assertEqual(batch_score[1].tolist(), [0, 0, 1, 1, 0])
        self.assertEqual(batch_score[2].tolist(), [1, 1, 0, 0, 1])
        self.assertEqual(batch_score[3].tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(batch_score[4].tolist(), [0, 0, 0, 0, 0])
