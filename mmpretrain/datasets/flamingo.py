# Copyright (c) OpenMMLab. All rights reserved.
import copy
import random
from abc import abstractmethod
from collections import Counter
from typing import List

import mmengine
import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend

from mmpretrain.registry import DATASETS


class FlamingoFewShotMixin:
    """Flamingo fewshot eval dataset minin.

    Args:
        num_shots (int): Number of shots to perform evaluation.
            Defaults to 0.
            Note: 0 does not mean a strict zero-shot in Flamingo setting.
            It will use 2 only-text prompt without in context images.
        num_support_examples (int): Number of support examples to get the
            few shots from. Defaults to 2048.
        num_query_examples (int): Number of query examples to perform the
            final evaluation. Defaults to 5000.
        incontext_prompt_temp (str): In context prompt template for few shot
            examples. Defaults to ''.
        final_prompt_temp (str): Final query prompt template. Defaults to ''.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self,
                 num_shots: int = 0,
                 num_support_examples: int = 2048,
                 num_query_examples: int = 5000,
                 incontext_prompt_temp: str = '',
                 final_prompt_temp: str = '',
                 **kwarg):
        self.num_shots = num_shots
        self.num_support_examples = num_support_examples
        self.num_query_examples = num_query_examples
        self.num_effective_examples = num_shots if num_shots > 0 else 2
        self.incontext_prompt_temp = incontext_prompt_temp
        self.final_prompt_temp = final_prompt_temp
        super().__init__(**kwarg)

    def get_subset_idx(self, total_num):
        random_idx = np.random.choice(
            total_num,
            self.num_support_examples + self.num_query_examples,
            replace=False)

        support_idx = random_idx[:self.num_support_examples]
        query_idx = random_idx[self.num_support_examples:]
        return support_idx, query_idx

    @abstractmethod
    def parse_basic_anno(self, anno: dict) -> dict:
        """Parse basic annotation for support and query set."""
        pass

    @abstractmethod
    def parse_fewshot_anno(self, anno: dict, support_list: List) -> dict:
        """Parse fewshot related annotation for query set with support list."""
        pass


@DATASETS.register_module()
class FlamingoEvalCOCOVQA(FlamingoFewShotMixin, BaseDataset):
    """Flamingo few shot VQAv2 dataset.

    Args:
        num_shots (int): Number of shots to perform evaluation.
            Defaults to 0.
            Note: 0 does not mean a strict zero-shot in Flamingo setting.
            It will use 2 only-text prompt without in context images.
        num_support_examples (int): Number of support examples to get the
            few shots from. Defaults to 2048.
        num_query_examples (int): Number of query examples to perform the
            final evaluation. Defaults to 5000.
        incontext_prompt_temp (str): In context prompt template for few shot
            examples.
            Defaults to '<image>Question:{} Short Answer:{}<|endofchunk|>'.
        final_prompt_temp (str): Final query prompt template.
            Defaults to '<image>Question:{} Short Answer:'.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self,
                 question_file,
                 num_shots: int = 0,
                 num_support_examples: int = 2048,
                 num_query_examples: int = 5000,
                 incontext_prompt_temp:
                 str = '<image>Question:{} Short Answer:{}<|endofchunk|>',
                 final_prompt_temp: str = '<image>Question:{} Short Answer:',
                 **kwarg):
        self.question_file = question_file
        super().__init__(
            num_shots=num_shots,
            num_support_examples=num_support_examples,
            num_query_examples=num_query_examples,
            incontext_prompt_temp=incontext_prompt_temp,
            final_prompt_temp=final_prompt_temp,
            **kwarg)

    def parse_basic_anno(self, anno: dict) -> dict:
        """Parse basic annotation for support and query set.

        Args:
            anno (dict): Annotation for single example.

        Return:
            dict: Parsed annotation for single example.
        """
        new_anno = {}
        img_prefix = self.data_prefix['img_path']
        new_anno[
            'img_path'] = f"{img_prefix}/train2014/COCO_train2014_{anno['image_id']:012d}.jpg"  # noqa
        answer = [a['answer'] for a in anno['answers']]
        count = Counter(answer)
        new_anno['gt_answer'] = list(count.keys())
        new_anno['gt_answer_weight'] = anno.get(
            'answer_weight',
            [i / len(answer) for i in count.values()],
        )
        return new_anno

    def parse_fewshot_anno(self, anno: dict, support_list: List) -> dict:
        """Parse fewshot related annotation for query set with support list.

        Args:
            anno (dict): Annotation for single example.
            support_list (List): List of support subset to subsample few shots.

        Return:
            dict: Parsed annotation for single example.
        """
        # prepare n shots examples
        anno_support = random.sample(support_list, self.num_effective_examples)

        # append image path for n shots
        anno['img_path'] = [shot['img_path']
                            for shot in anno_support] + [anno['img_path']]
        anno['nshot_prompt'] = ''.join([
            self.incontext_prompt_temp.format(shot['question'],
                                              shot['gt_answer'][0])
            for shot in anno_support
        ])

        # remove image related shots when perform zero-shot
        if self.num_shots == 0:
            # get the final image path when zero-shot
            anno['img_path'] = anno['img_path'][-1:]
            anno['nshot_prompt'] = anno['nshot_prompt'].replace('<image>', '')

        # add final prompt
        anno['nshot_prompt'] += self.final_prompt_temp.format(anno['question'])
        return anno

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        img_prefix = self.data_prefix['img_path']
        file_backend = get_file_backend(img_prefix)
        annotations = mmengine.load(self.ann_file)['annotations']
        # additional file path should be joined manually
        questions = mmengine.load(
            file_backend.join_path(img_prefix,
                                   self.question_file))['questions']

        assert len(questions) == len(annotations)
        num_data = len(annotations)
        support_idx, query_idx = self.get_subset_idx(num_data)

        # prepare support subset
        support_list = []
        for idx in support_idx:
            question = questions[idx]
            answers = annotations[idx]
            anno = copy.deepcopy(question)
            anno = {**anno, **self.parse_basic_anno(answers)}
            support_list.append(anno)

        # prepare query subset
        query_list = []
        for idx in query_idx:
            question = questions[idx]
            answers = annotations[idx]
            anno = copy.deepcopy(question)
            anno = {**anno, **self.parse_basic_anno(answers)}
            anno = self.parse_fewshot_anno(anno, support_list)
            query_list.append(anno)

        return query_list


@DATASETS.register_module()
class FlamingoEvalCOCOCaption(FlamingoFewShotMixin, BaseDataset):
    """Flamingo few shot COCO Caption dataset.

    Args:
        num_shots (int): Number of shots to perform evaluation.
            Defaults to 0.
            Note: 0 does not mean a strict zero-shot in Flamingo setting.
            It will use 2 only-text prompt without in context images.
        num_support_examples (int): Number of support examples to get the
            few shots from. Defaults to 2048.
        num_query_examples (int): Number of query examples to perform the
            final evaluation. Defaults to 5000.
        incontext_prompt_temp (str): In context prompt template for few shot
            examples. Defaults to '<image>Output:{}<|endofchunk|>'.
        final_prompt_temp (str): Final query prompt template.
            Defaults to '<image>Output:'.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self,
                 num_shots: int = 0,
                 num_support_examples: int = 2048,
                 num_query_examples: int = 5000,
                 incontext_prompt_temp: str = '<image>Output:{}<|endofchunk|>',
                 final_prompt_temp: str = '<image>Output:',
                 **kwarg):
        super().__init__(
            num_shots=num_shots,
            num_support_examples=num_support_examples,
            num_query_examples=num_query_examples,
            incontext_prompt_temp=incontext_prompt_temp,
            final_prompt_temp=final_prompt_temp,
            **kwarg)

    def parse_basic_anno(self, anno: dict) -> dict:
        """Parse basic annotation for support and query set.

        Args:
            anno (dict): Annotation for single example.

        Return:
            dict: Parsed annotation for single example.
        """
        img_prefix = self.data_prefix['img_path']
        anno[
            'img_path'] = f"{img_prefix}/train2014/COCO_train2014_{anno['image_id']:012d}.jpg"  # noqa
        anno['text'] = anno.pop('caption')
        return anno

    def parse_fewshot_anno(self, anno: dict, support_list: List) -> dict:
        """Parse fewshot related annotation for query set with support list.

        Args:
            anno (dict): Annotation for single example.
            support_list (List): List of support subset to subsample few shots.

        Return:
            dict: Parsed annotation for single example.
        """
        # prepare n shots examples
        anno_support = random.sample(support_list, self.num_effective_examples)

        # append image path for n shots
        anno['img_path'] = [shot['img_path']
                            for shot in anno_support] + [anno['img_path']]
        anno['nshot_prompt'] = ''.join([
            self.incontext_prompt_temp.format(shot['text'])
            for shot in anno_support
        ])

        # remove image related shots when perform zero-shot
        if self.num_shots == 0:
            # get the final image path when zero-shot
            anno['img_path'] = anno['img_path'][-1:]
            anno['nshot_prompt'] = anno['nshot_prompt'].replace('<image>', '')

        # add final prompt
        anno['nshot_prompt'] += self.final_prompt_temp
        return anno

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        annotations = mmengine.load(self.ann_file)['annotations']

        num_data = len(annotations)
        support_idx, query_idx = self.get_subset_idx(num_data)

        # prepare support subset
        support_list = []
        for idx in support_idx:
            anno = annotations[idx]
            anno = self.parse_basic_anno(anno)
            support_list.append(anno)

        # prepare query subset
        query_list = []
        for idx in query_idx:
            anno = annotations[idx]
            anno = self.parse_basic_anno(anno)
            anno = self.parse_fewshot_anno(anno, support_list)
            query_list.append(anno)

        return query_list
