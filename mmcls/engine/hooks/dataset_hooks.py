# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

from mmengine import is_seq_of
from mmengine.hooks import Hook
from mmengine.logging import MessageHub
from mmengine.registry import HOOKS
from mmengine.runner import BaseLoop

from mmcls.registry import DATASETS


@HOOKS.register_module()
class SendDatasetInfoMessageHubHook(Hook):
    """A Hook to send information to the MessageHub.

    Args:
        keys (str | List[str]): the key of information to send
            to the MessageHub.
    """
    priority = 'NORMAL'
    accept_keys = {'gt_labels', 'file_paths'}

    def __init__(
        self,
        keys: Union[str, List[str]],
    ):
        if isinstance(keys, str):
            self.keys = [keys]
        elif is_seq_of(keys, str):
            self.keys = keys
        else:
            raise ValueError(
                f'`keys` must be str of seq of str, but get {type(keys)}')

        for key in self.keys():
            assert key in self.accept_keys, (
                f"key must be in {self.accept_keys}, but get '{key}'.")

    def before_run(self, runner) -> None:
        """Create an ema copy of the model.

        Args:
            runner (Runner): The runner of the training process.
        """
        if isinstance(runner._train_loop, BaseLoop):
            train_dataset = runner.train_loop.dataloader.dataset
        elif isinstance(runner._train_loop, dict):
            assert isinstance(runner._train_dataloader, dict) and \
                    'dataset' in runner._train_dataloader, (
                        'Please set `train_dataloader.dataset` in config.')
            train_dataset_cfg = runner._train_dataloader['dataset']
            train_dataset = DATASETS.build(train_dataset_cfg)
        else:
            raise ValueError('Error type of `train_cfg` attr in config.')

        for key in self.keys:
            if key == 'gt_labels':
                gt_labels = train_dataset.gt_labels()
                MessageHub.get_current_instance().update_info(
                    'gt_labels', gt_labels)
            elif key == 'file_paths':
                file_paths = [
                    data['file_path'] for data in train_dataset.data_list
                ]
                MessageHub.get_current_instance().update_info(
                    'file_paths', file_paths)
