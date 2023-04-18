# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmengine.model import is_model_wrapper
from mmengine.runner import TestLoop, ValLoop, autocast

from mmpretrain.registry import LOOPS


@LOOPS.register_module()
class RetrievalValLoop(ValLoop):
    """Loop for multimodal retrieval val.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 valing. Defaults to
            False.
    """

    def run(self) -> dict:
        """Launch val."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        texts_local = []
        feats_local = []
        data_samples_local = []

        for idx, data_batch in enumerate(self.dataloader):
            with torch.no_grad():
                self.runner.call_hook(
                    'before_val_iter', batch_idx=idx, data_batch=data_batch)
                # predictions should be sequence of BaseDataElement
                with autocast(enabled=self.fp16):
                    if is_model_wrapper(self.runner.model):
                        data_preprocessor = self.runner.model.module.data_preprocessor  # noqa: E501
                    else:
                        data_preprocessor = self.runner.model.data_preprocessor

                    # get features for retrieval instead of data samples
                    data_batch = data_preprocessor(data_batch, False)
                    text_inputs, feats = self.runner.model._run_forward(
                        data_batch, mode='tensor')
                    texts_local.append(text_inputs)
                    feats_local.append(feats)
                    data_samples_local.extend(data_batch['data_samples'])
                self.runner.call_hook(
                    'after_val_iter',
                    batch_idx=idx,
                    data_batch=data_batch,
                    outputs=feats)

        # concatenate different features
        feats_local = [
            torch.cat(list(map(lambda x: x[i], feats_local)))
            for i in range(len(feats_local[0]))
        ]

        # get predictions
        if is_model_wrapper(self.runner.model):
            predict_all_func = self.runner.model.module.predict_all
        else:
            predict_all_func = self.runner.model.predict_all

        img_size = self.dataloader.dataset.img_size
        text_size = self.dataloader.dataset.text_size
        with torch.no_grad():
            i2t_data_samples, t2i_data_samples = predict_all_func(
                texts_local, feats_local, img_size, text_size,
                data_samples_local)

        # process in evaluator and compute metrics
        self.evaluator.process(i2t_data_samples, None)
        img_metrics = self.evaluator.evaluate(img_size)
        self.evaluator.process(t2i_data_samples, None)
        text_metrics = self.evaluator.evaluate(text_size)
        metrics = {'i2t': img_metrics, 't2i': text_metrics}

        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics


@LOOPS.register_module()
class RetrievalTestLoop(TestLoop):
    """Loop for multimodal retrieval test.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 testing. Defaults to
            False.
    """

    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        texts_local = []
        feats_local = []
        data_samples_local = []

        for idx, data_batch in enumerate(self.dataloader):
            with torch.no_grad():
                self.runner.call_hook(
                    'before_test_iter', batch_idx=idx, data_batch=data_batch)
                # predictions should be sequence of BaseDataElement
                with autocast(enabled=self.fp16):
                    if is_model_wrapper(self.runner.model):
                        data_preprocessor = self.runner.model.module.data_preprocessor  # noqa: E501
                    else:
                        data_preprocessor = self.runner.model.data_preprocessor

                    # get features for retrieval instead of data samples
                    data_batch = data_preprocessor(data_batch, False)
                    text_inputs, feats = self.runner.model._run_forward(
                        data_batch, mode='tensor')
                    texts_local.append(text_inputs)
                    feats_local.append(feats)
                    data_samples_local.extend(data_batch['data_samples'])
                self.runner.call_hook(
                    'after_test_iter',
                    batch_idx=idx,
                    data_batch=data_batch,
                    outputs=feats)

        # concatenate different features
        feats_local = [
            torch.cat(list(map(lambda x: x[i], feats_local)))
            for i in range(len(feats_local[0]))
        ]

        # get predictions
        if is_model_wrapper(self.runner.model):
            predict_all_func = self.runner.model.module.predict_all
        else:
            predict_all_func = self.runner.model.predict_all

        img_size = self.dataloader.dataset.img_size
        text_size = self.dataloader.dataset.text_size
        with torch.no_grad():
            i2t_data_samples, t2i_data_samples = predict_all_func(
                texts_local, feats_local, img_size, text_size,
                data_samples_local)

        # process in evaluator and compute metrics
        self.evaluator.process(i2t_data_samples, None)
        img_metrics = self.evaluator.evaluate(img_size)
        self.evaluator.process(t2i_data_samples, None)
        text_metrics = self.evaluator.evaluate(text_size)
        metrics = {'i2t': img_metrics, 't2i': text_metrics}

        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics
