# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from mmcv.image import imread
from mmengine.config import Config
from mmengine.dataset import BaseDataset, Compose, default_collate
from mmengine.device import get_device
from mmengine.infer import BaseInferencer
from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint

from mmpretrain.registry import TRANSFORMS
from mmpretrain.structures import DataSample
from .model import get_model, list_models

ModelType = Union[BaseModel, str, Config]
InputType = Union[str, np.ndarray, list]


class ImageRetrievalInferencer(BaseInferencer):
    """The inferencer for image to image retrieval.

    Args:
        model (BaseModel | str | Config): A model name or a path to the confi
            file, or a :obj:`BaseModel` object. The model name can be found
            by ``ImageClassificationInferencer.list_models()`` and you can also
            query it in :doc:`/modelzoo_statistics`.
        weights (str, optional): Path to the checkpoint. If None, it will try
            to find a pre-defined weight from the model you specified
            (only work if the ``model`` is a model name). Defaults to None.
        device (str, optional): Device to run inference. If None, use CPU or
            the device of the input model. Defaults to None.

    Example:
        1. Use a pre-trained model in MMClassification to inference an image.

           >>> from mmpretrain import ImageClassificationInferencer
           >>> inferencer = ImageClassificationInferencer('resnet50_8xb32_in1k')
           >>> inferencer('demo/demo.JPEG')
           [{'pred_score': array([...]),
             'pred_label': 65,
             'pred_score': 0.6649367809295654,
             'pred_class': 'sea snake'}]

        2. Use a config file and checkpoint to inference multiple images on GPU,
           and save the visualization results in a folder.

           >>> from mmpretrain import ImageClassificationInferencer
           >>> inferencer = ImageClassificationInferencer(
                   model='configs/resnet/resnet50_8xb32_in1k.py',
                   weights='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
                   device='cuda')
           >>> inferencer(['demo/dog.jpg', 'demo/bird.JPEG'], show_dir="./visualize/")
    """  # noqa: E501

    visualize_kwargs: set = {
        'rescale_factor', 'draw_score', 'show', 'show_dir'
    }

    def __init__(
        self,
        model: ModelType,
        prototype,
        prototype_vecs=None,
        prepare_batch_size=8,
        pretrained: Union[bool, str] = True,
        device: Union[str, torch.device, None] = None,
    ) -> None:
        device = device or get_device()

        if isinstance(model, BaseModel):
            if isinstance(pretrained, str):
                load_checkpoint(model, pretrained, map_location='cpu')
            model = model.to(device)
        else:
            model = get_model(model, pretrained, device)

        model.eval()

        self.config = model.config
        self.model = model
        self.pipeline = self._init_pipeline(self.config)
        self.collate_fn = default_collate
        self.visualizer = None

        self.prototype_dataset = self._prepare_prototype(
            prototype, prototype_vecs, prepare_batch_size)

    def _prepare_prototype(self, prototype, prototype_vecs=None, batch_size=8):
        from mmengine.dataset import DefaultSampler
        from torch.utils.data import DataLoader

        def build_dataloader(dataset):
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=self.collate_fn,
                sampler=DefaultSampler(dataset, shuffle=False),
                persistent_workers=False,
            )

        test_pipeline = self.config.test_dataloader.dataset.pipeline

        if isinstance(prototype, str):
            # A directory path of images
            from mmpretrain.datasets import CustomDataset
            dataset = CustomDataset(
                data_root=prototype, pipeline=test_pipeline, with_label=False)
            dataloader = build_dataloader(dataset)
        elif isinstance(prototype, dict):
            # A config of dataset
            from mmpretrain.registry import DATASETS
            prototype.setdefault('pipeline', test_pipeline)
            dataset = DATASETS.build(prototype)
            dataloader = build_dataloader(dataset)
        elif isinstance(prototype, DataLoader):
            dataset = prototype.dataset
            dataloader = prototype
        elif isinstance(prototype, BaseDataset):
            dataset = prototype
            dataloader = build_dataloader(dataset)
        else:
            raise TypeError(f'Unsupported prototype type {type(prototype)}.')

        if prototype_vecs is not None and Path(prototype_vecs).exists():
            self.model.prototype = prototype_vecs
        else:
            self.model.prototype = dataloader
        self.model.prepare_prototype()

        from mmengine.logging import MMLogger
        logger = MMLogger.get_current_instance()
        if prototype_vecs is None:
            logger.info('The prototype has been prepared, you can use '
                        '`save_prototype_vecs` to dump it into a pickle '
                        'file for the future usage.')
        elif not Path(prototype_vecs).exists():
            self.save_prototype_vecs(prototype_vecs)
            logger.info(f'The prototype has been saved at {prototype_vecs}.')

        return dataset

    def save_prototype_vecs(self, path):
        self.model.dump_prototype(path)

    def __call__(self,
                 inputs: InputType,
                 return_datasamples: bool = False,
                 batch_size: int = 1,
                 **kwargs) -> dict:
        """Call the inferencer.

        Args:
            inputs (str | array | list): The image path or array, or a list of
                images.
            return_datasamples (bool): Whether to return results as
                :obj:`DataSample`. Defaults to False.
            batch_size (int): Batch size. Defaults to 1.
            rescale_factor (float, optional): Rescale the image by the rescale
                factor for visualization. This is helpful when the image is too
                large or too small for visualization. Defaults to None.
            draw_score (bool): Whether to draw the prediction scores
                of prediction categories. Defaults to True.
            show (bool): Whether to display the visualization result in a
                window. Defaults to False.
            show_dir (str, optional): If not None, save the visualization
                results in the specified directory. Defaults to None.

        Returns:
            list: The inference results.
        """
        return super().__call__(inputs, return_datasamples, batch_size,
                                **kwargs)

    def _init_pipeline(self, cfg: Config) -> Callable:
        test_pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        if test_pipeline_cfg[0]['type'] == 'LoadImageFromFile':
            # Image loading is finished in `self.preprocess`.
            test_pipeline_cfg = test_pipeline_cfg[1:]
        test_pipeline = Compose(
            [TRANSFORMS.build(t) for t in test_pipeline_cfg])
        return test_pipeline

    def preprocess(self, inputs: List[InputType], batch_size: int = 1):

        def load_image(input_):
            img = imread(input_)
            if img is None:
                raise ValueError(f'Failed to read image {input_}.')
            return dict(
                img=img,
                img_shape=img.shape[:2],
                ori_shape=img.shape[:2],
            )

        pipeline = Compose([load_image, self.pipeline])

        chunked_data = self._get_chunk_data(map(pipeline, inputs), batch_size)
        yield from map(self.collate_fn, chunked_data)

    def visualize(self,
                  ori_inputs: List[InputType],
                  preds: List[DataSample],
                  show: bool = False,
                  draw_score=True,
                  show_dir=None):
        if not show and show_dir is None:
            return None

        raise NotImplementedError('Not implemented yet.')

    def postprocess(
        self,
        preds: List[DataSample],
        visualization: List[np.ndarray],
        return_datasamples=False,
        topk=1,
    ) -> dict:
        if return_datasamples:
            return preds

        results = []
        for data_sample in preds:
            match_scores, indices = torch.topk(data_sample.pred_score, k=topk)
            matches = []
            for match_score, sample_idx in zip(match_scores, indices):
                sample = self.prototype_dataset.get_data_info(sample_idx)
                matches.append({
                    'match_score': match_score,
                    'sample_idx': sample_idx,
                    'sample': sample
                })
            results.append(matches)

        return results

    @staticmethod
    def list_models(pattern: Optional[str] = None):
        """List all available model names.

        Args:
            pattern (str | None): A wildcard pattern to match model names.

        Returns:
            List[str]: a list of model names.
        """
        return list_models(pattern=pattern, task='Image Retrieval')
