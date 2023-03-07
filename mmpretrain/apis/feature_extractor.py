# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from mmcv.image import imread
from mmengine.config import Config
from mmengine.dataset import Compose, default_collate
from mmengine.device import get_device
from mmengine.infer import BaseInferencer
from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint

from mmpretrain.registry import TRANSFORMS
from .model import get_model, list_models

ModelType = Union[BaseModel, str, Config]
InputType = Union[str, np.ndarray, list]


class FeatureExtractor(BaseInferencer):
    """The inferencer for extract features.

    Args:
        model (BaseModel | str | Config): A model name or a path to the confi
            file, or a :obj:`BaseModel` object. The model name can be found
            by ``FeatureExtractor.list_models()``.
        pretrained (bool | str): When use name to specify model, you can
            use ``True`` to load the pre-defined pretrained weights. And you
            can also use a string to specify the path or link of weights to
            load. Defaults to True.
        device (str, optional): Device to run inference. If None, use CPU or
            the device of the input model. Defaults to None.
    """

    def __init__(
        self,
        model: ModelType,
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

    def __call__(self,
                 inputs: InputType,
                 batch_size: int = 1,
                 **kwargs) -> dict:
        """Call the inferencer.

        Args:
            inputs (str | array | list): The image path or array, or a list of
                images.
            batch_size (int): Batch size. Defaults to 1.
            **kwargs: Other keyword arguments accepted by the `extract_feat`
                method of the model.

        Returns:
            tensor | Tuple[tensor]: The extracted features.
        """
        ori_inputs = self._inputs_to_list(inputs)
        inputs = self.preprocess(ori_inputs, batch_size=batch_size)
        preds = []
        for data in inputs:
            preds.extend(self.forward(data, **kwargs))

        return preds

    @torch.no_grad()
    def forward(self, inputs: Union[dict, tuple], **kwargs):
        inputs = self.model.data_preprocessor(inputs, False)['inputs']
        outputs = self.model.extract_feat(inputs, **kwargs)

        def scatter(feats, index):
            if isinstance(feats, torch.Tensor):
                return feats[index]
            else:
                # Sequence of tensor
                return type(feats)([scatter(item, index) for item in feats])

        results = []
        for i in range(inputs.shape[0]):
            results.append(scatter(outputs, i))

        return results

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

    def visualize(self):
        raise NotImplementedError(
            "The FeatureExtractor doesn't support visualization.")

    def postprocess(self):
        raise NotImplementedError(
            "The FeatureExtractor doesn't need postprocessing.")

    @staticmethod
    def list_models(pattern: Optional[str] = None):
        """List all available model names.

        Args:
            pattern (str | None): A wildcard pattern to match model names.

        Returns:
            List[str]: a list of model names.
        """
        return list_models(pattern=pattern)
