# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from mmcv.image import imread
from mmengine.config import Config
from mmengine.dataset import Compose, default_collate
from mmengine.infer import BaseInferencer
from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint

from mmcls.registry import TRANSFORMS
from mmcls.structures import ClsDataSample
from .model import get_model, init_model, list_models

ModelType = Union[BaseModel, str, Config]
InputType = Union[str, np.ndarray]


def inference_model(model: ModelType, img: InputType, device=None):
    """Inference an image with the classifier.

    Args:
        model (BaseModel | str | Config): The loaded classifier or the model
            name or the config of the model.
        img (str | ndarray): The image filename or loaded image.
        device (str, optional): Device to run inference. If None, use CPU or
            the device of the input model. Defaults to None.

    Returns:
        result (dict): The classification results that contains:

        - ``pred_scores``: The classification scores of all categories.
        - ``pred_class``: The predicted category.
        - ``pred_label``: The predicted index of the category.
        - ``pred_score``: The score of the predicted category.

    Note:
        This function is reserved for compatibility and demo on a single image.
        We suggest to use :class:`ImageClassificationInferencer`, which is more
        powerful and configurable.
    """
    inferencer = ImageClassificationInferencer(model, device=device)
    return inferencer(img)[0]


class ImageClassificationInferencer(BaseInferencer):
    """The inferencer for image classification.

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

           >>> from mmcls import ImageClassificationInferencer
           >>> inferencer = ImageClassificationInferencer('resnet50_8xb32_in1k')
           >>> inferencer('demo/demo.JPEG')
           [{'pred_score': array([...]),
             'pred_label': 65,
             'pred_score': 0.6649367809295654,
             'pred_class': 'sea snake'}]

        2. Use a config file and checkpoint to inference multiple images on GPU,
           and save the visualization results in a folder.

           >>> from mmcls import ImageClassificationInferencer
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
        weights: Optional[str] = None,
        device: Union[str, torch.device, None] = None,
        classes=None,
    ) -> None:
        if isinstance(model, BaseModel):
            if weights is not None:
                load_checkpoint(model, weights, map_location='cpu')
            model = model.to(device)
        elif isinstance(model, str) and not Path(model).is_file():
            # Get model from model name
            pretrained = weights if weights is not None else True
            model = get_model(model, pretrained=pretrained, device=device)
        elif isinstance(model, (Config, str)):
            # Get model from config
            model = init_model(model, checkpoint=weights, device=device)
        else:
            raise TypeError(
                'The `model` can be a name of model and you can use '
                '`mmcls.list_models` to get an available name. It can '
                'also be a Config object or a path to the config file.')

        model.eval()

        self.cfg = model.cfg
        self.model = model
        self.pipeline = self._init_pipeline(self.cfg)
        self.collate_fn = default_collate
        self.visualizer = None

        self.classes = classes or getattr(self.model, 'CLASSES', None)

    def __call__(self,
                 inputs: List[InputType],
                 return_datasamples: bool = False,
                 batch_size: int = 1,
                 **kwargs) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            return_datasamples (bool): Whether to return results as
                :obj:`BaseDataElement`. Defaults to False.
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
                  preds: List[ClsDataSample],
                  show: bool = False,
                  rescale_factor: Optional[float] = None,
                  draw_score=True,
                  show_dir=None):
        if not show and show_dir is None:
            return None

        if self.visualizer is None:
            from mmcls.visualization import ClsVisualizer
            self.visualizer = ClsVisualizer()
            if self.classes is not None:
                self.visualizer._dataset_meta = dict(classes=self.classes)

        visualization = []
        for i, (input_, data_sample) in enumerate(zip(ori_inputs, preds)):
            image = imread(input_)
            if isinstance(input_, str):
                # The image loaded from path is BGR format.
                image = image[..., ::-1]
                name = Path(input_).stem
            else:
                name = str(i)

            if show_dir is not None:
                show_dir = Path(show_dir)
                show_dir.mkdir(exist_ok=True)
                out_file = str((show_dir / name).with_suffix('.png'))
            else:
                out_file = None

            self.visualizer.add_datasample(
                name,
                image,
                data_sample,
                show=show,
                rescale_factor=rescale_factor,
                draw_gt=False,
                draw_pred=True,
                draw_score=draw_score,
                out_file=out_file)
            visualization.append(self.visualizer.get_image())
        if show:
            self.visualizer.close()
        return visualization

    def postprocess(self,
                    preds: List[ClsDataSample],
                    visualization: List[np.ndarray],
                    return_datasamples=False) -> dict:
        if return_datasamples:
            return preds

        results = []
        for data_sample in preds:
            prediction = data_sample.pred_label
            pred_scores = prediction.score.detach().cpu().numpy()
            pred_score = torch.max(prediction.score).item()
            pred_label = torch.argmax(prediction.score).item()
            result = {
                'pred_scores': pred_scores,
                'pred_label': pred_label,
                'pred_score': float(pred_score),
            }
            if self.classes is not None:
                result['pred_class'] = self.classes[pred_label]
            results.append(result)

        return results

    @staticmethod
    def list_models(pattern: Optional[str] = None):
        """List all available model names.

        Args:
            pattern (str | None): A wildcard pattern to match model names.

        Returns:
            List[str]: a list of model names.
        """
        return list_models(pattern=pattern)
