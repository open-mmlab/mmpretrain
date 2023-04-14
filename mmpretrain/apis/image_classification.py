# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
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
from mmpretrain.structures import DataSample
from .model import get_model, list_models

ModelType = Union[BaseModel, str, Config]
InputType = Union[str, np.ndarray, list]


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
        1. Use a pre-trained model in MMPreTrain to inference an image.

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
        'resize', 'rescale_factor', 'draw_score', 'show', 'show_dir',
        'wait_time'
    }

    def __init__(
        self,
        model: ModelType,
        pretrained: Union[bool, str] = True,
        device: Union[str, torch.device, None] = None,
        classes=None,
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

        if classes is not None:
            self.classes = classes
        else:
            self.classes = getattr(model, 'dataset_meta', {}).get('classes')

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
            resize (int, optional): Resize the short edge of the image to the
                specified length before visualization. Defaults to None.
            rescale_factor (float, optional): Rescale the image by the rescale
                factor for visualization. This is helpful when the image is too
                large or too small for visualization. Defaults to None.
            draw_score (bool): Whether to draw the prediction scores
                of prediction categories. Defaults to True.
            show (bool): Whether to display the visualization result in a
                window. Defaults to False.
            wait_time (float): The display time (s). Defaults to 0, which means
                "forever".
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
                  wait_time: int = 0,
                  resize: Optional[int] = None,
                  rescale_factor: Optional[float] = None,
                  draw_score=True,
                  show_dir=None):
        if not show and show_dir is None:
            return None

        if self.visualizer is None:
            from mmpretrain.visualization import UniversalVisualizer
            self.visualizer = UniversalVisualizer()

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

            self.visualizer.visualize_cls(
                image,
                data_sample,
                classes=self.classes,
                resize=resize,
                show=show,
                wait_time=wait_time,
                rescale_factor=rescale_factor,
                draw_gt=False,
                draw_pred=True,
                draw_score=draw_score,
                name=name,
                out_file=out_file)
            visualization.append(self.visualizer.get_image())
        if show:
            self.visualizer.close()
        return visualization

    def postprocess(self,
                    preds: List[DataSample],
                    visualization: List[np.ndarray],
                    return_datasamples=False) -> dict:
        if return_datasamples:
            return preds

        results = []
        for data_sample in preds:
            pred_scores = data_sample.pred_score
            pred_score = float(torch.max(pred_scores).item())
            pred_label = torch.argmax(pred_scores).item()
            result = {
                'pred_scores': pred_scores.detach().cpu().numpy(),
                'pred_label': pred_label,
                'pred_score': pred_score,
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
        return list_models(pattern=pattern, task='Image Classification')
