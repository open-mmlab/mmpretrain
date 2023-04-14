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
        'draw_score', 'resize', 'show_dir', 'show', 'wait_time'
    }
    postprocess_kwargs: set = {'topk'}

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

        # An ugly hack to escape from the duplicated arguments check in the
        # base class
        self.visualize_kwargs.add('topk')

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
            resize (int, optional): Resize the long edge of the image to the
                specified length before visualization. Defaults to None.
            draw_score (bool): Whether to draw the match scores.
                Defaults to True.
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
                  topk: int = 3,
                  resize: Optional[int] = 224,
                  show: bool = False,
                  wait_time: int = 0,
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

            self.visualizer.visualize_image_retrieval(
                image,
                data_sample,
                self.prototype_dataset,
                topk=topk,
                resize=resize,
                draw_score=draw_score,
                show=show,
                wait_time=wait_time,
                name=name,
                out_file=out_file)
            visualization.append(self.visualizer.get_image())
        if show:
            self.visualizer.close()
        return visualization

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
                sample = self.prototype_dataset.get_data_info(
                    sample_idx.item())
                sample_idx = sample.pop('sample_idx')
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

    def _dispatch_kwargs(self, **kwargs):
        """Dispatch kwargs to preprocess(), forward(), visualize() and
        postprocess() according to the actual demands.

        Override this method to allow same argument for different methods.

        Returns:
            Tuple[Dict, Dict, Dict, Dict]: kwargs passed to preprocess,
            forward, visualize and postprocess respectively.
        """
        method_kwargs = set.union(
            self.preprocess_kwargs,
            self.forward_kwargs,
            self.visualize_kwargs,
            self.postprocess_kwargs,
        )

        union_kwargs = method_kwargs | set(kwargs.keys())
        if union_kwargs != method_kwargs:
            unknown_kwargs = union_kwargs - method_kwargs
            raise ValueError(
                f'unknown argument {unknown_kwargs} for `preprocess`, '
                '`forward`, `visualize` and `postprocess`')

        preprocess_kwargs = {}
        forward_kwargs = {}
        visualize_kwargs = {}
        postprocess_kwargs = {}

        for key, value in kwargs.items():
            if key in self.preprocess_kwargs:
                preprocess_kwargs[key] = value
            if key in self.forward_kwargs:
                forward_kwargs[key] = value
            if key in self.visualize_kwargs:
                visualize_kwargs[key] = value
            if key in self.postprocess_kwargs:
                postprocess_kwargs[key] = value

        return (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        )
