# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import mmcv
import numpy as np
from mmengine.dist import master_only
from mmengine.visualization import Visualizer

from mmpretrain.registry import VISUALIZERS
from mmpretrain.structures import ClsDataSample


def _get_adaptive_scale(img_shape: Tuple[int, int],
                        min_scale: float = 0.3,
                        max_scale: float = 3.0) -> float:
    """Get adaptive scale according to image shape.

    The target scale depends on the the short edge length of the image. If the
    short edge length equals 224, the output is 1.0. And output linear scales
    according the short edge length.

    You can also specify the minimum scale and the maximum scale to limit the
    linear scale.

    Args:
        img_shape (Tuple[int, int]): The shape of the canvas image.
        min_size (int): The minimum scale. Defaults to 0.3.
        max_size (int): The maximum scale. Defaults to 3.0.

    Returns:
        int: The adaptive scale.
    """
    short_edge_length = min(img_shape)
    scale = short_edge_length / 224.
    return min(max(scale, min_scale), max_scale)


@VISUALIZERS.register_module()
class ClsVisualizer(Visualizer):
    """Universal Visualizer for classification task.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        fig_save_cfg (dict): Keyword parameters of figure for saving.
            Defaults to empty dict.
        fig_show_cfg (dict): Keyword parameters of figure for showing.
            Defaults to empty dict.

    Examples:
        >>> import torch
        >>> import mmcv
        >>> from pathlib import Path
        >>> from mmpretrain.visualization import ClsVisualizer
        >>> from mmpretrain.structures import ClsDataSample
        >>> # Example image
        >>> img = mmcv.imread("./demo/bird.JPEG", channel_order='rgb')
        >>> # Example annotation
        >>> data_sample = ClsDataSample().set_gt_label(1).set_pred_label(1).\
        ...     set_pred_score(torch.tensor([0.1, 0.8, 0.1]))
        >>> # Setup the visualizer
        >>> vis = ClsVisualizer(
        ...     save_dir="./outputs",
        ...     vis_backends=[dict(type='LocalVisBackend')])
        >>> # Set classes names
        >>> vis.dataset_meta = {'classes': ['cat', 'bird', 'dog']}
        >>> # Show the example image with annotation in a figure.
        >>> # And it will ignore all preset storage backends.
        >>> vis.add_datasample('res', img, data_sample, show=True)
        >>> # Save the visualization result by the specified storage backends.
        >>> vis.add_datasample('res', img, data_sample)
        >>> assert Path('./outputs/vis_data/vis_image/res_0.png').exists()
        >>> # Save another visualization result with the same name.
        >>> vis.add_datasample('res', img, data_sample, step=1)
        >>> assert Path('./outputs/vis_data/vis_image/res_1.png').exists()
    """

    @master_only
    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       data_sample: Optional[ClsDataSample] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       draw_score: bool = True,
                       rescale_factor: Optional[float] = None,
                       show: bool = False,
                       text_cfg: dict = dict(),
                       wait_time: float = 0,
                       out_file: Optional[str] = None,
                       step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If ``out_file`` is specified, all storage backends are ignored
          and save the image to the ``out_file``.
        - If ``show`` is True, plot the result image in a window, please
          confirm you are able to access the graphical interface.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (:obj:`ClsDataSample`, optional): The annotation of the
                image. Defaults to None.
            draw_gt (bool): Whether to draw ground truth labels.
                Defaults to True.
            draw_pred (bool): Whether to draw prediction labels.
                Defaults to True.
            draw_score (bool): Whether to draw the prediction scores
                of prediction categories. Defaults to True.
            rescale_factor (float, optional): Rescale the image by the rescale
                factor before visualization. Defaults to None.
            show (bool): Whether to display the drawn image. Defaults to False.
            text_cfg (dict): Extra text setting, which accepts
                arguments of :attr:`mmengine.Visualizer.draw_texts`.
                Defaults to an empty dict.
            wait_time (float): The interval of show (s). Defaults to 0, which
                means "forever".
            out_file (str, optional): Extra path to save the visualization
                result. If specified, the visualizer will only save the result
                image to the out_file and ignore its storage backends.
                Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        classes = None
        if self.dataset_meta is not None:
            classes = self.dataset_meta.get('classes', None)

        if rescale_factor is not None:
            image = mmcv.imrescale(image, rescale_factor)

        texts = []
        self.set_image(image)

        if draw_gt and 'gt_label' in data_sample:
            gt_label = data_sample.gt_label
            idx = gt_label.label.tolist()
            class_labels = [''] * len(idx)
            if classes is not None:
                class_labels = [f' ({classes[i]})' for i in idx]
            labels = [str(idx[i]) + class_labels[i] for i in range(len(idx))]
            prefix = 'Ground truth: '
            texts.append(prefix + ('\n' + ' ' * len(prefix)).join(labels))

        if draw_pred and 'pred_label' in data_sample:
            pred_label = data_sample.pred_label
            idx = pred_label.label.tolist()
            score_labels = [''] * len(idx)
            class_labels = [''] * len(idx)
            if draw_score and 'score' in pred_label:
                score_labels = [
                    f', {pred_label.score[i].item():.2f}' for i in idx
                ]

            if classes is not None:
                class_labels = [f' ({classes[i]})' for i in idx]

            labels = [
                str(idx[i]) + score_labels[i] + class_labels[i]
                for i in range(len(idx))
            ]
            prefix = 'Prediction: '
            texts.append(prefix + ('\n' + ' ' * len(prefix)).join(labels))

        img_scale = _get_adaptive_scale(image.shape[:2])
        text_cfg = {
            'positions': np.array([(img_scale * 5, ) * 2]).astype(np.int32),
            'font_sizes': int(img_scale * 7),
            'font_families': 'monospace',
            'colors': 'white',
            'bboxes': dict(facecolor='black', alpha=0.5, boxstyle='Round'),
            **text_cfg
        }
        self.draw_texts('\n'.join(texts), **text_cfg)
        drawn_img = self.get_image()

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            # save the image to the target file instead of vis_backends
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step=step)
