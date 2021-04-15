import argparse
import warnings

import mmcv
import numpy as np
import onnx
import onnxruntime as rt
from mmcv import DictAction
from torch import nn

from mmcls.apis import single_gpu_test
from mmcls.datasets import build_dataloader, build_dataset


class ORTModel(nn.Module):
    """Warp onnxruntime model and verify model based on dataset.

    Args:
        onnx_file: the onnx file path.
        batch_infer: whether to support batch inference for the model.
    """

    def __init__(self, onnx_file, batch_infer=True):
        super(ORTModel, self).__init__()
        self.onnx_file = onnx_file
        self.batch_infer = batch_infer
        self.onnx_model = onnx.load(onnx_file)
        onnx.checker.check_model(self.onnx_model)

        # check if the input number is coorect.
        input_all = [node.name for node in self.onnx_model.graph.input]
        input_initializer = [
            node.name for node in self.onnx_model.graph.initializer
        ]
        self.net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(self.net_feed_input) == 1)

        # init onnxruntime inference seession
        self.sess = rt.InferenceSession(self.onnx_file)

    def forward(self, *input, **data):
        """Run onnxruntime inference.
        Args:
            data(dict): a dict containing images data and image meta data.

        Return:
            output(List[np.array]): a list of np.array, the dimension
                of np.array is the number of CLASSES.
        """
        retsults = []
        if self.batch_infer:
            result = self.sess.run(
                None,
                {self.net_feed_input[0]: data['img'].detach().numpy()})[0]
            batch_size = data['img'].size(0)
            for i in range(batch_size):
                retsults.append(result[i])
        else:
            batch_size = data['img'].size(0)
            for i in range(batch_size):
                result = self.sess.run(
                    None, {
                        self.net_feed_input[0]:
                        data['img'][i].unsqueeze(0).detach().numpy()
                    })[0]
                retsults.append(result[i])
        return retsults


def ort_test(config,
             onnx_model,
             output_file='results.pkl',
             batch_infer=True,
             metrics=None,
             metric_options=None):
    cfg = mmcv.Config.fromfile(config)
    # build dataset and dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        shuffle=False,
        round_up=False)
    # build onnxruntime model and run inference.
    model = ORTModel(onnx_model, batch_infer)
    outputs = single_gpu_test(model, data_loader)

    if metrics:
        results = dataset.evaluate(outputs, metrics, metric_options)
        for k, v in results.items():
            print(f'\n{k} : {v:.2f}')
    else:
        warnings.warn('Evaluation metrics are not specified.')
        scores = np.vstack(outputs)
        pred_score = np.max(scores, axis=1)
        pred_label = np.argmax(scores, axis=1)
        pred_class = [dataset.CLASSES[lb] for lb in pred_label]
        results = {
            'pred_score': pred_score,
            'pred_label': pred_label,
            'pred_class': pred_class
        }
    mmcv.dump(results, output_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Use Dataset to Verify Model Accuracy.')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('model', help='filename of the input ONNX model')
    parser.add_argument('--output-file', type=str, default='results.pkl')
    parser.add_argument(
        '--batch-infer',
        action='store_false',
        help='Whether to support batch inference for the model. \
            Defaults to True.')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
        '"accuracy", "precision", "recall", "f1_score", "support" for single '
        'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
        'multi-label dataset')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be parsed as a dict metric_options for dataset.evaluate()'
        ' function.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    ort_test(
        args.config,
        args.model,
        output_file=args.output_file,
        metrics=args.metrics,
        metric_options=args.metric_options,
        batch_infer=args.batch_infer)
