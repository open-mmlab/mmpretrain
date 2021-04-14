import argparse
import warnings

import mmcv
import numpy as np
import onnx
import onnxruntime as rt
from mmcv import DictAction

from mmcls.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Use Dataset to Verify Model Accuracy.')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('model', help='filename of the input ONNX model')
    parser.add_argument('--output-file', type=str, default='results.pkl')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
        '"accuracy", "precision", "recall", "f1_score", "support" for single '
        'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
        'multi-label dataset')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--batch-infer',
        action='store_false',
        help='Whether to support batch inference for the model. \
            Defaults to True.')
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


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # build dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        shuffle=False,
        round_up=False)

    onnx_model = onnx.load(args.model)
    onnx.checker.check_model(onnx_model)

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert (len(net_feed_input) == 1)

    outputs = []
    for i, data in enumerate(data_loader):
        # support batch inference
        if args.batch_infer:
            sess = rt.InferenceSession(args.model)
            result = sess.run(
                None, {net_feed_input[0]: data['img'].detach().numpy()})[0]
            batch_size = data['img'].size(0)
            for i in range(batch_size):
                outputs.append(result[i])
                prog_bar.update()
        else:
            batch_size = data['img'].size(0)
            for i in range(batch_size):
                sess = rt.InferenceSession(args.model)
                result = sess.run(
                    None, {
                        net_feed_input[0]:
                        data['img'][i].unsqueeze(0).detach().numpy()
                    })[0]
                outputs.append(result[i])
    if args.metrics:
        results = dataset.evaluate(outputs, args.metrics, args.metric_options)
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
    mmcv.dump(results, args.output_file)


if __name__ == '__main__':
    main()
