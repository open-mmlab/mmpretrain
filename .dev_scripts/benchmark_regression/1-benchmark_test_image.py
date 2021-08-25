import logging
import os.path as osp
from argparse import ArgumentParser
from pathlib import Path
from typing import OrderedDict

from mmcv import Config

from mmcls.apis import inference_model, init_model, show_result_pyplot
from mmcls.datasets.imagenet import ImageNet
from mmcls.utils import get_root_logger

MMCLS_ROOT = Path(__file__).absolute().parents[2]

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
]

CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree',
    'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy',
    'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
    'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
    'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train',
    'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf',
    'woman', 'worm'
]

classes_map = {
    'ImageNet-1k': ImageNet.CLASSES,
    'CIFAR-10': CIFAR10_CLASSES,
    'CIFAR-100': CIFAR100_CLASSES
}


def parse_args():
    parser = ArgumentParser(description='Valid all models in model-index.yml')
    parser.add_argument(
        '--checkpoint-root',
        help='Checkpoint file root path. If set, load checkpoint before test.')
    parser.add_argument('--img', default='demo/demo.JPEG', help='Image file')
    parser.add_argument('--model-name', help='model name to inference')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='the interval of show (s), 0 is block')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--log-level',
        default='error',
        choices=['info', 'error'],
        help='log level')
    args = parser.parse_args()
    return args


def inference(config_name, checkpoint, classes, args, logger=None):
    cfg = Config.fromfile(config_name)

    model = init_model(cfg, checkpoint, device=args.device)
    model.CLASSES = classes
    # test a single image
    result = inference_model(model, args.img)
    result['model'] = config_name.stem

    # show the results
    if args.show:
        show_result_pyplot(model, args.img, result, wait_time=args.wait_time)
    return result


# Sample test whether the inference code is correct
def main(args):
    model_index_file = MMCLS_ROOT / 'model-index.yml'
    model_index = Config.fromfile(model_index_file)
    models = OrderedDict()
    for file in model_index.Import:
        metafile = Config.fromfile(MMCLS_ROOT / file)
        models.update({model.Name: model for model in metafile.Models})

    # test single model
    if args.model_name:
        assert args.model_name in models, \
            f'Cannot find model "{args.model_name}".' \
            'Please select from: \n' + '\n'.join(models.keys())
        model_info = models[args.model_name]
        config_name = model_info.Config
        dataset = model_info.get('Dataset', 'ImageNet-1k')
        http_prefix = 'https://download.openmmlab.com/mmclassification/'
        if args.checkpoint_root is not None:
            checkpoint = osp.join(args.checkpoint_root,
                                  model_info.Weights[len(http_prefix):])
        else:
            checkpoint = None
        print(f'Processing: {config_name}', flush=True)
        # build the model from a config file and a checkpoint file
        inference(MMCLS_ROOT / config_name, checkpoint, classes_map[dataset],
                  args)
        print(f'Complete: {config_name}')
        return

    # test all model
    log_level = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'error': logging.ERROR
    }[args.log_level]
    logger = get_root_logger(
        log_file='benchmark_test_image.log', log_level=log_level)

    for model_key in models:
        model_info = models[model_key]
        logger.info(f'Processing: {model_info.Config}')
        if log_level == logging.ERROR:
            print(f'Processing: {model_info.Config}')
        config_name = model_info.Config
        http_prefix = 'https://download.openmmlab.com/mmclassification/'
        dataset = model_info.get('Dataset', 'ImageNet-1k')
        if args.checkpoint_root is not None:
            checkpoint = osp.join(args.checkpoint_root,
                                  model_info.Weights[len(http_prefix):])
        else:
            checkpoint = None
        try:
            # build the model from a config file and a checkpoint file
            result = inference(MMCLS_ROOT / config_name, checkpoint,
                               classes_map[dataset], args, logger)
        except Exception as e:
            logger.error(f'{config_name} " : {repr(e)}')
        else:
            logger.info(f'{config_name} " : {result}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
