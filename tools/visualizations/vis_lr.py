import argparse
import time
import re
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from unittest.mock import MagicMock

from mmcv.utils import Config
import seaborn as sns
import mmcv.parallel.collate 
from mmcv import Config, DictAction
from mmcv.runner import OptimizerHook
from mmcv.runner import EpochBasedRunner
import torch.nn as nn
from mmcv.utils import print_log
from mmcv.utils.config import Config

from mmcls.utils import get_root_logger, load_json_logs

def mock_time_consuming_functions():
    '''In order to speed up, mock the fuctions that takes much time'''
    # mock function sleep in tain if runner
    time.sleep = MagicMock()
    # skip functin log hook info
    EpochBasedRunner.get_hook_info = MagicMock(return_value="")
    # mock function collate in dataloader
    mmcv.parallel.collate = MagicMock(return_value=dict())
    # skip  function loss.backward
    OptimizerHook.after_train_iter = MagicMock()

class SimpleModel(nn.Module):
    '''simple model that do nothing in train_step'''
    def __init__(self, **cfg) -> None:
        super(SimpleModel, self).__init__()
        self.cnov = nn.Conv2d(1, 1, 1)

    def train_step(self, data, optimizer):
        return {"loss" : 0}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize a Dataset Pipeline')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--work-dir', help='the dir to save log and picture')
    parser.add_argument('--title', type=str, help='title of figure')
    parser.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser.add_argument(
        '--window-size',
        default='12*7',
        help='size of the window to display images, in format of "$W*$H".')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    if args.window_size != '':
        assert re.match(r'\d+\*\d+', args.window_size), \
            "'window-size' must be in format 'W*H'."

    return args

def retrieve_data_cfg(config_path, cfg_options):
    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    data_cfg = cfg.data.train
    while 'dataset' in data_cfg:
        data_cfg = data_cfg['dataset']
    
    return cfg

def plot_curve(log_dict, args, key, cfg, timestamp, by_epoch=True):
    """Plot train key-iter graph."""
    sns.set_style(args.style)
    wind_w, wind_h = args.window_size.split('*')
    wind_w, wind_h = int(wind_w), int(wind_h)
    plt.figure(figsize=(wind_w, wind_h))  
    # if legend is None, use {filename}_{key} as legend
    legend = f'{osp.basename(args.config)[:-4]}_{key}'

    epochs = list(log_dict.keys())
    if key not in log_dict[epochs[0]]:
        raise KeyError(
            f'{args.config} does not contain key {key} '
            f'in train mode')
    xs, ys = [], []
    num_iters_per_epoch = log_dict[epochs[0]]['iter'][-1]
    for epoch in epochs:
        iters = log_dict[epoch]['iter']
        if log_dict[epoch]['mode'][-1] == 'val':
            iters = iters[:-1]
        if by_epoch:
            xs.append(
                np.array(iters) / num_iters_per_epoch + epoch - 1)
        else:
            xs.append(
                np.array(iters) + (epoch - 1) * num_iters_per_epoch)
        ys.append(np.array(log_dict[epoch][key][:len(iters)]))
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    if by_epoch:
        plt.xlabel('Epochs')
    else:
        plt.xlabel('Iters')
    plt.ylabel('Learning Rate')
    plt.plot(xs, ys, label=legend, linewidth=1)
    plt.legend()
    if args.title is None:
        plt.title(f"{osp.basename(args.config)} LR-schedule")
    else:
        plt.title(args.title)
    save_path = osp.join(cfg.work_dir, timestamp + ".jpg")
    plt.savefig(save_path)
    plt.show()
    plt.cla()


def do_train(cfg, timestamp):
    '''Use a model without forward and backward to simulate the training 
    process'''
    model = SimpleModel()
    cfg.data.train.pipeline = []
    from mmcls.datasets.builder import build_dataset
    datasets = [build_dataset(cfg.data.train)]

    from mmcls.apis import train_model
    train_model(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=False,
        timestamp=timestamp,
        device='cpu',
        meta=dict())

def print_info(cfg, base_path, lr_config):
    '''print sometine usefully message to remind users'''
    print("\n LR Config : ", lr_config)
    print(f"Logs and picture are save in  {cfg.work_dir}")
    print(f"Details of the lr can be seen in {base_path + '.log'}")
    print(f"Format json data is saved in {base_path + '.log.json'}")
    print(f"picture is saved in {base_path + '.jpg'}\n")

def main():
    mock_time_consuming_functions()
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.cfg_options)
    lr_config = str(cfg.lr_config)
    cfg.gpu_ids = range(1)
    cfg.seed = 1
    cfg.checkpoint_config = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    
    # init work_dir
    config_basename = osp.splitext(osp.basename(args.config))[0]
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', config_basename)

    # init logger
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    logger.info("Lr config : \n\n" + lr_config + "\n")

    # simulation training process
    do_train(cfg, timestamp)

    # analyze training logs and draw graphs
    log_jsonfile = osp.abspath(log_file + ".json")
    log_dict = load_json_logs([log_jsonfile])[0]
    by_epoch = True if cfg.runner.type == 'EpochBasedRunner' else False
    plot_curve(log_dict, args, "lr", cfg, timestamp, by_epoch)
    
    # print sometine usefully message
    base_path = osp.join(cfg.work_dir, timestamp)
    print_info(cfg, base_path, lr_config)

if __name__ == '__main__':
    main()
 