# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import datetime
import subprocess
from pathlib import Path

import torch
from mmcv import digit_version


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('in_file', help='input checkpoint filename')
    parser.add_argument('out_file', help='output checkpoint filename')
    args = parser.parse_args()
    return args


def process_checkpoint(in_file, out_file):
    checkpoint = torch.load(in_file, map_location='cpu')
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    if digit_version(torch.__version__) >= digit_version('1.6'):
        torch.save(checkpoint, out_file, _use_new_zipfile_serialization=False)
    else:
        torch.save(checkpoint, out_file)

    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    if out_file.endswith('.pth'):
        out_file_name = out_file[:-4]
    else:
        out_file_name = out_file

    current_date = datetime.datetime.now().strftime('%Y%m%d')
    final_file = out_file_name + f'_{current_date}-{sha[:8]}.pth'
    subprocess.Popen(['mv', out_file, final_file])

    print(f'Successfully generated the publish-ckpt as {final_file}.')


def main():
    args = parse_args()
    out_dir = Path(args.out_file).parent
    if not out_dir.exists():
        raise ValueError(f'Directory {out_dir} does not exist, '
                         'please generate it manually.')
    process_checkpoint(args.in_file, args.out_file)


if __name__ == '__main__':
    main()
