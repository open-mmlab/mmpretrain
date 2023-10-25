# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import gradio as gr
import torch

from mmpretrain.registry import MODELS, TRANSFORMS
from .config.ram_swin_large_14m import get_ram_cfg, test_transforms_cfg
from .run.inference import inference

parser = argparse.ArgumentParser(
    description='RAM(Recognize Anything Model) demo')
parser.add_argument(
    'ram_ckpt', type=str, help='pretrained file for ram (absolute path)')
parser.add_argument(
    'clip_ckpt',
    type=str,
    help='clip vit-base-p16 pretrained file (absolute path)')
args = parser.parse_args()

if torch.cuda.is_available():
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())
    ]
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    devices = [torch.device('mps')]
else:
    devices = [torch.device('cpu')]


def get_free_device():
    if hasattr(torch.cuda, 'mem_get_info'):
        free = [torch.cuda.mem_get_info(gpu)[0] for gpu in devices]
        select = max(zip(free, range(len(free))))[1]
    else:
        import random
        select = random.randint(0, len(devices) - 1)
    return devices[select]


device = get_free_device()


def ram_inference(image, tag_list, mode, threshold):
    test_transforms = TRANSFORMS.get('Compose')(transforms=test_transforms_cfg)
    model = MODELS.build(get_ram_cfg(mode=mode))
    model.load_state_dict(torch.load(args.ram_ckpt))
    model.device = device

    if mode == 'openset':
        categories = tag_list
        if categories != '':
            categories = categories.strip().split()
        else:
            categories = None
        model.set_openset(
            categories=categories,
            clip_ckpt=args.clip_ckpt,
            threshold=threshold)

    sample = dict(img=image)
    result = inference(sample, model, test_transforms, mode=mode)
    tag, tag_chinese, logits =  \
        result.get('tag_output')[0][0], result.get('tag_output')[1][0],\
        result.get('logits_output')[0]

    def wrap(tags, logits):
        if tags is None:
            return 'Openset mode has no tag_en'
        tag_lst = tags.split('|')
        rt_lst = []
        for i, tag in enumerate(tag_lst):
            tag = tag.strip()
            rt_lst.append(tag + f': {logits[i]:.2f}')
        return ' | '.join(rt_lst)

    return [wrap(tag, logits), wrap(tag_chinese, logits)]


def build_gradio():
    inputs = [
        gr.components.Image(label='image'),
        gr.components.Textbox(
            lines=2,
            label='tag_list',
            placeholder=
            'please input the categories split by keyboard "blank": ',
            value=''),
        gr.components.Radio(['normal', 'openset'],
                            label='mode',
                            value='normal'),
        gr.components.Slider(
            minimum=0, maximum=1, value=0.68, step=0.01, label='threshold')
    ]
    return gr.Interface(
        fn=ram_inference,
        inputs=inputs,
        outputs=[
            gr.components.Textbox(),
            gr.components.Textbox(info="it's translated from the english tags")
        ])


def main():
    build_gradio().launch()


if __name__ == '__main__':
    main()
