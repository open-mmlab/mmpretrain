from functools import partial
from pathlib import Path
from typing import Callable

import gradio as gr
import torch
from mmengine.logging import MMLogger

import mmpretrain
from mmpretrain.apis import (ImageCaptionInferencer,
                             ImageClassificationInferencer,
                             ImageRetrievalInferencer,
                             TextToImageRetrievalInferencer,
                             VisualGroundingInferencer,
                             VisualQuestionAnsweringInferencer)
from mmpretrain.utils.dependency import WITH_MULTIMODAL
from mmpretrain.visualization import UniversalVisualizer

mmpretrain.utils.progress.disable_progress_bar = True

logger = MMLogger('mmpretrain', logger_name='mmpre')
if torch.cuda.is_available():
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())
    ]
    logger.info(f'Available GPUs: {len(devices)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    devices = [torch.device('mps')]
    logger.info('Available MPS.')
else:
    devices = [torch.device('cpu')]
    logger.info('Available CPU.')


def get_free_device():
    if hasattr(torch.cuda, 'mem_get_info'):
        free = [torch.cuda.mem_get_info(gpu)[0] for gpu in devices]
        select = max(zip(free, range(len(free))))[1]
    else:
        import random
        select = random.randint(0, len(devices) - 1)
    return devices[select]


class InferencerCache:
    max_size = 2
    _cache = []

    @classmethod
    def get_instance(cls, instance_name, callback: Callable):
        if len(cls._cache) > 0:
            for i, cache in enumerate(cls._cache):
                if cache[0] == instance_name:
                    # Re-insert to the head of list.
                    cls._cache.insert(0, cls._cache.pop(i))
                    logger.info(f'Use cached {instance_name}.')
                    return cache[1]

        if len(cls._cache) == cls.max_size:
            cls._cache.pop(cls.max_size - 1)
            torch.cuda.empty_cache()
        device = get_free_device()
        instance = callback(device=device)
        logger.info(f'New instance {instance_name} on {device}.')
        cls._cache.insert(0, (instance_name, instance))
        return instance


class ImageCaptionTab:

    def __init__(self) -> None:
        self.model_list = ImageCaptionInferencer.list_models()
        self.tab = self.create_ui()

    def create_ui(self):
        with gr.Row():
            with gr.Column():
                select_model = gr.Dropdown(
                    label='Choose a model',
                    elem_id='image_caption_models',
                    elem_classes='select_model',
                    choices=self.model_list,
                    value='blip-base_3rdparty_coco-caption',
                )
            with gr.Column():
                image_input = gr.Image(
                    label='Input',
                    source='upload',
                    elem_classes='input_image',
                    interactive=True,
                    tool='editor',
                )
                caption_output = gr.Textbox(
                    label='Result',
                    lines=2,
                    elem_classes='caption_result',
                    interactive=False,
                )
                run_button = gr.Button(
                    'Run',
                    elem_classes='run_button',
                )
                run_button.click(
                    self.inference,
                    inputs=[select_model, image_input],
                    outputs=caption_output,
                )

    def inference(self, model, image):
        image = image[:, :, ::-1]
        inferencer_name = self.__class__.__name__ + model
        inferencer = InferencerCache.get_instance(
            inferencer_name, partial(ImageCaptionInferencer, model))

        result = inferencer(image)[0]
        return result['pred_caption']


class ImageClassificationTab:

    def __init__(self) -> None:
        self.short_list = [
            'resnet50_8xb32_in1k',
            'resnet50_8xb256-rsb-a1-600e_in1k',
            'swin-base_16xb64_in1k',
            'convnext-base_32xb128_in1k',
            'vit-base-p16_32xb128-mae_in1k',
        ]
        self.long_list = ImageClassificationInferencer.list_models()
        self.tab = self.create_ui()

    def create_ui(self):
        with gr.Row():
            with gr.Column():
                select_model = gr.Dropdown(
                    label='Choose a model',
                    elem_id='image_classification_models',
                    elem_classes='select_model',
                    choices=self.short_list,
                    value='swin-base_16xb64_in1k',
                )
                expand = gr.Checkbox(label='Browse all models')

                def browse_all_model(value):
                    models = self.long_list if value else self.short_list
                    return gr.update(choices=models)

                expand.select(
                    fn=browse_all_model, inputs=expand, outputs=select_model)
            with gr.Column():
                in_image = gr.Image(
                    label='Input',
                    source='upload',
                    elem_classes='input_image',
                    interactive=True,
                    tool='editor',
                )
                out_cls = gr.Label(
                    label='Result',
                    num_top_classes=5,
                    elem_classes='cls_result',
                )
                run_button = gr.Button(
                    'Run',
                    elem_classes='run_button',
                )
                run_button.click(
                    self.inference,
                    inputs=[select_model, in_image],
                    outputs=out_cls,
                )

    def inference(self, model, image):
        image = image[:, :, ::-1]

        inferencer_name = self.__class__.__name__ + model
        inferencer = InferencerCache.get_instance(
            inferencer_name, partial(ImageClassificationInferencer, model))
        result = inferencer(image)[0]['pred_scores'].tolist()

        if inferencer.classes is not None:
            classes = inferencer.classes
        else:
            classes = list(range(len(result)))

        return dict(zip(classes, result))


class ImageRetrievalTab:

    def __init__(self) -> None:
        self.model_list = ImageRetrievalInferencer.list_models()
        self.tab = self.create_ui()

    def create_ui(self):
        with gr.Row():
            with gr.Column():
                select_model = gr.Dropdown(
                    label='Choose a model',
                    elem_id='image_retri_models',
                    elem_classes='select_model',
                    choices=self.model_list,
                    value='resnet50-arcface_inshop',
                )
                topk = gr.Slider(minimum=1, maximum=6, value=3, step=1)
            with gr.Column():
                prototype = gr.File(
                    label='Retrieve from',
                    file_count='multiple',
                    file_types=['image'])
                image_input = gr.Image(
                    label='Query',
                    source='upload',
                    elem_classes='input_image',
                    interactive=True,
                    tool='editor',
                )
                retri_output = gr.Gallery(
                    label='Result',
                    elem_classes='img_retri_result',
                ).style(
                    columns=[3], object_fit='contain', height='auto')
                run_button = gr.Button(
                    'Run',
                    elem_classes='run_button',
                )
                run_button.click(
                    self.inference,
                    inputs=[select_model, prototype, image_input, topk],
                    outputs=retri_output,
                )

    def inference(self, model, prototype, image, topk):
        image = image[:, :, ::-1]

        import hashlib

        proto_signature = ''.join(file.name for file in prototype).encode()
        proto_signature = hashlib.sha256(proto_signature).hexdigest()
        inferencer_name = self.__class__.__name__ + model + proto_signature
        tmp_dir = Path(prototype[0].name).parent
        cache_file = tmp_dir / f'{inferencer_name}.pth'

        inferencer = InferencerCache.get_instance(
            inferencer_name,
            partial(
                ImageRetrievalInferencer,
                model,
                prototype=[file.name for file in prototype],
                prototype_cache=str(cache_file),
            ),
        )

        result = inferencer(image, topk=min(topk, len(prototype)))[0]
        return [(str(item['sample']['img_path']),
                 str(item['match_score'].cpu().item())) for item in result]


class TextToImageRetrievalTab:

    def __init__(self) -> None:
        self.model_list = TextToImageRetrievalInferencer.list_models()
        self.tab = self.create_ui()

    def create_ui(self):
        with gr.Row():
            with gr.Column():
                select_model = gr.Dropdown(
                    label='Choose a model',
                    elem_id='t2i_retri_models',
                    elem_classes='select_model',
                    choices=self.model_list,
                    value='blip-base_3rdparty_coco-retrieval',
                )
                topk = gr.Slider(minimum=1, maximum=6, value=3, step=1)
            with gr.Column():
                prototype = gr.File(
                    file_count='multiple', file_types=['image'])
                text_input = gr.Textbox(
                    label='Query',
                    elem_classes='input_text',
                    interactive=True,
                )
                retri_output = gr.Gallery(
                    label='Result',
                    elem_classes='img_retri_result',
                ).style(
                    columns=[3], object_fit='contain', height='auto')
                run_button = gr.Button(
                    'Run',
                    elem_classes='run_button',
                )
                run_button.click(
                    self.inference,
                    inputs=[select_model, prototype, text_input, topk],
                    outputs=retri_output,
                )

    def inference(self, model, prototype, text, topk):
        import hashlib

        proto_signature = ''.join(file.name for file in prototype).encode()
        proto_signature = hashlib.sha256(proto_signature).hexdigest()
        inferencer_name = self.__class__.__name__ + model + proto_signature
        tmp_dir = Path(prototype[0].name).parent
        cache_file = tmp_dir / f'{inferencer_name}.pth'

        inferencer = InferencerCache.get_instance(
            inferencer_name,
            partial(
                TextToImageRetrievalInferencer,
                model,
                prototype=[file.name for file in prototype],
                prototype_cache=str(cache_file),
            ),
        )

        result = inferencer(text, topk=min(topk, len(prototype)))[0]
        return [(str(item['sample']['img_path']),
                 str(item['match_score'].cpu().item())) for item in result]


class VisualGroundingTab:

    def __init__(self) -> None:
        self.model_list = VisualGroundingInferencer.list_models()
        self.tab = self.create_ui()
        self.visualizer = UniversalVisualizer(
            fig_save_cfg=dict(figsize=(16, 9)))

    def create_ui(self):
        with gr.Row():
            with gr.Column():
                select_model = gr.Dropdown(
                    label='Choose a model',
                    elem_id='vg_models',
                    elem_classes='select_model',
                    choices=self.model_list,
                    value='ofa-base_3rdparty_refcoco',
                )
            with gr.Column():
                image_input = gr.Image(
                    label='Image',
                    source='upload',
                    elem_classes='input_image',
                    interactive=True,
                    tool='editor',
                )
                text_input = gr.Textbox(
                    label='The object to search',
                    elem_classes='input_text',
                    interactive=True,
                )
                vg_output = gr.Image(
                    label='Result',
                    source='upload',
                    interactive=False,
                    elem_classes='vg_result',
                )
                run_button = gr.Button(
                    'Run',
                    elem_classes='run_button',
                )
                run_button.click(
                    self.inference,
                    inputs=[select_model, image_input, text_input],
                    outputs=vg_output,
                )

    def inference(self, model, image, text):

        inferencer_name = self.__class__.__name__ + model

        inferencer = InferencerCache.get_instance(
            inferencer_name,
            partial(VisualGroundingInferencer, model),
        )

        result = inferencer(
            image[:, :, ::-1], text, return_datasamples=True)[0]
        vis = self.visualizer.visualize_visual_grounding(
            image, result, resize=512)
        return vis


class VisualQuestionAnsweringTab:

    def __init__(self) -> None:
        self.model_list = VisualQuestionAnsweringInferencer.list_models()
        # The fine-tuned OFA vqa models requires extra object description.
        self.model_list.remove('ofa-base_3rdparty-finetuned_vqa')
        self.tab = self.create_ui()

    def create_ui(self):
        with gr.Row():
            with gr.Column():
                select_model = gr.Dropdown(
                    label='Choose a model',
                    elem_id='vqa_models',
                    elem_classes='select_model',
                    choices=self.model_list,
                    value='ofa-base_3rdparty-zeroshot_coco-vqa',
                )
            with gr.Column():
                image_input = gr.Image(
                    label='Input',
                    source='upload',
                    elem_classes='input_image',
                    interactive=True,
                    tool='editor',
                )
                question_input = gr.Textbox(
                    label='Question',
                    elem_classes='question_input',
                )
                answer_output = gr.Textbox(
                    label='Answer',
                    elem_classes='answer_result',
                )
                run_button = gr.Button(
                    'Run',
                    elem_classes='run_button',
                )
                run_button.click(
                    self.inference,
                    inputs=[select_model, image_input, question_input],
                    outputs=answer_output,
                )

    def inference(self, model, image, question):
        image = image[:, :, ::-1]

        inferencer_name = self.__class__.__name__ + model
        inferencer = InferencerCache.get_instance(
            inferencer_name, partial(VisualQuestionAnsweringInferencer, model))

        result = inferencer(image, question)[0]
        return result['pred_answer']


if __name__ == '__main__':
    title = 'MMPretrain Inference Demo'
    with gr.Blocks(analytics_enabled=False, title=title) as demo:
        gr.Markdown(f'# {title}')
        with gr.Tabs():
            with gr.TabItem('Image Classification'):
                ImageClassificationTab()
            with gr.TabItem('Image-To-Image Retrieval'):
                ImageRetrievalTab()
            if WITH_MULTIMODAL:
                with gr.TabItem('Image Caption'):
                    ImageCaptionTab()
                with gr.TabItem('Text-To-Image Retrieval'):
                    TextToImageRetrievalTab()
                with gr.TabItem('Visual Grounding'):
                    VisualGroundingTab()
                with gr.TabItem('Visual Question Answering'):
                    VisualQuestionAnsweringTab()
            else:
                with gr.TabItem('Multi-modal tasks'):
                    gr.Markdown(
                        'To inference multi-modal models, please install '
                        'the extra multi-modal dependencies, please refer '
                        'to https://mmpretrain.readthedocs.io/en/latest/'
                        'get_started.html#installation')

    demo.launch()
