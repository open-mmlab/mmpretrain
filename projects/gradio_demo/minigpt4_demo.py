import argparse

import gradio as gr
import numpy as np
import torch
from conversation import EN_CONV_VISION, ZH_CONV_VISION, Chat

from mmpretrain import ImageCaptionInferencer

parser = argparse.ArgumentParser(description='MiniGPT4 demo')
parser.add_argument(
    'cfg', type=str, help='config file for minigpt4 (absolute path)')
parser.add_argument(
    'ckpt', type=str, help='pretrained file for minigpt4 (absolute path)')
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
inferencer = ImageCaptionInferencer(model=args.cfg, pretrained=args.ckpt)
model = inferencer.model
chat = Chat(inferencer, device=device, is_half=(device.type != 'cpu'))


def reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return (None, gr.update(value=None, interactive=True),
            gr.update(
                value=None,
                placeholder='Please upload your image first',
                interactive=False),
            gr.update(value='Upload & Start Chat',
                      interactive=True), chat_state, img_list,
            gr.update(value='Restart', interactive=False),
            gr.update(value='English', interactive=True))


def upload_img(gr_img, language, chat_state):
    if gr_img is None:
        return (None,
                gr.update(
                    placeholder='Please upload your image first',
                    interactive=False),
                gr.update(value='Upload & Start Chat',
                          interactive=True), chat_state, None,
                gr.update(value='Restart', interactive=False),
                gr.update(value='English', interactive=True))

    if (language == 'English'):
        chat_state = EN_CONV_VISION.copy()
    else:
        chat_state = ZH_CONV_VISION.copy()
    img_list = []
    gr_img_array = np.asarray(gr_img)
    chat.upload_img(gr_img_array, chat_state, img_list)
    return (gr.update(interactive=False),
            gr.update(placeholder='Type and press Enter', interactive=True),
            gr.update(value='Start Chatting',
                      interactive=False), chat_state, img_list,
            gr.update(value='Restart',
                      interactive=True), gr.update(interactive=False))


def ask(user_message, chatbot, chat_state):
    if (len(user_message) == 0):
        return gr.update(
            value=None,
            placeholder='Input should not be empty!',
            interactive=True), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def answer(chatbot, chat_state, img_list):
    llm_message = chat.answer(
        conv=chat_state,
        img_list=img_list,
        generation_cfg=model.generation_cfg)
    chatbot[-1][1] = llm_message
    return chatbot, chat_state, img_list


if __name__ == '__main__':
    title = 'MMPretrain MiniGPT-4 Inference Demo'
    with gr.Blocks(analytics_enabled=False, title=title) as demo:
        gr.Markdown(f'# {title}')
        with gr.Row():
            with gr.Column():
                image = gr.Image(type='pil')
                language = gr.Dropdown(['English', 'Chinese'],
                                       label='Language',
                                       info='Select chatbot\'s language',
                                       value='English',
                                       interactive=True)
                upload_button = gr.Button(
                    value='Upload & Start Chat', interactive=True)
                clear = gr.Button(value='Restart', interactive=False)

            with gr.Column():
                chat_state = gr.State()
                img_list = gr.State()
                chatbot = gr.Chatbot(
                    label='MiniGPT-4', min_width=320, height=600)
                text_input = gr.Textbox(
                    label='User',
                    placeholder='Please upload your image first',
                    interactive=False)

        upload_button.click(upload_img, [image, language, chat_state], [
            image, text_input, upload_button, chat_state, img_list, clear,
            language
        ])
        text_input.submit(ask, [text_input, chatbot, chat_state],
                          [text_input, chatbot, chat_state]).then(
                              answer, [chatbot, chat_state, img_list],
                              [chatbot, chat_state, img_list])
        clear.click(reset, [chat_state, img_list], [
            chatbot, image, text_input, upload_button, chat_state, img_list,
            clear, language
        ])

    demo.launch(share=True)
