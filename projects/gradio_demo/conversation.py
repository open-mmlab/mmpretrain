# Modified from
# https://github.com/Vision-CAIR/MiniGPT-4/blob/main/minigpt4/conversation/conversation.py
import dataclasses
from typing import List

import torch


@dataclasses.dataclass
class Conversation:
    system: str
    roles: List[str]
    messages: List[List[str]]
    sep: str = '###'

    def get_prompt(self):
        ret = self.system + self.sep
        for role, message in self.messages:
            if message:
                ret += role + ': ' + message + self.sep
            else:
                ret += role + ':'
        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=[role for role in self.roles],
            messages=[[y for y in x] for x in self.messages],
            sep=self.sep,
        )

    def dict(self):
        return {
            'system': self.system,
            'roles': self.roles,
            'messages': self.messages,
            'offset': self.offset,
            'sep': self.sep,
        }


EN_CONV_VISION = Conversation(
    system='Give the following image. '
    'You will be able to see the image once I provide it to you. '
    'Please answer my questions in detail.',
    roles=['Ask', 'Answer'],
    messages=[],
    sep='###',
)

ZH_CONV_VISION = Conversation(
    system='给定一张图片，请仔细观察这张图片，并回答我的问题。',
    roles=['问', '答'],
    messages=[],
    sep='###',
)


class Chat:

    def __init__(self, inferencer, device, is_half=False):
        self.device = device
        self.inferencer = inferencer
        self.model = inferencer.model
        self.is_half = is_half
        if is_half:
            self.model = self.model.half()
        self.model = self.model.to(device)
        self.max_length = 2000

    def upload_img(self, image, conv, img_list):
        img = next(self.inferencer.preprocess([image]))
        img = self.model.data_preprocessor(img, False)['images']
        img = img.to(self.device)
        image_emb, _ = self.model.encode_img(img)
        img_list.append(image_emb)
        conv.append_message(conv.roles[0], '<Img><ImageHere></Img>')

    def get_context_emb(self, conv, img_list):
        prompt = conv.get_prompt()
        prompt_segs = prompt.split('<ImageHere>')
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors='pt',
                add_special_tokens=(i == 0)).to(self.device).input_ids
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [
            self.model.llama_model.model.embed_tokens(seg_token)
            for seg_token in seg_tokens
        ]
        mixed_embs = [
            emb for pair in zip(seg_embs[:-1], img_list) for emb in pair
        ] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[
                0] and conv.messages[-1][1][-6:] == '</Img>':
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer(self, conv, img_list, generation_cfg):
        conv.append_message(conv.roles[1], None)
        embs = self.get_context_emb(conv, img_list)
        cur_max_len = generation_cfg['max_new_tokens'] + embs.shape[1]
        if cur_max_len > self.max_length:
            print('Warning: The number of tokens in current conversation'
                  'exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, cur_max_len - self.max_length)
        embs = embs[:, begin_idx:]
        if self.is_half:
            embs = embs.half()
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            eos_token_id=self.model.end_token_id,
            **generation_cfg)

        output_token = outputs[0]
        if output_token[0] == 0:
            output_token = output_token[1:]
        elif output_token[0] == 1:
            output_token = output_token[1:]
            output_text = self.model.llama_tokenizer.decode(
                output_token,
                add_special_tokens=False,
                skip_special_tokens=True)
        output_text = output_text.split('###')[0]
        conv.messages[-1][1] = output_text
        return output_text
