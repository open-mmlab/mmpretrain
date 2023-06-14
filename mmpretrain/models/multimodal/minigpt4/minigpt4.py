import random
from typing import List, Optional

import torch
import torch.nn as nn
from mmengine.device import get_device
from mmengine.model import BaseModel
from transformers import StoppingCriteriaList

from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample
from .utils import StoppingCriteriaSub


@MODELS.register_module()
class MiniGPT4(BaseModel):
    """The model of MiniGPT-4.

    Args:
        vision_encoder='eva_clip_g',
        q_former_model="",
        lang_encoder='',
        tokenizer = '',
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        freeze_vit=True,
        freeze_q_former=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='',
        low_resource=False,  # use 8 bit and put vit in cpu
    """

    def __init__(self,
                 vision_encoder,
                 q_former_model='',
                 lang_encoder='',
                 tokenizer='',
                 freeze_vit=True,
                 freeze_q_former=True,
                 num_query_token=32,
                 prompt_template='',
                 raw_prompts=None,
                 max_txt_len=32,
                 end_sym='\n',
                 generation_cfg=dict(),
                 low_resource=False,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        if data_preprocessor is None:
            data_preprocessor = {}
        data_preprocessor.setdefault('type', 'MultiModalDataPreprocessor')
        data_preprocessor = MODELS.build(data_preprocessor)

        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # build vision model
        vision_encoder_weight = vision_encoder.pop('pretrained', None)
        self.vision_encoder = MODELS.build(vision_encoder)
        self.ln_vision = nn.LayerNorm(self.vision_encoder.embed_dims)

        if vision_encoder_weight is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(self.vision_encoder, vision_encoder_weight)
        if freeze_vit:
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()

        # build Qformer
        q_former_model_weight = q_former_model.pop('pretrained', None)
        self.q_former = MODELS.build(q_former_model)
        self.q_former.cls = None
        self.q_former.bert.embeddings.word_embeddings = None
        self.q_former.bert.embeddings.position_embeddings = None
        for layer in self.q_former.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.q_former.config.hidden_size))
        self.query_tokens.data.normal_(
            mean=0.0, std=self.q_former.config.initializer_range)

        if q_former_model_weight is not None:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            logger.info(f'Loading checkpoint from {q_former_model_weight}')
            state_dict = torch.load(q_former_model_weight)['state_dict']
            incompatible_keys = self.load_state_dict(state_dict, strict=False)
            logger.info(incompatible_keys)

        if freeze_q_former:
            for name, param in self.q_former.named_parameters():
                param.requires_grad = False
            self.q_former.eval()
            self.query_tokens.requires_grad = False

        # build language model
        self.llama_tokenizer = TOKENIZER.build(tokenizer)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        self.low_resource = low_resource
        self.llama_model = MODELS.build(lang_encoder)

        # if self.low_resource:
        #     self.llama_model = LlamaForCausalLM.from_pretrained(
        #         llama_model,
        #         torch_dtype=torch.float16,
        #         load_in_8bit=True,
        #         device_map={'': device_8bit})
        # else:
        #     self.llama_model = LlamaForCausalLM.from_pretrained(
        #         llama_model,
        #         torch_dtype=torch.float16,
        #     )
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False

        # build linear projection layer
        self.llama_proj = nn.Linear(self.q_former.config.hidden_size,
                                    self.llama_model.config.hidden_size)
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.end_token_id = self.llama_tokenizer.encode(end_sym)[-1]

        # set prompts
        if raw_prompts is not None:
            filted_prompts = [
                raw_prompt for raw_prompt in raw_prompts
                if "<ImageHere>" in raw_prompt
            ]
            self.prompt_list = [
                prompt_template.format(p) for p in filted_prompts
            ]
        else:
            self.prompt_list = []

        #
        self.generation_cfg = dict(
            max_new_tokens=300,
            num_beams=1,
            do_sample=True,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1.0,
            temperature=1.0,
            **generation_cfg)

        self.device = get_device()
        stop_words_ids = [
            torch.tensor([835]).to(self.device),
            torch.tensor([2277, 29937]).to(self.device),
        ]
        self.stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)])

    # def vit_to_cpu(self):
    #     self.ln_vision.to("cpu")
    #     self.ln_vision.float()
    #     self.vision_encoder.to("cpu")
    #     self.vision_encoder.float()

    def encode_img(self, images):
        device = images.device
        # if self.low_resource:
        #     self.vit_to_cpu()
        #     image = image.to("cpu")
        x = self.vision_encoder(images)[0]
        image_embeds = self.ln_vision(x).to(device)
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.q_former.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_llama = self.llama_proj(query_output.last_hidden_state)
        atts_llama = torch.ones(
            inputs_llama.size()[:-1], dtype=torch.long).to(images.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt",
                add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt",
                add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(
                p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(
                p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat(
                [p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(
                -1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def loss(self,
             images: torch.Tensor,
             data_samples: Optional[List[DataSample]] = None):
        img_embeds, atts_img = self.encode_img(images)
        if hasattr(data_samples, 'question_split'):  # VQA dataset
            vqa_prompt = '###Human: <Img><ImageHere></Img> '
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img,
                                                    vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img,
                                                    prompt)

        self.llama_tokenizer.padding_side = "right"

        text = [t + self.end_sym for t in data_samples["text_input"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False).to(images.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id,
            -100)

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1] + 1],
                       dtype=torch.long).to(images.device).fill_(
                           -100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device
                         ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(
            to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds],
                                  dim=1)
        attention_mask = torch.cat(
            [atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return dict(loss=loss)

    def predict(self,
                images: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None):
        with torch.no_grad():
            img_embeds, atts_img = self.encode_img(images)

        if hasattr(data_samples, 'question_split'):  # VQA dataset
            vqa_prompt = '###Human: <Img><ImageHere></Img> '
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img,
                                                    vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img,
                                                    prompt)

        batch_size = img_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1], dtype=torch.long,
            device=img_embeds.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            stopping_criteria=self.stopping_criteria,
            **self.generation_cfg)

        return self.post_process(outputs, data_samples)

    def post_process(
            self, outputs: torch.Tensor,
            data_samples: Optional[List[DataSample]]) -> List[DataSample]:
        """Perform post process for outputs for different task.

        Args:
            outputs (torch.Tensor): The generated outputs.
            data_samples (List[DataSample], optional): The annotation
                data of every samples.

        Returns:
            List[DataSample]: Return list of data samples.
        """
        outputs = self.llama_tokenizer.batch_decode(
            outputs, skip_special_tokens=True)

        if data_samples is None:
            data_samples = [DataSample() for _ in range(len(outputs))]

        for output, data_sample in zip(outputs, data_samples):
            output = output.split('###')[0]
            output = output.split('Assistant:')[-1].strip()
            data_sample.pred_answer = output

        return data_samples

    def forward(
        self,
        images: torch.Tensor,
        data_samples: Optional[list] = None,
        mode: str = 'predict',
        **kwargs,
    ):
        """The unified entry for a forward process in both training and test.
        The method accepts the following modes:

        - "predict": Forward and return a list of data samples contain the
          predict results.

        Args:
            images (torch.Tensor): the preprocessed image tensor of shape
                ``(N, C, H, W)``.
            data_samples (List[DataSample], optional): The annotation data
                of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'predict'.
        """
        if mode == 'loss':
            return self.loss(images, data_samples)
        elif mode == 'predict':
            return self.predict(images, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')
