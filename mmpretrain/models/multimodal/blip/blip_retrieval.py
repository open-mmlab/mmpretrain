# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from collections import ChainMap
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import mmengine.dist as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from mmengine.utils import track_iter_progress
from torch import distributed as torch_dist
from torch.utils.data import DataLoader

from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample


def all_gather_diff_size(data: torch.Tensor):
    """Gather tensor with different lengths in a list.

    Note:
        Only the first dimension should be different.

    Args:
        data (Tensor): Tensor to be gathered.

    Returns:
        list[Tensor]: Return a list containing data from the whole group if
        in distributed environment, otherwise a list only containing
        :attr:`data` itself.
    """
    data_size = torch.tensor(data.size(0), device=data.device)
    sizes_list = dist.all_gather(data_size)

    max_length = max(sizes_list)
    size_diff = max_length.item() - data_size.item()
    if size_diff:
        padding = torch.zeros(
            size_diff, *data.size()[1:], device=data.device, dtype=data.dtype)
        data = torch.cat((data, padding))

    gather_list = dist.all_gather(data)

    all_data = []
    for tensor, size in zip(gather_list, sizes_list):

        all_data.append(tensor[:size])
    return all_data


@MODELS.register_module()
class BLIPRetriever(BaseModel):
    """BLIP Retriever.

    Args:
        vision_backbone (dict): Backbone for extracting image features.
        text_backbone (dict): Backbone for extracting text features.
        multimodal_backbone (Optional[dict]): Backbone for extracting
            multi-modal features.
        vision_neck (Optional[dict]): The neck module to process image features
            from vision backbone. Defaults to None.
        text_neck (Optional[dict]): The neck module to process text features
            from text backbone. Defaults to None.
        head (Optional[Union[List[dict], dict]]): The head module to calculate
            loss from processed single modality features.
            See :mod:`mmmultimodal.models.heads`.
            Notice that if the head is not set, `loss` method cannot be used.
            Defaults to None.
        multimodal_head (Optional[Union[List[dict], dict]]): The multi-modal
            head module to calculate loss from processed multimodal features.
            See :mod:`mmmultimodal.models.heads`.
            Notice that if the head is not set, `loss` method cannot be used.
            Defaults to None.
        momentum (float): Momentum used for momentum contrast.
            Defaults to .995.
        negative_all_rank (bool): Whether to sample negative data from all
            ranks for image text matching in training. Defaults to True.
        temperature (float): Temperature parameter that controls the
            concentration level of the distribution. Defaults to 0.07.
        topk (int): Select topk similarity as candidates for compute matching
            scores. Notice that this is not the topk in evaluation.
            Defaults to 256.
        train_cfg (Optional[dict]): The training setting. The acceptable
            fields are:
            - augments (List[dict]): The batch augmentation methods to use.
              More details can be found in
              :mod:`mmmultimodal.model.utils.augment`.
            Defaults to None.
        data_preprocessor (Optional[dict]): The config for preprocessing input
            data. If None or no specified type, it will use
            "MutimodalDataPreprocessor" as type.
            See :class:`MutimodalDataPreprocessor` for more details.
            Defaults to None.
        prototype (Union[DataLoader, dict, str, torch.Tensor]): Database to be
            retrieved. The following four types are supported.
            - DataLoader: The original dataloader serves as the prototype.
            - dict: The configuration to construct Dataloader.
            - str: The path of the saved vector.
            - torch.Tensor: The saved tensor whose dimension should be dim.
        init_cfg (Optional[dict]): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 vision_backbone: dict,
                 text_backbone: dict,
                 multimodal_backbone: Optional[dict] = None,
                 vision_neck: Optional[dict] = None,
                 text_neck: Optional[dict] = None,
                 head: Optional[Union[List[dict], dict]] = None,
                 multimodal_head: Optional[Union[List[dict], dict]] = None,
                 tokenizer: Optional[dict] = None,
                 momentum: float = .995,
                 negative_all_rank: bool = True,
                 temperature: float = 0.07,
                 topk: int = 256,
                 max_txt_len: int = 20,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 prototype: Union[DataLoader, dict, str, torch.Tensor] = None,
                 init_cfg: Optional[dict] = None):
        if data_preprocessor is None:
            data_preprocessor = {}
            # The build process is in MMEngine, so we need to add scope here.
            data_preprocessor.setdefault('type', 'MultiModalDataPreprocessor')

        if train_cfg is not None and 'augments' in train_cfg:
            # Set batch augmentations by `train_cfg`
            data_preprocessor['batch_augments'] = train_cfg

        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        self.vision_backbone = MODELS.build(vision_backbone)
        self.text_backbone = MODELS.build(text_backbone)

        if multimodal_backbone is not None:
            self.multimodal_backbone = MODELS.build(multimodal_backbone)

        if vision_neck is not None:
            self.vision_neck = MODELS.build(vision_neck)

        if text_neck is not None:
            self.text_neck = MODELS.build(text_neck)

        if head is not None:
            self.head = MODELS.build(head)

        if multimodal_head is not None:
            self.multimodal_head = MODELS.build(multimodal_head)

        assert isinstance(
            prototype, (str, torch.Tensor, dict, DataLoader, type(None))), (
                'The `prototype` in  `Retriever` must be a path, '
                'a torch.Tensor, a dataloader, a dataloader dict format config'
                ' or None.')
        self.prototype = prototype
        self.prototype_inited = False

        if tokenizer is not None:
            self.tokenizer = TOKENIZER.build(tokenizer)

        self.momentum = momentum
        self.negative_all_rank = negative_all_rank
        self.temp = nn.Parameter(temperature * torch.ones([]))
        # Shares the same para
        self.head.temp = self.temp

        # create the momentum encoder
        self.vision_backbone_m = deepcopy(self.vision_backbone)
        self.text_backbone_m = deepcopy(self.text_backbone)

        self.vision_neck_m = deepcopy(self.vision_neck)
        self.text_neck_m = deepcopy(self.text_neck)

        self.model_pairs = [
            [self.vision_backbone, self.vision_backbone_m],
            [self.text_backbone, self.text_backbone_m],
            [self.vision_neck, self.vision_neck_m],
            [self.text_neck, self.text_neck_m],
        ]
        self.copy_params()

        # multimodal backone shares weights with text backbone in BLIP
        # No need to set up

        # Notice that this topk is used for select k candidate to compute
        # image-text score, but not the final metric topk in evaluation.
        self.topk = topk

        self.max_txt_len = max_txt_len

    @property
    def device(self):
        return next(self.parameters()).device

    def preprocess_text(self, data_samples):
        sample_item = data_samples[0]

        if sample_item is not None and 'text' in sample_item:
            if isinstance(sample_item.get('text'), (list, tuple)):
                texts = []
                for sample in data_samples:
                    texts.extend(sample.get('text'))
            elif isinstance(sample_item.get('text'), str):
                texts = [sample.get('text') for sample in data_samples]
            else:
                raise TypeError('text must be a string or a list of strings')

        # perform tokenize first if satisfied conditions
        texts = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors='pt',
        ).to(self.device)

        return texts

    def forward(self,
                images: dict,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor') -> Union[Tuple, dict]:
        """The unified entry for a forward process in both training and test.
        The method should accept two modes: "tensor", and "loss":

        - "tensor": Forward the whole network and return tensor without any
          post-processing, same as a common nn.Module.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.
        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        For unified "predict" mode in other mm repos. It is noticed that
        image-text retrieval cannot perform batch prediction since it will go
        through all the samples. A standard process of retrieval evaluation is
        to extract and collect all feats, and then predict all samples.
        Therefore the `predict` mode here is remained as a trigger
        to inform use to choose the right configurations.

        Args:
            images (torch.Tensor): The input inputs tensor of shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. It's required if ``mode="loss"``.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.
            - If ``mode="tensor"``, return a tuple.
            - If ``mode="loss"``, return a dict of tensor.
        """
        texts = self.preprocess_text(data_samples)
        inputs = {'images': images, 'texts': texts}
        if mode == 'tensor':
            # save tokenized inputs for re-use in predict_all
            return texts, self.extract_feat(inputs)
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            raise RuntimeError(
                'Batch prediction is not supported for image-text retrieval '
                'evaluation. Please set `val_cfg` and `test_cfg` field to '
                '`RetrievalValLoop` and `RetrievalTestLoop` correspondingly '
                'in config file for evaluation.')
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(
        self,
        inputs: dict,
    ) -> Tuple[torch.Tensor]:
        """Extract features from the input dict.

        Args:
            inputs (dict): A batch of inputs. The input tensor with of
                at least one modality. For image, the value is a tensor
                of shape (N, C, ...) in general.
                For text, the value is a dict of tokenized text inputs.

        Returns:
            Tuple[torch.Tensor]: The output features.
                If multimodal_backbone is not exist, tuple of torch.Tensor
                will be returned.
        """
        assert 'images' in inputs or 'texts' in inputs, \
            'At least single modality should be passed as inputs.'

        results = ()
        for key in ['images', 'texts']:
            # extract image and text feats if exists inputs
            if key in inputs:
                results = results + self._extract_feat(
                    inputs[key], modality=key)

        return results

    def _extract_feat(self, inputs: Union[torch.Tensor, dict],
                      modality: str) -> Tuple[torch.Tensor]:
        """Extract features from the single modality.

        Args:
            inputs (Union[torch.Tensor, dict]): A batch of inputs.
                For image, a tensor of shape (N, C, ...) in general.
                For text, a dict of tokenized text inputs.
            modality (str): Modality feature to be extracted. Only two
                options are supported.

                - ``images``: Only extract image features, mostly used for
                    inference.
                - ``texts``: Only extract text features, mostly used for
                    inference.

        Returns:
            Tuple[torch.Tensor]: The output features.
        """

        if modality == 'images':
            # extract image features
            image_embeds = self.vision_backbone(inputs)[0]
            image_feat = F.normalize(
                self.vision_neck(image_embeds[:, 0, :]), dim=-1)
            return image_embeds, image_feat
        elif modality == 'texts':
            # extract text features
            text_output = self.text_backbone(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                token_type_ids=None,
                return_dict=True,
                mode='text',
            )
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(
                self.text_neck(text_embeds[:, 0, :]), dim=-1)
            return text_embeds, text_feat
        else:
            raise RuntimeError(f'Invalid modality "{modality}".')

    def loss(
        self,
        inputs: dict,
        data_samples: Optional[List[DataSample]] = None,
    ) -> Dict[str, torch.tensor]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (dict): A batch of inputs. The input tensor with of
                at least one modality. For image, the value is a tensor
                of shape (N, C, ...) in general.
                For text, the value is a dict of tokenized text inputs.
            data_samples (Optional[List[DataSample]]):
                The annotation data of every samples. Defaults to None.

        Returns:
            Dict[str, torch.tensor]: a dictionary of loss components of
                both head and multimodal head.
        """
        image_embeds, image_feat, text_embeds, text_feat = self.extract_feat(
            inputs)

        img_inputs = inputs['images']
        text_inputs = inputs['texts']
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.vision_backbone_m(img_inputs)[0]
            image_feat_m = F.normalize(
                self.vision_neck_m(image_embeds_m[:, 0, :]), dim=-1)

            text_output_m = self.text_backbone_m(
                text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                token_type_ids=None,
                return_dict=True,
                mode='text',
            )
            text_embeds_m = text_output_m.last_hidden_state
            text_feat_m = F.normalize(
                self.text_neck_m(text_embeds_m[:, 0, :]), dim=-1)

        loss = self.head.loss(
            ([image_feat, text_feat, image_feat_m, text_feat_m], ),
            data_samples)

        # prepare for itm
        encoder_input_ids = text_inputs.input_ids.clone()
        encoder_input_ids[:,
                          0] = self.tokenizer.additional_special_tokens_ids[0]
        output_pos = self.text_backbone(
            encoder_input_ids,
            attention_mask=text_inputs.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        idx = torch.tensor([i.image_id for i in data_samples]).view(-1, 1)
        bs = idx.size(0)
        idxs = torch.cat(dist.all_gather(idx))
        if self.negative_all_rank:
            # compute sample similarity
            with torch.no_grad():
                mask = torch.eq(idx, idxs.t()).to(self.device)

                image_feat_world = torch.cat(dist.all_gather(image_feat))
                text_feat_world = torch.cat(dist.all_gather(text_feat))

                sim_i2t = image_feat @ text_feat_world.t() / self.temp
                sim_t2i = text_feat @ image_feat_world.t() / self.temp

                weights_i2t = F.softmax(sim_i2t, dim=1)
                weights_i2t.masked_fill_(mask, 0)

                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_t2i.masked_fill_(mask, 0)

            world_size = dist.get_world_size()
            if world_size == 1:
                image_embeds_world = image_embeds
            else:
                image_embeds_world = torch.cat(
                    torch_dist.nn.all_gather(image_embeds))

            # select a negative image (from all ranks) for each text
            image_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds_world[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

            # select a negative text (from all ranks) for each image
            input_ids_world = torch.cat(dist.all_gather(encoder_input_ids))
            att_mask_world = torch.cat(
                dist.all_gather(text_inputs.attention_mask))

            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(input_ids_world[neg_idx])
                text_atts_neg.append(att_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text_inputs.attention_mask, text_atts_neg],
                                  dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.text_backbone(
            text_ids_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = torch.cat(
            [
                output_pos.last_hidden_state[:, 0, :],
                output_neg.last_hidden_state[:, 0, :],
            ],
            dim=0,
        )

        # create false data samples
        data_samples.extend(
            [DataSample(is_matched=False) for _ in range(2 * bs)])
        loss_multimodal = self.multimodal_head.loss((vl_embeddings, ),
                                                    data_samples)

        return dict(ChainMap(loss, loss_multimodal))

    def predict_all(self,
                    text: Dict[str, torch.tensor],
                    feats: Tuple[torch.tensor],
                    img_size: int,
                    text_size: int,
                    data_samples: Optional[List[DataSample]] = None,
                    **kwargs) -> List[DataSample]:
        """Predict retrieval results after forward all batches. Therefore, we
        use all computed feats as inputs rather than predict on each batch.
        This method should be used with `RetrievaTestLoop` and
        `RetrievaValLoop`.

        Args:
            text (Dict[str, torch.tensor]): A dict of tokenized text inputs.
            img_feats (Tuple[torch.tensor]): The features extracted from the
                vision backbone.
            text_feats (Tuple[torch.tensor]): The features extracted from the
                text backbone.
            img_size (int): The size of all image samples, which is needed for
                gathering without redundant.
            text_size (int): The size of all text samples.
            img_data_samples (List[DataSample], optional): The
                annotation data of every samples. Defaults to None.
            text_data_samples (List[DataSample], optional): The
                annotation data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.

        Returns:
            List[DataSample]: the raw data_samples with
                the predicted results.
        """
        img_embeds, img_feats, _, text_feats = feats

        def _gather_all_in_order(tensor, size):
            tensor_list = all_gather_diff_size(tensor)
            return torch.cat(tensor_list)[:size]

        # all gather features from all rank for further prediction
        img_embeds_all = _gather_all_in_order(img_embeds, img_size)
        img_feats_all = _gather_all_in_order(img_feats, img_size)

        text_feats_all = _gather_all_in_order(text_feats, text_size)

        text_ids = torch.cat([t.input_ids for t in text])
        text_ids[:, 0] = self.tokenizer.additional_special_tokens_ids[0]
        text_atts = torch.cat([t.attention_mask for t in text])

        text_ids_all = _gather_all_in_order(text_ids, text_size)
        text_atts_all = _gather_all_in_order(text_atts, text_size)

        result_i2t = self.compute_score_matrix_i2t(img_feats, img_embeds,
                                                   text_feats_all,
                                                   text_ids_all, text_atts_all)
        result_t2i = self.compute_score_matrix_t2i(img_feats_all,
                                                   img_embeds_all, text_feats,
                                                   text_ids, text_atts)
        return self._get_predictions(result_i2t, data_samples,
                                     'i2t'), self._get_predictions(
                                         result_t2i, data_samples, 't2i')

    def compute_score_matrix_i2t(self, img_feats, img_embeds, text_feats,
                                 text_ids, text_atts):
        """Compare the score matrix for image-to-text retrieval. Every image
        should compare to all the text features.

        Args:
            img_feats (torch.Tensor): The input img feats tensor with shape
                (M, C). M stands for numbers of samples on a single GPU.
            img_embeds (torch.Tensor): The input img embeds tensor with shape
                (M, C). M stands for numbers of samples on a single GPU.
            text_feats (torch.Tensor): The input text feats tensor with shape
                (N, C). N stands for numbers of all samples on all GPUs.
            text_ids (torch.Tensor): The input tensor with shape (N, C).
            text_atts (torch.Tensor): The input tensor with shape (N, C).

        Returns:
            torch.Tensor: Score matrix of image-to-text retrieval.
        """

        # compute i2t sim matrix
        sim_matrix_i2t = img_feats @ text_feats.t()
        score_matrix_i2t = torch.full((img_feats.size(0), text_feats.size(0)),
                                      -100.0).to(self.device)

        for i in track_iter_progress(range(img_feats.size(0))):
            sims = sim_matrix_i2t[i]
            topk_sim, topk_idx = sims.topk(k=self.topk, dim=0)

            encoder_output = img_embeds[i].repeat(self.topk, 1, 1)
            encoder_att = torch.ones(
                encoder_output.size()[:-1], dtype=torch.long).to(self.device)
            output = self.text_backbone(
                text_ids[topk_idx],
                attention_mask=text_atts[topk_idx],
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True,
            )
            score = self.multimodal_head(
                (output.last_hidden_state[:, 0, :], ))[:, 1]
            score_matrix_i2t[i, topk_idx] = score + topk_sim

        return score_matrix_i2t

    def compute_score_matrix_t2i(self, img_feats, img_embeds, text_feats,
                                 text_ids, text_atts):
        """Compare the score matrix for text-to-image retrieval. Every text
        should compare to all the image features.

        Args:
            img_feats (torch.Tensor): The input img feats tensor with shape
                (M, C). M stands for numbers of samples on a single GPU.
            img_embeds (torch.Tensor): The input img embeds tensor with shape
                (M, C). M stands for numbers of samples on a single GPU.
            text_feats (torch.Tensor): The input text feats tensor with shape
                (N, C). N stands for numbers of all samples on all GPUs.
            text_ids (torch.Tensor): The input tensor with shape (M, C).
            text_atts (torch.Tensor): The input tensor with shape (M, C).

        Returns:
            torch.Tensor: Score matrix of text-to-image retrieval.
        """

        # compute t2i sim matrix
        sim_matrix_t2i = text_feats @ img_feats.t()
        score_matrix_t2i = torch.full((text_feats.size(0), img_feats.size(0)),
                                      -100.0).to(self.device)

        for i in track_iter_progress(range(text_feats.size(0))):
            sims = sim_matrix_t2i[i]
            topk_sim, topk_idx = sims.topk(k=self.topk, dim=0)

            encoder_output = img_embeds[topk_idx]
            encoder_att = torch.ones(
                encoder_output.size()[:-1], dtype=torch.long).to(self.device)
            output = self.text_backbone(
                text_ids[i].repeat(self.topk, 1),
                attention_mask=text_atts[i].repeat(self.topk, 1),
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True,
            )
            score = self.multimodal_head(
                (output.last_hidden_state[:, 0, :], ))[:, 1]
            score_matrix_t2i[i, topk_idx] = score + topk_sim

        return score_matrix_t2i

    def _get_predictions(self,
                         result: torch.Tensor,
                         data_samples: List[DataSample],
                         type: str = 'i2t'):
        """Post-process the output of retriever.

        Args:
            result (torch.Tensor): Score matrix of single retrieve,
                either from image or text.
            data_samples (List[DataSample], optional): The annotation
                data of every samples.
            type (str): Retrieve type, either `i2t` for image to text, or `t2i`
                text to image. Defaults to `i2t`.

        Returns:
            List[DataSample]: the raw data_samples with
                the predicted results.
        """

        # create data sample if not exists
        if data_samples is None:
            data_samples = [DataSample() for _ in range(result.size(0))]
        elif len(data_samples) != result.size(0):
            new_data_samples = [DataSample() for _ in range(result.size(0))]
            gt_image_id_list = list(
                itertools.chain(*[ds.gt_image_id for ds in data_samples]))
            gt_text_id_list = list(
                itertools.chain(*[ds.gt_text_id for ds in data_samples]))

            assert len(gt_image_id_list) == len(gt_text_id_list) == len(
                new_data_samples)
            for ds, gt_image_id, gt_text_id in zip(new_data_samples,
                                                   gt_image_id_list,
                                                   gt_text_id_list):
                if type == 'i2t':
                    ds.gt_label = gt_text_id
                elif type == 't2i':
                    ds.gt_label = gt_image_id
                else:
                    raise ValueError(f'Type {type} is not supported.')
            data_samples = new_data_samples
        else:
            for ds in data_samples:
                if type == 'i2t':
                    ds.gt_label = ds.gt_text_id
                elif type == 't2i':
                    ds.gt_label = ds.gt_image_id
                else:
                    raise ValueError(f'Type {type} is not supported.')

        for data_sample, score in zip(data_samples, result):
            idx = score.argmax(keepdim=True).detach()

            data_sample.set_pred_score(score)
            data_sample.set_pred_label(idx)
        return data_samples

    # TODO: add temperaily
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(),
                                      model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for (name,
                 param), (name_m,
                          param_m) in zip(model_pair[0].named_parameters(),
                                          model_pair[1].named_parameters()):
                # hack to behave the same
                if any([i in name for i in ['8', '9', '10', '11']
                        ]) and 'layers' in name and any(
                            [i in name for i in ['attn', 'ffn']]):
                    param_m.data = param.data
                else:
                    param_m.data = param_m.data * self.momentum + \
                        param.data * (1.0 - self.momentum)
