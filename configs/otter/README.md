# Otter

> [Otter: A Multi-Modal Model with In-Context Instruction Tuning](https://arxiv.org/abs/2305.03726)

<!-- [ALGORITHM] -->

## Abstract

Large language models (LLMs) have demonstrated significant universal capabilities as few/zero-shot learners in various tasks due to their pre-training on vast amounts of text data, as exemplified by GPT-3, which boosted to InstrctGPT and ChatGPT, effectively following natural language instructions to accomplish real-world tasks. In this paper, we propose to introduce instruction tuning into multi-modal models, motivated by the Flamingo model's upstream interleaved format pretraining dataset. We adopt a similar approach to construct our MultI-Modal In-Context Instruction Tuning (MIMIC-IT) dataset. We then introduce Otter, a multi-modal model based on OpenFlamingo (open-sourced version of DeepMind's Flamingo), trained on MIMIC-IT and showcasing improved instruction-following ability and in-context learning. We also optimize OpenFlamingo's implementation for researchers, democratizing the required training resources from 1$\times$ A100 GPU to 4$\times$ RTX-3090 GPUs, and integrate both OpenFlamingo and Otter into Huggingface Transformers for more researchers to incorporate the models into their customized training and inference pipelines.

<div align=center>
<img src="https://camo.githubusercontent.com/70613ab882a7827808148a2c577029d544371e707b0832a0b01151c54ce553c3/68747470733a2f2f692e706f7374696d672e63632f5477315a304243572f6f7474657276302d322d64656d6f2e706e67" width="80%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Use the model**

```python
import torch
from mmpretrain import get_model, inference_model

model = get_model('otter-9b_3rdparty_caption', pretrained=True, device='cuda', generation_cfg=dict(max_new_tokens=50))
out = inference_model(model, 'demo/cat-dog.png')
print(out)
# {'pred_caption': 'The image features two adorable small puppies sitting next to each other on the grass. One puppy is brown and white, while the other is tan and white. They appear to be relaxing outdoors, enjoying each other'}
```

**Test Command**

Prepare your dataset according to the [docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset).

Test:

```shell
python tools/test.py configs/otter/otter-9b_caption.py https://download.openmmlab.com/mmclassification/v1/otter/otter-9b-adapter_20230613-51c5be8d.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Caption on COCO

| Model                         | Params (M) |  BLEU-4  |  CIDER   |            Config             |                                                 Download                                                 |
| :---------------------------- | :--------: | :------: | :------: | :---------------------------: | :------------------------------------------------------------------------------------------------------: |
| `otter-9b_3rdparty_caption`\* |  8220.45   | Upcoming | Upcoming | [config](otter-9b_caption.py) | [model](https://download.openmmlab.com/mmclassification/v1/otter/otter-9b-adapter_20230613-51c5be8d.pth) |

*Models with * are converted from the [official repo](https://github.com/Luodian/Otter/tree/main). The config files of these models are only for inference. We haven't reproduce the training results.*

### Visual Question Answering on VQAv2

| Model                     | Params (M) | Accuracy |          Config           |                                                 Download                                                 |
| :------------------------ | :--------: | :------: | :-----------------------: | :------------------------------------------------------------------------------------------------------: |
| `otter-9b_3rdparty_vqa`\* |  8220.45   | Upcoming | [config](otter-9b_vqa.py) | [model](https://download.openmmlab.com/mmclassification/v1/otter/otter-9b-adapter_20230613-51c5be8d.pth) |

*Models with * are converted from the [official repo](https://github.com/Luodian/Otter/tree/main). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@article{li2023otter,
  title={Otter: A Multi-Modal Model with In-Context Instruction Tuning},
  author={Li, Bo and Zhang, Yuanhan and Chen, Liangyu and Wang, Jinghao and Yang, Jingkang and Liu, Ziwei},
  journal={arXiv preprint arXiv:2305.03726},
  year={2023}
}

@article{li2023mimicit,
    title={MIMIC-IT: Multi-Modal In-Context Instruction Tuning},
    author={Bo Li and Yuanhan Zhang and Liangyu Chen and Jinghao Wang and Fanyi Pu and Jingkang Yang and Chunyuan Li and Ziwei Liu},
    year={2023},
    eprint={2306.05425},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
