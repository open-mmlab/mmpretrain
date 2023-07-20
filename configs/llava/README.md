# LLaVA

> [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)

<!-- [ALGORITHM] -->

## Abstract

Instruction tuning large language models (LLMs) using machine-generated instruction-following data has improved zero-shot capabilities on new tasks, but the idea is less explored in the multimodal field. In this paper, we present the first attempt to use language-only GPT-4 to generate multimodal language-image instruction-following data. By instruction tuning on such generated data, we introduce LLaVA: Large Language and Vision Assistant, an end-to-end trained large multimodal model that connects a vision encoder and LLM for general-purpose visual and language understanding.Our early experiments show that LLaVA demonstrates impressive multimodel chat abilities, sometimes exhibiting the behaviors of multimodal GPT-4 on unseen images/instructions, and yields a 85.1% relative score compared with GPT-4 on a synthetic multimodal instruction-following dataset. When fine-tuned on Science QA, the synergy of LLaVA and GPT-4 achieves a new state-of-the-art accuracy of 92.53%. We make GPT-4 generated visual instruction tuning data, our model and code base publicly available.

<div align=center>
<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/26739999/246466979-c2f41b71-1de3-4da8-b20a-eaebe722c339.png" width="80%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Prepare the checkpoint**

According to the license of LLaMA, we cannot provide the merged checkpoint directly. Please use the below
script to download and get the merged the checkpoint.

```shell
python tools/model_converters/llava-delta2mmpre.py huggyllama/llama-7b liuhaotian/LLaVA-Lightning-7B-delta-v1-1 ./LLaVA-Lightning-7B-delta-v1-1.pth
```

**Use the model**

```python
import torch
from mmpretrain import get_model, inference_model

model = get_model('llava-7b-v1_caption', pretrained='MERGED_CHECKPOINT_PATH', device='cuda')
out = inference_model(model, 'demo/cat-dog.png')
print(out)
# {'pred_caption': 'In the image, there are two cats sitting on a blanket.'}
```

**Test Command**

Prepare your dataset according to the [docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset).

Test:

```shell
python tools/test.py configs/llava/llava-7b-v1_caption.py MERGED_CHECKPOINT_PATH
```

<!-- [TABS-END] -->

## Models and results

### Image Caption on COCO

| Model                 | Params (M) |  BLEU-4  |  CIDER   |              Config              |        Download        |
| :-------------------- | :--------: | :------: | :------: | :------------------------------: | :--------------------: |
| `llava-7b-v1_caption` |  7045.82   | Upcoming | Upcoming | [config](llava-7b-v1_caption.py) | See the above tutorial |

## Citation

```bibtex
@misc{liu2023llava,
      title={Visual Instruction Tuning},
      author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
      publisher={arXiv:2304.08485},
      year={2023},
}
```
