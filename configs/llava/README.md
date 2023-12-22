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

**Use the model**

```python
import torch
from mmpretrain import get_model, inference_model

out = inference_model('llava-7b-v1_caption', 'demo/cat-dog.png', device='cuda')
print(out)
# {'pred_caption': 'In the image, there are two cats sitting on a blanket.'}
```

<!-- [TABS-END] -->

## Models and results

### Image Caption on COCO

| Model                   | Params (M) |               Config               |                                                    Download                                                     |
| :---------------------- | :--------: | :--------------------------------: | :-------------------------------------------------------------------------------------------------------------: |
| `llava-7b-v1_caption`   |  7045.82   |  [config](llava-7b-v1_caption.py)  |  [ckpt](https://download.openmmlab.com/mmclassification/v1/llava/llava-7b-v1_liuhaotian_20231025-c9e119b6.pth)  |
| `llava-7b-v1.5_caption` |  7062.90   | [config](llava-7b-v1.5_caption.py) | [ckpt](https://download.openmmlab.com/mmclassification/v1/llava/llava-7b-v1.5_liuhaotian_20231025-5828aa5a.pth) |
| `llava-7b-v1.5_vqa`     |  7062.90   |   [config](llava-7b-v1.5_vqa.py)   | [ckpt](https://download.openmmlab.com/mmclassification/v1/llava/llava-7b-v1.5_liuhaotian_20231025-5828aa5a.pth) |

## Citation

```bibtex
@misc{liu2023llava,
      title={Visual Instruction Tuning},
      author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
      publisher={arXiv:2304.08485},
      year={2023},
}
```
