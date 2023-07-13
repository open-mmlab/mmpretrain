# MiniGPT4

> [MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models](https://arxiv.org/abs/2304.10592)

<!-- [ALGORITHM] -->

## Abstract

The recent GPT-4 has demonstrated extraordinary multi-modal abilities, such as directly generating websites from handwritten text and identifying humorous elements within images. These features are rarely observed in previous vision-language models. We believe the primary reason for GPT-4's advanced multi-modal generation capabilities lies in the utilization of a more advanced large language model (LLM). To examine this phenomenon, we present MiniGPT-4, which aligns a frozen visual encoder with a frozen LLM, Vicuna, using just one projection layer. Our findings reveal that MiniGPT-4 possesses many capabilities similar to those exhibited by GPT-4 like detailed image description generation and website creation from hand-written drafts. Furthermore, we also observe other emerging capabilities in MiniGPT-4, including writing stories and poems inspired by given images, providing solutions to problems shown in images, teaching users how to cook based on food photos, etc. In our experiment, we found that only performing the pretraining on raw image-text pairs could produce unnatural language outputs that lack coherency including repetition and fragmented sentences. To address this problem, we curate a high-quality, well-aligned dataset in the second stage to finetune our model using a conversational template. This step proved crucial for augmenting the model's generation reliability and overall usability. Notably, our model is highly computationally efficient, as we only train a projection layer utilizing approximately 5 million aligned image-text pairs. Our code, pre-trained model, and collected dataset are available at https://minigpt-4.github.io/.

<div align=center>
<img src="https://github.com/open-mmlab/mmpretrain/assets/36138628/1d8f328d-6c91-493e-8992-29e84a0fc3c8" width="80%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Use the model**

```python
from mmpretrain import inference_model

result = inference_model('minigpt-4_vicuna-7b_caption', 'demo/cat-dog.png')
print(result)
# {'pred_caption': 'This image shows a small dog and a kitten sitting on a blanket in a field of flowers. The dog is looking up at the kitten with a playful expression on its face. The background is a colorful striped blanket, and there are flowers all around them. The image is well composed with the two animals sitting in the center of the frame, surrounded by the flowers and blanket.'}
```

<!-- [TABS-END] -->

## Models and results

For Vicuna model, please refer to [MiniGPT-4 page](https://github.com/Vision-CAIR/MiniGPT-4) for preparation guidelines.

### Pretrained models

| Model                           | Params (M) | Flops (G) |                  Config                  |                                                    Download                                                    |
| :------------------------------ | :--------: | :-------: | :--------------------------------------: | :------------------------------------------------------------------------------------------------------------: |
| `minigpt-4_vicuna-7b_caption`\* |  8121.32   |    N/A    | [config](minigpt-4_vicuna-7b_caption.py) | [model](https://download.openmmlab.com/mmpretrain/v1.0/minigpt4/minigpt-4_linear-projection_20230615-714b5f52.pth) |

*Models with * are converted from the [official repo](https://github.com/Vision-CAIR/MiniGPT-4/tree/main). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@article{zhu2023minigpt,
  title={MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models},
  author={Zhu, Deyao and Chen, Jun and Shen, Xiaoqian and Li, Xiang and Elhoseiny, Mohamed},
  journal={arXiv preprint arXiv:2304.10592},
  year={2023}
}
```
