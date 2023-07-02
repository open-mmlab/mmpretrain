# MiniGPT4

> [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://arxiv.org/abs/2305.06500)

<!-- [ALGORITHM] -->

## Abstract

Large-scale pre-training and instruction tuning have been successful at creating general-purpose language models with broad competence. However, building general-purpose vision-language models is challenging due to the rich input distributions and task diversity resulting from the additional visual input. Although
vision-language pretraining has been widely studied, vision-language instruction tuning remains under-explored. In this paper, we conduct a systematic and comprehensive study on vision-language instruction tuning based on the pretrained BLIP-2 models. We gather 26 publicly available datasets, covering a wide variety of tasks and capabilities, and transform them into instruction tuning format. Additionally, we introduce an instruction-aware Query Transformer, which extracts informative features tailored to the given instruction. Trained on 13 held-in datasets, InstructBLIP attains state-of-the-art zero-shot performance across all 13 held-out datasets, substantially outperforming BLIP-2 and larger Flamingo models. Our models also lead to state-of-the-art performance when finetuned on individual downstream tasks (e.g., 90.7% accuracy on ScienceQA questions with image contexts). Furthermore, we qualitatively demonstrate the advantages of InstructBLIP over concurrent multimodal models. All InstructBLIP models are open-sourced.

<div align=center>
<img src="https://github.com/open-mmlab/mmpretrain/assets/48375204/4211e0d8-951f-48d0-b81d-34be2e777390" width="80%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Use the model**

```python
from mmpretrain import inference_model

result = inference_model('instructblip-vicuna7b_3rdparty-zeroshot_caption', 'demo/cat-dog.png')
print(result)
# {'pred_caption': 'a blanket next to each other in the grass\na cute puppy and kitten wallpapers'}
```

<!-- [TABS-END] -->

## Models and results

For Vicuna model, please refer to [MiniGPT-4 page](https://github.com/Vision-CAIR/MiniGPT-4) for preparation guidelines.

### Pretrained models

| Model                           | Params (M) | Flops (G) |                  Config                  |                                                    Download                                                    |
| :------------------------------ | :--------: | :-------: | :--------------------------------------: | :------------------------------------------------------------------------------------------------------------: |
| `instructblip-vicuna7b_3rdparty-zeroshot_caption`\* |  8121.32   |    N/A    | [config](instructblip-vicuna7b_8xb32_caption.py) | [model]() |

*Models with * are converted from the [official repo](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@article{dai2023instructblip,
  title={InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning},
  author={Dai, Wenliang and Li, Junnan and Li, Dongxu and Tiong, Anthony Meng Huat and Zhao, Junqi and Wang, Weisheng and Li, Boyang and Fung, Pascale and Hoi, Steven},
  journal={arXiv preprint arXiv:2305.06500},
  year={2023}
}
```
