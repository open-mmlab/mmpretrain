# CLIP

> [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on. We release our code and pre-trained model weights at this https URL.

<!-- [IMAGE] -->

<div align=center>
<img src="https://raw.githubusercontent.com/Scarecrow0/figures_cache/main/clip_main_fig.png" width="100%"/>
</div>

## Results and models

### ImageNet-1k

| Model                                           | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config                          | Download                                            |
| :---------------------------------------------- | --------: | -------: | --------: | --------: | :------------------------------ | :-------------------------------------------------- |
| vit_base_patch16_clip_384.openai_ft_in12k_in1k  |     86.57 |    49.37 |     86.87 |     98.05 | vit-base-p16_pt-64xb64_in1k-384 | vit_base_patch16_clip_384.openai_ft_in12k_in1k.bin  |
| vit_base_patch32_clip_384.openai_ft_in12k_in1k  |     88.22 |    12.66 |     85.13 |     97.42 | vit-base-p32_pt-64xb64_in1k-384 | vit_base_patch32_clip_384.openai_ft_in12k_in1k.bin  |
| vit_base_patch16_clip_224.laion2b_ft_in1k       |     86.57 |    16.86 |     85.49 |     97.59 | vit-base-p16_pt-64xb64_in1k-224 | vit_base_patch16_clip_224.laion2b_ft_in1k.bin       |
| vit_base_patch32_clip_224.laion2b_ft_in12k_in1k |     88.22 |     4.36 |     83.06 |     96.49 | vit-base-p32_pt-64xb64_in1k-224 | vit_base_patch32_clip_224.laion2b_ft_in12k_in1k.bin |
| vit_base_patch16_clip_224.laion2b_ft_in12k_in1k |     86.57 |    16.86 |     86.02 |     97.76 | vit-base-p16_pt-64xb64_in1k-224 | vit_base_patch16_clip_224.laion2b_ft_in12k_in1k.bin |
| vit_base_patch16_clip_224.openai_ft_in1k        |     86.57 |    16.86 |      85.3 |      97.5 | vit-base-p16_pt-64xb64_in1k-224 | vit_base_patch16_clip_224.openai_ft_in1k.bin        |
| vit_base_patch32_clip_224.openai_ft_in1k        |     88.22 |     4.36 |     81.77 |     95.89 | vit-base-p32_pt-64xb64_in1k-224 | vit_base_patch32_clip_224.openai_ft_in1k.bin        |
| vit_base_patch16_clip_384.laion2b_ft_in1k       |     86.57 |    49.37 |     86.52 |     97.97 | vit-base-p16_pt-64xb64_in1k-384 | vit_base_patch16_clip_384.laion2b_ft_in1k.bin       |
| vit_base_patch32_clip_384.laion2b_ft_in12k_in1k |     88.22 |    12.66 |     85.39 |     97.67 | vit-base-p32_pt-64xb64_in1k-384 | vit_base_patch32_clip_384.laion2b_ft_in12k_in1k.bin |
| vit_base_patch16_clip_384.openai_ft_in1k        |     86.57 |    49.37 |     86.25 |      97.9 | vit-base-p16_pt-64xb64_in1k-384 | vit_base_patch16_clip_384.openai_ft_in1k.bin        |
| vit_base_patch32_clip_448.laion2b_ft_in12k_in1k |     88.22 |     17.2 |     85.76 |     97.63 | vit-base-p32_pt-64xb64_in1k-448 | vit_base_patch32_clip_448.laion2b_ft_in12k_in1k.bin |
| vit_base_patch16_clip_384.laion2b_ft_in12k_in1k |     86.57 |    49.37 |     87.17 |     98.02 | vit-base-p16_pt-64xb64_in1k-384 | vit_base_patch16_clip_384.laion2b_ft_in12k_in1k.bin |
| vit_base_patch16_clip_224.openai_ft_in12k_in1k  |     86.57 |    16.86 |     85.99 |     97.72 | vit-base-p16_pt-64xb64_in1k-224 | vit_base_patch16_clip_224.openai_ft_in12k_in1k.bin  |
| vit_base_patch32_clip_224.laion2b_ft_in1k       |     88.22 |     4.36 |     82.46 |     96.12 | vit-base-p32_pt-64xb64_in1k-224 | vit_base_patch32_clip_224.laion2b_ft_in1k.bin       |

*Models with * are converted from the [official repo](https://github.com/rwightman/pytorch-image-models). The config files of these models are only for inference. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

## Citation

```bibtex
@misc{https://doi.org/10.48550/arxiv.2103.00020,
  doi = {10.48550/ARXIV.2103.00020},

  url = {https://arxiv.org/abs/2103.00020},

  author = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},

  keywords = {Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},

  title = {Learning Transferable Visual Models From Natural Language Supervision},

  publisher = {arXiv},

  year = {2021},

  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
