# Flamingo

> [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)

<!-- [ALGORITHM] -->

## Abstract

Building models that can be rapidly adapted to novel tasks using only a handful of annotated examples is an open challenge for multimodal machine learning research. We introduce Flamingo, a family of Visual Language Models (VLM) with this ability. We propose key architectural innovations to: (i) bridge powerful pretrained vision-only and language-only models, (ii) handle sequences of arbitrarily interleaved visual and textual data, and (iii) seamlessly ingest images or videos as inputs. Thanks to their flexibility, Flamingo models can be trained on large-scale multimodal web corpora containing arbitrarily interleaved text and images, which is key to endow them with in-context few-shot learning capabilities. We perform a thorough evaluation of our models, exploring and measuring their ability to rapidly adapt to a variety of image and video tasks. These include open-ended tasks such as visual question-answering, where the model is prompted with a question which it has to answer; captioning tasks, which evaluate the ability to describe a scene or an event; and close-ended tasks such as multiple-choice visual question-answering. For tasks lying anywhere on this spectrum, a single Flamingo model can achieve a new state of the art with few-shot learning, simply by prompting the model with task-specific examples. On numerous benchmarks, Flamingo outperforms models fine-tuned on thousands of times more task-specific data.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/236371424-3b9d2e16-3966-4c64-8b87-e33fd6348824.png" width="80%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Use the model**

```python
from mmpretrain import inference_model

result = inference_model('flamingo_3rdparty-zeroshot_caption', 'demo/cat-dog.png')
print(result)
# {'pred_caption': 'A dog and a cat are looking at each other. '}
```

**Test Command**

Prepare your dataset according to the [docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset).

Test:

```shell
python tools/test.py configs/flamingo/flamingo_zeroshot_caption.py https://download.openmmlab.com/mmclassification/v1/flamingo/openflamingo-9b-adapter_20230505-554310c8.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Caption on COCO

| Model                                  | Params (G) | CIDER |                 Config                 |                                                   Download                                                    |
| :------------------------------------- | :--------: | :---: | :------------------------------------: | :-----------------------------------------------------------------------------------------------------------: |
| `flamingo_3rdparty-zeroshot_caption`\* |   8.220    | 65.50 | [config](flamingo_zeroshot_caption.py) | [model](https://download.openmmlab.com/mmclassification/v1/flamingo/openflamingo-9b-adapter_20230505-554310c8.pth) |

*Models with * are converted from the [openflamingo](https://github.com/mlfoundations/open_flamingo). The config files of these models are only for inference. We haven't reproduce the training results.*

### Visual Question Answering on VQAv2

| Model                              | Params (G) | Accuracy |               Config               |                                                      Download                                                      |
| :--------------------------------- | :--------: | :------: | :--------------------------------: | :----------------------------------------------------------------------------------------------------------------: |
| `flamingo_3rdparty-zeroshot_vqa`\* |    8.22    |  43.50   | [config](flamingo_zeroshot_vqa.py) | [model](https://download.openmmlab.com/mmclassification/v1/flamingo/openflamingo-9b-adapter_20230505-554310c8.pth) |

*Models with * are converted from the [openflamingo](https://github.com/mlfoundations/open_flamingo). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@article{Alayrac2022FlamingoAV,
  title={Flamingo: a Visual Language Model for Few-Shot Learning},
  author={Jean-Baptiste Alayrac and Jeff Donahue and Pauline Luc and Antoine Miech and Iain Barr and Yana Hasson and Karel Lenc and Arthur Mensch and Katie Millican and Malcolm Reynolds and Roman Ring and Eliza Rutherford and Serkan Cabi and Tengda Han and Zhitao Gong and Sina Samangooei and Marianne Monteiro and Jacob Menick and Sebastian Borgeaud and Andy Brock and Aida Nematzadeh and Sahand Sharifzadeh and Mikolaj Binkowski and Ricardo Barreira and Oriol Vinyals and Andrew Zisserman and Karen Simonyan},
  journal={ArXiv},
  year={2022},
  volume={abs/2204.14198}
}
```

```bibtex
@software{anas_awadalla_2023_7733589,
  author = {Awadalla, Anas and Gao, Irena and Gardner, Joshua and Hessel, Jack and Hanafy, Yusuf and Zhu, Wanrong and Marathe, Kalyani and Bitton, Yonatan and Gadre, Samir and Jitsev, Jenia and Kornblith, Simon and Koh, Pang Wei and Ilharco, Gabriel and Wortsman, Mitchell and Schmidt, Ludwig},
  title = {OpenFlamingo},
  month        = mar,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.1.1},
  doi          = {10.5281/zenodo.7733589},
  url          = {https://doi.org/10.5281/zenodo.7733589}
}
```
