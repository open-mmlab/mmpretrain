# BLIP

> [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086)

<!-- [ALGORITHM] -->

## Abstract

Vision-Language Pre-training (VLP) has advanced the performance for many vision-language tasks. However, most existing pre-trained models only excel in either understanding-based tasks or generation-based tasks. Furthermore, performance improvement has been largely achieved by scaling up the dataset with noisy image-text pairs collected from the web, which is a suboptimal source of supervision. In this paper, we propose BLIP, a new VLP framework which transfers flexibly to both vision-language understanding and generation tasks. BLIP effectively utilizes the noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones. We achieve state-of-the-art results on a wide range of vision-language tasks, such as image-text retrieval (+2.7% in average recall@1), image captioning (+2.8% in CIDEr), and VQA (+1.6% in VQA score). BLIP also demonstrates strong generalization ability when directly transferred to video-language tasks in a zero-shot manner.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/236374275-94d2f94b-d9a7-4f12-b694-f15a2be00be6.png" width="90%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Use the model**

```python
from mmpretrain import inference_model

result = inference_model('blip-base_3rdparty_caption', 'demo/cat-dog.png')
print(result)
# {'pred_caption': 'a puppy and a cat sitting on a blanket'}
```

**Test Command**

Prepare your dataset according to the [docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset).

Test:

```shell
python tools/test.py configs/blip/blip-base_8xb32_caption.py https://download.openmmlab.com/mmclassification/v1/blip/blip-base_3rdparty_coco-caption_20230419-a5b71af3.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Caption on COCO

| Model                          | Params (M) | BLEU-4 | CIDER  |                 Config                 |                                                    Download                                                    |
| :----------------------------- | :--------: | :----: | :----: | :------------------------------------: | :------------------------------------------------------------------------------------------------------------: |
| `blip-base_3rdparty_caption`\* |   223.97   | 40.12  | 132.82 | [config](./blip-base_8xb32_caption.py) | [model](https://download.openmmlab.com/mmclassification/v1/blip/blip-base_3rdparty_coco-caption_20230419-a5b71af3.pth) |

### Image Caption on NoCaps

| Model                          | Params (M) | SPICE | CIDER  |                Config                 |                                                     Download                                                     |
| :----------------------------- | :--------: | :---: | :----: | :-----------------------------------: | :--------------------------------------------------------------------------------------------------------------: |
| `blip-base_3rdparty_caption`\* |   223.97   | 14.69 | 109.12 | [config](./blip-base_8xb32_nocaps.py) | [model](https://download.openmmlab.com/mmclassification/v1/blip/blip-base_3rdparty_coco-caption_20230419-a5b71af3.pth) |

### Image Caption on Flickr30k

| Model                          | Params (M) | SPICE | CIDER |                      Config                      |                                                Download                                                |
| :----------------------------- | :--------: | :---: | :---: | :----------------------------------------------: | :----------------------------------------------------------------------------------------------------: |
| `blip-base_3rdparty_caption`\* |   223.97   | 15.58 | 68.89 | [config](./blip-base_8xb32_caption_flickr30k.py) | [model](https://download.openmmlab.com/mmclassification/v1/blip/blip-base_3rdparty_coco-caption_20230419-a5b71af3.pth) |

### Visual Grounding on RefCOCO

| Model                     | Params (M) | Accuracy (testA) | Accuracy (testB) |                Config                |                                             Download                                              |
| :------------------------ | :--------: | :--------------: | :--------------: | :----------------------------------: | :-----------------------------------------------------------------------------------------------: |
| `blip-base_8xb16_refcoco` |   498.49   |      86.14       |      77.33       | [config](blip-base_8xb16_refcoco.py) | [model](https://download.openmmlab.com/mmclassification/v1/blip/blip-base_8xb16_refcoco_20230508-d2d10f4c.pth) \| [log](https://download.openmmlab.com/mmclassification/v1/blip/blip-base_8xb16_refcoco_20230508-d2d10f4c.json) |

### Visual Question Answering on VQAv2

| Model                      | Params (M) | Accuracy |               Config               |                                                       Download                                                        |
| :------------------------- | :--------: | :------: | :--------------------------------: | :-------------------------------------------------------------------------------------------------------------------: |
| `blip-base_3rdparty_vqa`\* |   361.48   |  78.20   | [config](./blip-base_8xb32_vqa.py) | [model](https://download.openmmlab.com/mmclassification/v1/blip/blip-base_3rdparty-capflit_vqa_20230505-81488941.pth) |

### Visual Question Answering on OK-VQA

| Model                      | Params (M) | Accuracy |                Config                |                                                       Download                                                        |
| :------------------------- | :--------: | :------: | :----------------------------------: | :-------------------------------------------------------------------------------------------------------------------: |
| `blip-base_3rdparty_vqa`\* |   361.48   |  40.59#  | [config](./blip-base_8xb32_okvqa.py) | [model](https://download.openmmlab.com/mmclassification/v1/blip/blip-base_3rdparty-capflit_vqa_20230505-81488941.pth) |

### Visual Question Answering on OCR-VQA

| Model                      | Params (M) | Accuracy |                Config                 |                                                       Download                                                        |
| :------------------------- | :--------: | :------: | :-----------------------------------: | :-------------------------------------------------------------------------------------------------------------------: |
| `blip-base_3rdparty_vqa`\* |   361.48   |  28.30#  | [config](./blip-base_8xb32_ocrvqa.py) | [model](https://download.openmmlab.com/mmclassification/v1/blip/blip-base_3rdparty-capflit_vqa_20230505-81488941.pth) |

### Image-To-Text Retrieval on COCO

| Model                            | Params (M) | Recall@1 | Recall@5 |                  Config                  |                                                Download                                                |
| :------------------------------- | :--------: | :------: | :------: | :--------------------------------------: | :----------------------------------------------------------------------------------------------------: |
| `blip-base_3rdparty_retrieval`\* |   447.49   |  82.52   |  95.34   | [config](./blip-base_8xb32_retrieval.py) | [model](https://download.openmmlab.com/mmclassification/v1/blip/blip-base_3rdparty_coco-retrieval_20230419-a1804d2c.pth) |

### Text-To-Image Retrieval on COCO

| Model                            | Params (M) | Recall@1 | Recall@5 |                  Config                  |                                                Download                                                |
| :------------------------------- | :--------: | :------: | :------: | :--------------------------------------: | :----------------------------------------------------------------------------------------------------: |
| `blip-base_3rdparty_retrieval`\* |   447.49   |  64.82   |  86.28   | [config](./blip-base_8xb32_retrieval.py) | [model](https://download.openmmlab.com/mmclassification/v1/blip/blip-base_3rdparty_coco-retrieval_20230419-a1804d2c.pth) |

### Image-To-Text Retrieval on Flickr30k

| Model                            | Params (M) | Recall@1 | Recall@5 |                       Config                       |                                           Download                                           |
| :------------------------------- | :--------: | :------: | :------: | :------------------------------------------------: | :------------------------------------------------------------------------------------------: |
| `blip-base_3rdparty_retrieval`\* |   447.49   |  95.10#  |  99.60#  | [config](./blip-base_8xb32_retrieval_flickr30k.py) | [model](https://download.openmmlab.com/mmclassification/v1/blip/blip-base_3rdparty_coco-retrieval_20230419-a1804d2c.pth) |

### Text-To-Image Retrieval on Flickr30k

| Model                            | Params (M) | Recall@1 | Recall@5 |                       Config                       |                                           Download                                           |
| :------------------------------- | :--------: | :------: | :------: | :------------------------------------------------: | :------------------------------------------------------------------------------------------: |
| `blip-base_3rdparty_retrieval`\* |   447.49   |  85.26#  |  96.58#  | [config](./blip-base_8xb32_retrieval_flickr30k.py) | [model](https://download.openmmlab.com/mmclassification/v1/blip/blip-base_3rdparty_coco-retrieval_20230419-a1804d2c.pth) |

### NLVR on NLVR2

| Model                       | Params (M) | Top-1 (%) |               Config                |                                                    Download                                                    |
| :-------------------------- | :--------: | :-------: | :---------------------------------: | :------------------------------------------------------------------------------------------------------------: |
| `blip-base_3rdparty_nlvr`\* |   259.37   |   82.33   | [config](./blip-base_8xb32_nlvr.py) | [model](https://download.openmmlab.com/mmclassification/v1/blip/blip-base_3rdparty_nlvr_20230427-3b14d33f.pth) |

*Models with * are converted from the [official repo](https://github.com/salesforce/LAVIS). The config files of these models are only for inference. We haven't reproduce the training results.*

*Results with # denote zero-shot evaluation. The corresponding model hasn't been finetuned on that dataset.*

## Citation

```bibtex
@inproceedings{li2022blip,
      title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
      author={Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi},
      year={2022},
      booktitle={ICML},
}
```
