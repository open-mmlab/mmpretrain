## Evaluate the fine-tuned model on ImageNet variants

It's a common practice to evaluate the ImageNet-(1K, 21K) fine-tuned model on the ImageNet-1K validation set. This set
shares similar data distribution with the training set, but in real world, the inference data is more likely to share
different data distribution with the training set. To have a full evaluation of model's performance on
out-of-distribution datasets, research community introduces the ImageNet-variant datasets, which shares different data
distribution with that of ImageNet-(1K, 21K)., MMClassification supports evaluating the fine-tuned model on
[ImageNet-Adversarial (A)](https://arxiv.org/abs/1907.07174), [ImageNet-Rendition (R)](https://arxiv.org/abs/2006.16241),
[ImageNet-Corruption (C)](https://arxiv.org/abs/1903.12261), and [ImageNet-Sketch (S)](https://arxiv.org/abs/1905.13549).
You can follow these steps below to have a try:

### Prepare the datasets

You can download these datasets from [OpenDataLab](https://opendatalab.com/) and refactor these datasets under the
`data` folder in the following format:

```text
   imagenet-a
        ├── meta
        │   └── val.txt
        ├── val
   imagenet-r
        ├── meta
        │   └── val.txt
        ├── val/
   imagenet-s
        ├── meta
        │   └── val.txt
        ├── val/
   imagenet-c
        ├── meta
        │   └── val.txt
        ├── val/
```

`val.txt` is the annotation file, which should have the same style as that of ImageNet-1K. You can refer to
[prepare_dataset](https://mmclassification.readthedocs.io/en/1.x/user_guides/dataset_prepare.html) to generate the
annotation file.

### Configure the dataset and test evaluator

Once the dataset is ready, you need to configure the `dataset` and `test_evaluator`. You have two options to
write the default settings:

#### 1. Change the configuration file directly

There are few modifications to the config file, but change the `data_root` of the test dataloader and pass the
annotation file to the `test_evaluator`.

```python
# You should replace imagenet-x below with imagenet-c, imagenet-r, imagenet-a
# or imagenet-s
test_dataloader=dict(dataset=dict(data_root='data/imagenet-x'))
test_evaluator=dict(ann_file='data/imagenet-x/meta/val.txt')
```

#### 2. Overwrite the default settings from command line

For example, you can overwrite the default settings by passing `--cfg-options`:

```bash
--cfg-options test_dataloader.dataset.data_root='data/imagenet-x' \
              test_evaluator.ann_file='data/imagenet-x/meta/val.txt'
```

### Start test

This step is the common test step, you can follow this [guide](https://mmclassification.readthedocs.io/en/1.x/user_guides/train_test.html)
to evaluate your fine-tuned model on out-of-distribution datasets.

To make it easier, we also provide an off-the-shelf config files, for [ImageNet-C](https://github.com/open-mmlab/mmclassification/tree/dev-1.x/projects/example_project/ood_eval/vit_ood-eval_toy-example.py) and [ImageNet-C](https://github.com/open-mmlab/mmclassification/tree/dev-1.x/projects/example_project/ood_eval/vit_ood-eval_toy-example_imagnet-c.py), and you can have a try.
