# Changelog

## v0.18.0(30/11/2021)

### Highlights

- Support MLP-Mixer backbone and provide pre-trained checkpoints.
- Add a tool to visualize the learning rate curve of the training phase. Welcome to use with the [tutorial](https://mmclassification.readthedocs.io/en/latest/tools/visualization.html#learning-rate-schedule-visualization)!

### New Features

- Add MLP Mixer Backbone. ([#528](https://github.com/open-mmlab/mmclassification/pull/528), [#539](https://github.com/open-mmlab/mmclassification/pull/539))
- Support positive weights in BCE. ([#516](https://github.com/open-mmlab/mmclassification/pull/516))
- Add a tool to visualize learning rate in each iterations ([#498](https://github.com/open-mmlab/mmclassification/pull/498))

### Improvements

- Use CircleCI to do unit tests. ([#567](https://github.com/open-mmlab/mmclassification/pull/567))
- Focal loss for single label tasks ([#548](https://github.com/open-mmlab/mmclassification/pull/548))
- Remove useless `import_modules_from_string`. ([#544](https://github.com/open-mmlab/mmclassification/pull/544))
- Rename config files according to the config name standard. ([#508](https://github.com/open-mmlab/mmclassification/pull/508))
- Use `reset_classifier` to remove head of timm backbones. ([#534](https://github.com/open-mmlab/mmclassification/pull/534))
- Support passing arguments to loss from head. ([#523](https://github.com/open-mmlab/mmclassification/pull/523))
- Refactor `Resize` transform and add `Pad` transform. ([#506](https://github.com/open-mmlab/mmclassification/pull/506))
- Update mmcv dependency version ([#509](https://github.com/open-mmlab/mmclassification/pull/509))

### Bug Fixes

- Fix bug when using `ClassBalancedDataset` ([#555](https://github.com/open-mmlab/mmclassification/pull/555))
- Fix a bug when using iter-based runner with 'val' workflow ([#542](https://github.com/open-mmlab/mmclassification/pull/542))
- Fix interpolation method checking in `Resize` ([#547](https://github.com/open-mmlab/mmclassification/pull/547))
- Fix a bug when load checkpoints in mulit-GPUs environment. ([#527](https://github.com/open-mmlab/mmclassification/pull/527))
- Fix an error on indexing scalar metrics in `analyze_result.py` ([#518](https://github.com/open-mmlab/mmclassification/pull/518))
- Fix wrong condition judgment in `analyze_logs.py` and prevent empty curve. ([#510](https://github.com/open-mmlab/mmclassification/pull/510))

### Docs Update

- Fix vit config and model broken links ([#564](https://github.com/open-mmlab/mmclassification/pull/564))
- Add abstract and image for every paper. ([#546](https://github.com/open-mmlab/mmclassification/pull/546))
- Add mmflow and mim in banner and readme ([#543](https://github.com/open-mmlab/mmclassification/pull/543))
- Add schedule and runtime tutorial docs ([#499](https://github.com/open-mmlab/mmclassification/pull/499))
- Add the top-5 acc in ResNet-CIFAR README. ([#531](https://github.com/open-mmlab/mmclassification/pull/531))
- Fix TOC of `visualization.md` and add example images. ([#513](https://github.com/open-mmlab/mmclassification/pull/513))
- Use docs link of other projects and add MMCV docs. ([#511](https://github.com/open-mmlab/mmclassification/pull/511))

## v0.17.0(29/10/2021)

### Highlights

- Support Tokens-to-Token ViT backbone and Res2Net backbone. Welcome to use!
- Support ImageNet21k dataset.
- Add a pipeline visualization tool. Try it with the [tutorials](https://mmclassification.readthedocs.io/en/latest/tools/visualization.html#pipeline-visualization)!

### New Features

- Add Tokens-to-Token ViT backbone and converted checkpoints. ([#467](https://github.com/open-mmlab/mmclassification/pull/467))
- Add Res2Net backbone and converted weights. ([#465](https://github.com/open-mmlab/mmclassification/pull/465))
- Support ImageNet21k dataset. ([#461](https://github.com/open-mmlab/mmclassification/pull/461))
- Support seesaw loss. ([#500](https://github.com/open-mmlab/mmclassification/pull/500))
- Add a pipeline visualization tool. ([#406](https://github.com/open-mmlab/mmclassification/pull/406))
- Add a tool to find broken files. ([#482](https://github.com/open-mmlab/mmclassification/pull/482))
- Add a tool to test TorchServe. ([#468](https://github.com/open-mmlab/mmclassification/pull/468))

### Improvements

- Refator Vision Transformer. ([#395](https://github.com/open-mmlab/mmclassification/pull/395))
- Use context manager to reuse matplotlib figures. ([#432](https://github.com/open-mmlab/mmclassification/pull/432))

### Bug Fixes

- Remove `DistSamplerSeedHook` if use `IterBasedRunner`. ([#501](https://github.com/open-mmlab/mmclassification/pull/501))
- Set the priority of `EvalHook` to "LOW" to avoid a bug when using `IterBasedRunner`. ([#488](https://github.com/open-mmlab/mmclassification/pull/488))
- Fix a wrong parameter of `get_root_logger` in `apis/train.py`. ([#486](https://github.com/open-mmlab/mmclassification/pull/486))
- Fix version check in dataset builder. ([#474](https://github.com/open-mmlab/mmclassification/pull/474))

### Docs Update

- Add English Colab tutorials and update Chinese Colab tutorials. ([#483](https://github.com/open-mmlab/mmclassification/pull/483), [#497](https://github.com/open-mmlab/mmclassification/pull/497))
- Add tutuorial for config files. ([#487](https://github.com/open-mmlab/mmclassification/pull/487))
- Add model-pages in Model Zoo. ([#480](https://github.com/open-mmlab/mmclassification/pull/480))
- Add code-spell pre-commit hook and fix a large mount of typos. ([#470](https://github.com/open-mmlab/mmclassification/pull/470))

## v0.16.0(30/9/2021)

### Highlights

- We have improved compatibility with downstream repositories like MMDetection and MMSegmentation. We will add some examples about how to use our backbones in MMDetection.
- Add RepVGG backbone and checkpoints. Welcome to use it!
- Add timm backbones wrapper, now you can simply use backbones of pytorch-image-models in MMClassification!

### New Features

- Add RepVGG backbone and checkpoints. ([#414](https://github.com/open-mmlab/mmclassification/pull/414))
- Add timm backbones wrapper. ([#427](https://github.com/open-mmlab/mmclassification/pull/427))

### Improvements

- Fix TnT compatibility and verbose warning. ([#436](https://github.com/open-mmlab/mmclassification/pull/436))
- Support setting `--out-items` in `tools/test.py`.  ([#437](https://github.com/open-mmlab/mmclassification/pull/437))
- Add datetime info and saving model using torch<1.6 format. ([#439](https://github.com/open-mmlab/mmclassification/pull/439))
- Improve downstream repositories compatibility. ([#421](https://github.com/open-mmlab/mmclassification/pull/421))
- Rename the option `--options` to `--cfg-options` in some tools. ([#425](https://github.com/open-mmlab/mmclassification/pull/425))
- Add PyTorch 1.9 and Python 3.9 build workflow, and remove some CI. ([#422](https://github.com/open-mmlab/mmclassification/pull/422))

### Bug Fixes

- Fix format error in `test.py` when metric returns `np.ndarray`. ([#441](https://github.com/open-mmlab/mmclassification/pull/441))
- Fix `publish_model` bug if no parent of `out_file`. ([#463](https://github.com/open-mmlab/mmclassification/pull/463))
- Fix num_classes bug in pytorch2onnx.py. ([#458](https://github.com/open-mmlab/mmclassification/pull/458))
- Fix missing runtime requirement `packaging`. ([#459](https://github.com/open-mmlab/mmclassification/pull/459))
- Fix saving simplified model bug in ONNX export tool. ([#438](https://github.com/open-mmlab/mmclassification/pull/438))

### Docs Update

- Update `getting_started.md` and `install.md`. And rewrite `finetune.md`. ([#466](https://github.com/open-mmlab/mmclassification/pull/466))
- Use PyTorch style docs theme. ([#457](https://github.com/open-mmlab/mmclassification/pull/457))
- Update metafile and Readme. ([#435](https://github.com/open-mmlab/mmclassification/pull/435))
- Add `CITATION.cff`. ([#428](https://github.com/open-mmlab/mmclassification/pull/428))

## v0.15.0(31/8/2021)

### Highlights
- Support `hparams` argument in `AutoAugment` and `RandAugment` to provide hyperparameters for sub-policies.
- Support custom squeeze channels in `SELayer`.
- Support classwise weight in losses.

### New Features

- Add `hparams` argument in `AutoAugment` and `RandAugment` and some other improvement. ([#398](https://github.com/open-mmlab/mmclassification/pull/398))
- Support classwise weight in losses. ([#388](https://github.com/open-mmlab/mmclassification/pull/388))
- Enhance `SELayer` to support custom squeeze channels. ([#417](https://github.com/open-mmlab/mmclassification/pull/417))

### Code Refactor

- Better result visualization. ([#419](https://github.com/open-mmlab/mmclassification/pull/419))
- Use `post_process` function to handle pred result processing. ([#390](https://github.com/open-mmlab/mmclassification/pull/390))
- Update `digit_version` function. ([#402](https://github.com/open-mmlab/mmclassification/pull/402))
- Avoid albumentations to install both opencv and opencv-headless. ([#397](https://github.com/open-mmlab/mmclassification/pull/397))
- Avoid unnecessary listdir when building ImageNet. ([#396](https://github.com/open-mmlab/mmclassification/pull/396))
- Use dynamic mmcv download link in TorchServe dockerfile. ([#387](https://github.com/open-mmlab/mmclassification/pull/387))

### Docs Improvement

- Add readme of some algorithms and update meta yml. ([#418](https://github.com/open-mmlab/mmclassification/pull/418))
- Add Copyright information. ([#413](https://github.com/open-mmlab/mmclassification/pull/413))
- Fix typo 'metirc'. ([#411](https://github.com/open-mmlab/mmclassification/pull/411))
- Update QQ group QR code. ([#393](https://github.com/open-mmlab/mmclassification/pull/393))
- Add PR template and modify issue template. ([#380](https://github.com/open-mmlab/mmclassification/pull/380))

## v0.14.0(4/8/2021)

### Highlights
- Add transformer-in-transformer backbone and pretrain checkpoints, refers to [the paper](https://arxiv.org/abs/2103.00112).
- Add Chinese colab tutorial.
- Provide dockerfile to build mmcls dev docker image.

### New Features

- Add transformer in transformer backbone and pretrain checkpoints. ([#339](https://github.com/open-mmlab/mmclassification/pull/339))
- Support mim, welcome to use mim to manage your mmcls project. ([#376](https://github.com/open-mmlab/mmclassification/pull/376))
- Add Dockerfile. ([#365](https://github.com/open-mmlab/mmclassification/pull/365))
- Add ResNeSt configs. ([#332](https://github.com/open-mmlab/mmclassification/pull/332))

### Improvements

- Use the `presistent_works` option if available, to accelerate training. ([#349](https://github.com/open-mmlab/mmclassification/pull/349))
- Add Chinese ipynb tutorial. ([#306](https://github.com/open-mmlab/mmclassification/pull/306))
- Refactor unit tests. ([#321](https://github.com/open-mmlab/mmclassification/pull/321))
- Support to test mmdet inference with mmcls backbone. ([#343](https://github.com/open-mmlab/mmclassification/pull/343))
- Use zero as default value of `thrs` in metrics. ([#341](https://github.com/open-mmlab/mmclassification/pull/341))

### Bug Fixes

- Fix ImageNet dataset annotation file parse bug. ([#370](https://github.com/open-mmlab/mmclassification/pull/370))
- Fix docstring typo and init bug in ShuffleNetV1. ([#374](https://github.com/open-mmlab/mmclassification/pull/374))
- Use local ATTENTION registry to avoid conflict with other repositories. ([#376](https://github.com/open-mmlab/mmclassification/pull/375))
- Fix swin transformer config bug. ([#355](https://github.com/open-mmlab/mmclassification/pull/355))
- Fix `patch_cfg` argument bug in SwinTransformer. ([#368](https://github.com/open-mmlab/mmclassification/pull/368))
- Fix duplicate `init_weights` call in ViT init function. ([#373](https://github.com/open-mmlab/mmclassification/pull/373))
- Fix broken `_base_` link in a resnet config. ([#361](https://github.com/open-mmlab/mmclassification/pull/361))
- Fix vgg-19 model link missing. ([#363](https://github.com/open-mmlab/mmclassification/pull/363))

## v0.13.0(3/7/2021)

- Support Swin-Transformer backbone and add training configs for Swin-Transformer on ImageNet.

### New Features

- Support Swin-Transformer backbone and add training configs for Swin-Transformer on ImageNet. (#271)
- Add pretained model of RegNetX. (#269)
- Support adding custom hooks in config file. (#305)
- Improve and add Chinese translation of `CONTRIBUTING.md` and all tools tutorials. (#320)
- Dump config before training. (#282)
- Add torchscript and torchserve deployment tools. (#279, #284)

### Improvements

- Improve test tools and add some new tools. (#322)
- Correct MobilenetV3 backbone structure and add pretained models. (#291)
- Refactor `PatchEmbed` and `HybridEmbed` as independent components. (#330)
- Refactor mixup and cutmix as `Augments` to support more functions. (#278)
- Refactor weights initialization method. (#270, #318, #319)
- Refactor `LabelSmoothLoss` to support multiple calculation formulas. (#285)

### Bug Fixes

- Fix bug for CPU training. (#286)
- Fix missing test data when `num_imgs` can not be evenly divided by `num_gpus`. (#299)
- Fix build compatible with pytorch v1.3-1.5. (#301)
- Fix `magnitude_std` bug in `RandAugment`. (#309)
- Fix bug when `samples_per_gpu` is 1. (#311)

## v0.12.0(3/6/2021)

- Finish adding Chinese tutorials and build Chinese documentation on readthedocs.
- Update ResNeXt checkpoints and ResNet checkpoints on CIFAR.

### New Features

- Improve and add Chinese translation of `data_pipeline.md` and `new_modules.md`. (#265)
- Build Chinese translation on readthedocs. (#267)
- Add an argument efficientnet_style to `RandomResizedCrop` and `CenterCrop`. (#268)

### Improvements

- Only allow directory operation when rank==0 when testing. (#258)
- Fix typo in `base_head`. (#274)
- Update ResNeXt checkpoints. (#283)

### Bug Fixes

- Add attribute `data.test` in MNIST configs. (#264)
- Download CIFAR/MNIST dataset only on rank 0. (#273)
- Fix MMCV version compatibility. (#276)
- Fix CIFAR color channels bug and update checkpoints in model zoo. (#280)

## v0.11.1(21/5/2021)

- Refine `new_dataset.md` and add Chinese translation of `finture.md`, `new_dataset.md`.

### New Features

- Add `dim` argument for `GlobalAveragePooling`. (#236)
- Add random noise to `RandAugment` magnitude. (#240)
- Refine `new_dataset.md` and add Chinese translation of `finture.md`, `new_dataset.md`. (#243)

### Improvements

- Refactor arguments passing for Heads. (#239)
- Allow more flexible `magnitude_range` in `RandAugment`. (#249)
- Inherits MMCV registry so that in the future OpenMMLab repos like MMDet and MMSeg could directly use the backbones supported in MMCls. (#252)

### Bug Fixes

- Fix typo in `analyze_results.py`. (#237)
- Fix typo in unittests. (#238)
- Check if specified tmpdir exists when testing to avoid deleting existing data. (#242 & #258)
- Add missing config files in `MANIFEST.in`. (#250 & #255)
- Use temporary directory under shared directory to collect results to avoid unavailability of temporary directory for multi-node testing. (#251)

## v0.11.0(1/5/2021)

- Support cutmix trick.
- Support random augmentation.
- Add `tools/deployment/test.py` as a ONNX runtime test tool.
- Support ViT backbone and add training configs for ViT on ImageNet.
- Add Chinese `README.md` and some Chinese tutorials.

### New Features

- Support cutmix trick. (#198)
- Add `simplify` option in `pytorch2onnx.py`. (#200)
- Support random augmentation. (#201)
- Add config and checkpoint for training ResNet on CIFAR-100. (#208)
- Add `tools/deployment/test.py` as a ONNX runtime test tool. (#212)
- Support ViT backbone and add training configs for ViT on ImageNet. (#214)
- Add finetuning configs for ViT on ImageNet. (#217)
- Add `device` option to support training on CPU. (#219)
- Add Chinese `README.md` and some Chinese tutorials. (#221)
- Add `metafile.yml` in configs to support interaction with paper with code(PWC) and MMCLI. (#225)
- Upload configs and converted checkpoints for ViT fintuning on ImageNet. (#230)

### Improvements

- Fix `LabelSmoothLoss` so that label smoothing and mixup could be enabled at the same time. (#203)
- Add `cal_acc` option in `ClsHead`. (#206)
- Check `CLASSES` in checkpoint to avoid unexpected key error. (#207)
- Check mmcv version when importing mmcls to ensure compatibility. (#209)
- Update `CONTRIBUTING.md` to align with that in MMCV. (#210)
- Change tags to html comments in configs README.md. (#226)
- Clean codes in ViT backbone. (#227)
- Reformat `pytorch2onnx.md` tutorial. (#229)
- Update `setup.py` to support MMCLI. (#232)

### Bug Fixes

- Fix missing `cutmix_prob` in ViT configs. (#220)
- Fix backend for resize in ResNeXt configs. (#222)

## v0.10.0(1/4/2021)

- Support AutoAugmentation
- Add tutorials for installation and usage.

### New Features

- Add `Rotate` pipeline for data augmentation. (#167)
- Add `Invert` pipeline for data augmentation. (#168)
- Add `Color` pipeline for data augmentation. (#171)
- Add `Solarize` and `Posterize` pipeline for data augmentation. (#172)
- Support fp16 training. (#178)
- Add tutorials for installation and basic usage of MMClassification.(#176)
- Support `AutoAugmentation`, `AutoContrast`, `Equalize`, `Contrast`, `Brightness` and `Sharpness` pipelines for data augmentation. (#179)

### Improvements

- Support dynamic shape export to onnx. (#175)
- Release training configs and update model zoo for fp16 (#184)
- Use MMCV's EvalHook in MMClassification (#182)

### Bug Fixes

- Fix wrong naming in vgg config (#181)

## v0.9.0(1/3/2021)

- Implement mixup trick.
- Add a new tool to create TensorRT engine from ONNX, run inference and verify outputs in Python.

### New Features

- Implement mixup and provide configs of training ResNet50 using mixup. (#160)
- Add `Shear` pipeline for data augmentation. (#163)
- Add `Translate` pipeline for data augmentation. (#165)
- Add `tools/onnx2tensorrt.py` as a tool to create TensorRT engine from ONNX, run inference and verify outputs in Python. (#153)

### Improvements

- Add `--eval-options` in `tools/test.py` to support eval options override, matching the behavior of other open-mmlab projects. (#158)
- Support showing and saving painted results in `mmcls.apis.test` and `tools/test.py`, matching the behavior of other open-mmlab projects. (#162)

### Bug Fixes

- Fix configs for VGG, replace checkpoints converted from other repos with the ones trained by ourselves and upload the missing logs in the model zoo. (#161)

## v0.8.0(31/1/2021)

- Support multi-label task.
- Support more flexible metrics settings.
- Fix bugs.

### New Features

- Add evaluation metrics: mAP, CP, CR, CF1, OP, OR, OF1 for multi-label task. (#123)
- Add BCE loss for multi-label task. (#130)
- Add focal loss for multi-label task. (#131)
- Support PASCAL VOC 2007 dataset for multi-label task. (#134)
- Add asymmetric loss for multi-label task. (#132)
- Add analyze_results.py to select images for success/fail demonstration. (#142)
- Support new metric that calculates the total number of occurrences of each label. (#143)
- Support class-wise evaluation results. (#143)
- Add thresholds in eval_metrics. (#146)
- Add heads and a baseline config for multilabel task. (#145)

### Improvements

- Remove the models with 0 checkpoint and ignore the repeated papers when counting papers to gain more accurate model statistics. (#135)
- Add tags in README.md. (#137)
- Fix optional issues in docstring. (#138)
- Update stat.py to classify papers. (#139)
- Fix mismatched columns in README.md. (#150)
- Fix test.py to support more evaluation metrics. (#155)

### Bug Fixes

- Fix bug in VGG weight_init. (#140)
- Fix bug in 2 ResNet configs in which outdated heads were used. (#147)
- Fix bug of misordered height and width in `RandomCrop` and `RandomResizedCrop`. (#151)
- Fix missing `meta_keys` in `Collect`. (#149 & #152)

## v0.7.0(31/12/2020)

- Add more evaluation metrics.
- Fix bugs.

### New Features

- Remove installation of MMCV from requirements. (#90)
- Add 3 evaluation metrics: precision, recall and F-1 score. (#93)
- Allow config override during testing and inference with `--options`. (#91 & #96)

### Improvements

- Use `build_runner` to make runners more flexible. (#54)
- Support to get category ids in `BaseDataset`. (#72)
- Allow `CLASSES` override during `BaseDateset` initialization. (#85)
- Allow input image as ndarray during inference. (#87)
- Optimize MNIST config. (#98)
- Add config links in model zoo documentation. (#99)
- Use functions from MMCV to collect environment. (#103)
- Refactor config files so that they are now categorized by methods. (#116)
- Add README in config directory. (#117)
- Add model statistics. (#119)
- Refactor documentation in consistency with other MM repositories. (#126)

### Bug Fixes

- Add missing `CLASSES` argument to dataset wrappers. (#66)
- Fix slurm evaluation error during training. (#69)
- Resolve error caused by shape in `Accuracy`. (#104)
- Fix bug caused by extremely insufficient data in distributed sampler.(#108)
- Fix bug in `gpu_ids` in distributed training. (#107)
- Fix bug caused by extremely insufficient data in collect results during testing (#114)

## v0.6.0(11/10/2020)

- Support new method: ResNeSt and VGG.
- Support new dataset: CIFAR10.
- Provide new tools to do model inference, model conversion from pytorch to onnx.

### New Features

- Add model inference. (#16)
- Add pytorch2onnx. (#20)
- Add PIL backend for transform `Resize`. (#21)
- Add ResNeSt. (#25)
- Add VGG and its pretained models. (#27)
- Add CIFAR10 configs and models. (#38)
- Add albumentations transforms. (#45)
- Visualize results on image demo. (#58)

### Improvements

- Replace urlretrieve with urlopen in dataset.utils. (#13)
- Resize image according to its short edge. (#22)
- Update ShuffleNet config. (#31)
- Update pre-trained models for shufflenet_v2, shufflenet_v1, se-resnet50, se-resnet101. (#33)

### Bug Fixes

- Fix init_weights in `shufflenet_v2.py`. (#29)
- Fix the parameter `size` in test_pipeline. (#30)
- Fix the parameter in cosine lr schedule. (#32)
- Fix the convert tools for mobilenet_v2. (#34)
- Fix crash in CenterCrop transform when image is greyscale (#40)
- Fix outdated configs. (#53)
