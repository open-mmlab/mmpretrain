# Changelog

## v0.25.0(06/12/2022)

### Highlights

- Support MLU backend.

### New Features

- Support MLU backend. ([#1159](https://github.com/open-mmlab/mmclassification/pull/1159))
- Support Activation Checkpointing for ConvNeXt. ([#1152](https://github.com/open-mmlab/mmclassification/pull/1152))

### Improvements

- Add `dist_train_arm.sh` for ARM device and update NPU results. ([#1218](https://github.com/open-mmlab/mmclassification/pull/1218))

### Bug Fixes

- Fix a bug caused `MMClsWandbHook` stuck. ([#1242](https://github.com/open-mmlab/mmclassification/pull/1242))
- Fix the redundant `device_ids` in `tools/test.py`. ([#1215](https://github.com/open-mmlab/mmclassification/pull/1215))

### Docs Update

- Add version banner and version warning in master docs. ([#1216](https://github.com/open-mmlab/mmclassification/pull/1216))
- Update NPU support doc. ([#1198](https://github.com/open-mmlab/mmclassification/pull/1198))
- Fixed typo in `pytorch2torchscript.md`. ([#1173](https://github.com/open-mmlab/mmclassification/pull/1173))
- Fix typo in `miscellaneous.md`. ([#1137](https://github.com/open-mmlab/mmclassification/pull/1137))
- further detail for the doc for `ClassBalancedDataset`. ([#901](https://github.com/open-mmlab/mmclassification/pull/901))

## v0.24.1(31/10/2022)

### New Features

- Support mmcls with NPU backend. ([#1072](https://github.com/open-mmlab/mmclassification/pull/1072))

### Bug Fixes

- Fix performance issue in convnext DDP train. ([#1098](https://github.com/open-mmlab/mmclassification/pull/1098))

## v0.24.0(30/9/2022)

### Highlights

- Support HorNet, EfficientFormerm, SwinTransformer V2 and MViT backbones.
- Support Standford Cars dataset.

### New Features

- Support HorNet Backbone. ([#1013](https://github.com/open-mmlab/mmclassification/pull/1013))
- Support EfficientFormer. ([#954](https://github.com/open-mmlab/mmclassification/pull/954))
- Support Stanford Cars dataset. ([#893](https://github.com/open-mmlab/mmclassification/pull/893))
- Support CSRA head. ([#881](https://github.com/open-mmlab/mmclassification/pull/881))
- Support Swin Transform V2. ([#799](https://github.com/open-mmlab/mmclassification/pull/799))
- Support MViT and add checkpoints. ([#924](https://github.com/open-mmlab/mmclassification/pull/924))

### Improvements

- [Improve] replace loop of progressbar in api/test. ([#878](https://github.com/open-mmlab/mmclassification/pull/878))
- [Enhance] RepVGG for YOLOX-PAI. ([#1025](https://github.com/open-mmlab/mmclassification/pull/1025))
- [Enhancement] Update VAN. ([#1017](https://github.com/open-mmlab/mmclassification/pull/1017))
- [Refactor] Re-write `get_sinusoid_encoding` from third-party implementation. ([#965](https://github.com/open-mmlab/mmclassification/pull/965))
- [Improve] Upgrade onnxsim to v0.4.0. ([#915](https://github.com/open-mmlab/mmclassification/pull/915))
- [Improve] Fixed typo in `RepVGG`. ([#985](https://github.com/open-mmlab/mmclassification/pull/985))
- [Improve] Using `train_step` instead of `forward` in PreciseBNHook ([#964](https://github.com/open-mmlab/mmclassification/pull/964))
- [Improve] Use `forward_dummy` to calculate FLOPS. ([#953](https://github.com/open-mmlab/mmclassification/pull/953))

### Bug Fixes

- Fix warning with `torch.meshgrid`. ([#860](https://github.com/open-mmlab/mmclassification/pull/860))
- Add matplotlib minimum version requriments. ([#909](https://github.com/open-mmlab/mmclassification/pull/909))
- val loader should not drop last by default. ([#857](https://github.com/open-mmlab/mmclassification/pull/857))
- Fix config.device bug in toturial. ([#1059](https://github.com/open-mmlab/mmclassification/pull/1059))
- Fix attenstion clamp max params ([#1034](https://github.com/open-mmlab/mmclassification/pull/1034))
- Fix device mismatch in Swin-v2. ([#976](https://github.com/open-mmlab/mmclassification/pull/976))
- Fix the output position of Swin-Transformer. ([#947](https://github.com/open-mmlab/mmclassification/pull/947))

### Docs Update

- Fix typo in config.md. ([#827](https://github.com/open-mmlab/mmclassification/pull/827))
- Add version for torchvision to avoide error. ([#903](https://github.com/open-mmlab/mmclassification/pull/903))
- Fixed typo for `--out-dir` option of analyze_results.py. ([#898](https://github.com/open-mmlab/mmclassification/pull/898))
- Refine the docstring of RegNet ([#935](https://github.com/open-mmlab/mmclassification/pull/935))

## v0.23.2(28/7/2022)

### New Features

- Support MPS device. ([#894](https://github.com/open-mmlab/mmclassification/pull/894))

### Bug Fixes

- Fix a bug in Albu which caused crashing. ([#918](https://github.com/open-mmlab/mmclassification/pull/918))

## v0.23.1(2/6/2022)

### New Features

- Dedicated MMClsWandbHook for MMClassification (Weights and Biases Integration) ([#764](https://github.com/open-mmlab/mmclassification/pull/764))

### Improvements

- Use mdformat instead of markdownlint to format markdown. ([#844](https://github.com/open-mmlab/mmclassification/pull/844))

### Bug Fixes

- Fix wrong `--local_rank`.

### Docs Update

- Update install tutorials. ([#854](https://github.com/open-mmlab/mmclassification/pull/854))
- Fix wrong link in README. ([#835](https://github.com/open-mmlab/mmclassification/pull/835))

## v0.23.0(1/5/2022)

### New Features

- Support DenseNet. ([#750](https://github.com/open-mmlab/mmclassification/pull/750))
- Support VAN. ([#739](https://github.com/open-mmlab/mmclassification/pull/739))

### Improvements

- Support training on IPU and add fine-tuning configs of ViT. ([#723](https://github.com/open-mmlab/mmclassification/pull/723))

### Docs Update

- New style API reference, and easier to use! Welcome [view it](https://mmclassification.readthedocs.io/en/master/api/models.html). ([#774](https://github.com/open-mmlab/mmclassification/pull/774))

## v0.22.1(15/4/2022)

### New Features

- [Feature] Support resize relative position embedding in `SwinTransformer`. ([#749](https://github.com/open-mmlab/mmclassification/pull/749))
- [Feature] Add PoolFormer backbone and checkpoints. ([#746](https://github.com/open-mmlab/mmclassification/pull/746))

### Improvements

- [Enhance] Improve CPE performance by reduce memory copy. ([#762](https://github.com/open-mmlab/mmclassification/pull/762))
- [Enhance] Add extra dataloader settings in configs. ([#752](https://github.com/open-mmlab/mmclassification/pull/752))

## v0.22.0(30/3/2022)

### Highlights

- Support a series of CSP Network, such as CSP-ResNet, CSP-ResNeXt and CSP-DarkNet.
- A new `CustomDataset` class to help you build dataset of yourself!
- Support ConvMixer, RepMLP and new dataset - CUB dataset.

### New Features

- [Feature] Add CSPNet and backbone and checkpoints ([#735](https://github.com/open-mmlab/mmclassification/pull/735))
- [Feature] Add `CustomDataset`. ([#738](https://github.com/open-mmlab/mmclassification/pull/738))
- [Feature] Add diff seeds to diff ranks.  ([#744](https://github.com/open-mmlab/mmclassification/pull/744))
- [Feature] Support ConvMixer. ([#716](https://github.com/open-mmlab/mmclassification/pull/716))
- [Feature] Our `dist_train` & `dist_test` tools support distributed training on multiple machines. ([#734](https://github.com/open-mmlab/mmclassification/pull/734))
- [Feature] Add RepMLP backbone and checkpoints. ([#709](https://github.com/open-mmlab/mmclassification/pull/709))
- [Feature] Support CUB dataset. ([#703](https://github.com/open-mmlab/mmclassification/pull/703))
- [Feature] Support ResizeMix. ([#676](https://github.com/open-mmlab/mmclassification/pull/676))

### Improvements

- [Enhance] Use `--a-b` instead of `--a_b` in arguments. ([#754](https://github.com/open-mmlab/mmclassification/pull/754))
- [Enhance] Add `get_cat_ids` and `get_gt_labels` to KFoldDataset. ([#721](https://github.com/open-mmlab/mmclassification/pull/721))
- [Enhance] Set torch seed in `worker_init_fn`. ([#733](https://github.com/open-mmlab/mmclassification/pull/733))

### Bug Fixes

- [Fix] Fix the discontiguous output feature map of ConvNeXt. ([#743](https://github.com/open-mmlab/mmclassification/pull/743))

### Docs Update

- [Docs] Add brief installation steps in README for copy&paste. ([#755](https://github.com/open-mmlab/mmclassification/pull/755))
- [Docs] fix logo url link from mmocr to mmcls. ([#732](https://github.com/open-mmlab/mmclassification/pull/732))

## v0.21.0(04/03/2022)

### Highlights

- Support ResNetV1c and Wide-ResNet, and provide pre-trained models.
- Support dynamic input shape for ViT-based algorithms. Now our ViT, DeiT, Swin-Transformer and T2T-ViT support forwarding with any input shape.
- Reproduce training results of DeiT. And our DeiT-T and DeiT-S have higher accuracy comparing with the official weights.

### New Features

- Add ResNetV1c. ([#692](https://github.com/open-mmlab/mmclassification/pull/692))
- Support Wide-ResNet. ([#715](https://github.com/open-mmlab/mmclassification/pull/715))
- Support gem pooling ([#677](https://github.com/open-mmlab/mmclassification/pull/677))

### Improvements

- Reproduce training results of DeiT. ([#711](https://github.com/open-mmlab/mmclassification/pull/711))
- Add ConvNeXt pretrain models on ImageNet-1k. ([#707](https://github.com/open-mmlab/mmclassification/pull/707))
- Support dynamic input shape for ViT-based algorithms. ([#706](https://github.com/open-mmlab/mmclassification/pull/706))
- Add `evaluate` function for ConcatDataset. ([#650](https://github.com/open-mmlab/mmclassification/pull/650))
- Enhance vis-pipeline tool. ([#604](https://github.com/open-mmlab/mmclassification/pull/604))
- Return code 1 if scripts runs failed. ([#694](https://github.com/open-mmlab/mmclassification/pull/694))
- Use PyTorch official `one_hot` to implement `convert_to_one_hot`. ([#696](https://github.com/open-mmlab/mmclassification/pull/696))
- Add a new pre-commit-hook to automatically add a copyright. ([#710](https://github.com/open-mmlab/mmclassification/pull/710))
- Add deprecation message for deploy tools. ([#697](https://github.com/open-mmlab/mmclassification/pull/697))
- Upgrade isort pre-commit hooks. ([#687](https://github.com/open-mmlab/mmclassification/pull/687))
- Use `--gpu-id` instead of `--gpu-ids` in non-distributed multi-gpu training/testing. ([#688](https://github.com/open-mmlab/mmclassification/pull/688))
- Remove deprecation. ([#633](https://github.com/open-mmlab/mmclassification/pull/633))

### Bug Fixes

- Fix Conformer forward with irregular input size. ([#686](https://github.com/open-mmlab/mmclassification/pull/686))
- Add `dist.barrier` to fix a bug in directory checking. ([#666](https://github.com/open-mmlab/mmclassification/pull/666))

## v0.20.1(07/02/2022)

### Bug Fixes

- Fix the MMCV dependency version.

## v0.20.0(30/01/2022)

### Highlights

- Support K-fold cross-validation. The tutorial will be released later.
- Support HRNet, ConvNeXt, Twins and EfficientNet.
- Support model conversion from PyTorch to Core-ML by a tool.

### New Features

- Support K-fold cross-validation. ([#563](https://github.com/open-mmlab/mmclassification/pull/563))
- Support HRNet and add pre-trained models. ([#660](https://github.com/open-mmlab/mmclassification/pull/660))
- Support ConvNeXt and add pre-trained models. ([#670](https://github.com/open-mmlab/mmclassification/pull/670))
- Support Twins and add pre-trained models. ([#642](https://github.com/open-mmlab/mmclassification/pull/642))
- Support EfficientNet and add pre-trained models.([#649](https://github.com/open-mmlab/mmclassification/pull/649))
- Support `features_only` option in `TIMMBackbone`. ([#668](https://github.com/open-mmlab/mmclassification/pull/668))
- Add conversion script from pytorch to Core-ML model. ([#597](https://github.com/open-mmlab/mmclassification/pull/597))

### Improvements

- New-style CPU training and inference. ([#674](https://github.com/open-mmlab/mmclassification/pull/674))
- Add setup multi-processing both in train and test. ([#671](https://github.com/open-mmlab/mmclassification/pull/671))
- Rewrite channel split operation in ShufflenetV2. ([#632](https://github.com/open-mmlab/mmclassification/pull/632))
- Deprecate the support for "python setup.py test". ([#646](https://github.com/open-mmlab/mmclassification/pull/646))
- Support single-label, softmax, custom eps by asymmetric loss. ([#609](https://github.com/open-mmlab/mmclassification/pull/609))
- Save class names in best checkpoint created by evaluation hook. ([#641](https://github.com/open-mmlab/mmclassification/pull/641))

### Bug Fixes

- Fix potential unexcepted behaviors if `metric_options` is not specified in multi-label evaluation. ([#647](https://github.com/open-mmlab/mmclassification/pull/647))
- Fix API changes in  `pytorch-grad-cam&gt;=1.3.7`. ([#656](https://github.com/open-mmlab/mmclassification/pull/656))
- Fix bug which breaks `cal_train_time` in `analyze_logs.py`. ([#662](https://github.com/open-mmlab/mmclassification/pull/662))

### Docs Update

- Update README in configs according to OpenMMLab standard. ([#672](https://github.com/open-mmlab/mmclassification/pull/672))
- Update installation guide and README. ([#624](https://github.com/open-mmlab/mmclassification/pull/624))

## v0.19.0(31/12/2021)

### Highlights

- The feature extraction function has been enhanced. See [#593](https://github.com/open-mmlab/mmclassification/pull/593) for more details.
- Provide the high-acc ResNet-50 training settings from [*ResNet strikes back*](https://arxiv.org/abs/2110.00476).
- Reproduce the training accuracy of T2T-ViT & RegNetX, and provide self-training checkpoints.
- Support DeiT & Conformer backbone and checkpoints.
- Provide a CAM visualization tool based on [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam), and detailed [user guide](https://mmclassification.readthedocs.io/en/latest/tools/visualization.html#class-activation-map-visualization)!

### New Features

- Support Precise BN. ([#401](https://github.com/open-mmlab/mmclassification/pull/401))
- Add CAM visualization tool. ([#577](https://github.com/open-mmlab/mmclassification/pull/577))
- Repeated Aug and Sampler Registry. ([#588](https://github.com/open-mmlab/mmclassification/pull/588))
- Add DeiT backbone and checkpoints. ([#576](https://github.com/open-mmlab/mmclassification/pull/576))
- Support LAMB optimizer. ([#591](https://github.com/open-mmlab/mmclassification/pull/591))
- Implement the conformer backbone. ([#494](https://github.com/open-mmlab/mmclassification/pull/494))
- Add the frozen function for Swin Transformer model. ([#574](https://github.com/open-mmlab/mmclassification/pull/574))
- Support using checkpoint in Swin Transformer to save memory. ([#557](https://github.com/open-mmlab/mmclassification/pull/557))

### Improvements

- [Reproduction] Reproduce RegNetX training accuracy. ([#587](https://github.com/open-mmlab/mmclassification/pull/587))
- [Reproduction] Reproduce training results of T2T-ViT. ([#610](https://github.com/open-mmlab/mmclassification/pull/610))
- [Enhance] Provide high-acc training settings of ResNet. ([#572](https://github.com/open-mmlab/mmclassification/pull/572))
- [Enhance] Set a random seed when the user does not set a seed. ([#554](https://github.com/open-mmlab/mmclassification/pull/554))
- [Enhance] Added `NumClassCheckHook` and unit tests. ([#559](https://github.com/open-mmlab/mmclassification/pull/559))
- [Enhance] Enhance feature extraction function. ([#593](https://github.com/open-mmlab/mmclassification/pull/593))
- [Enhance] Improve efficiency of precision, recall, f1_score and support. ([#595](https://github.com/open-mmlab/mmclassification/pull/595))
- [Enhance] Improve accuracy calculation performance. ([#592](https://github.com/open-mmlab/mmclassification/pull/592))
- [Refactor] Refactor `analysis_log.py`. ([#529](https://github.com/open-mmlab/mmclassification/pull/529))
- [Refactor] Use new API of matplotlib to handle blocking input in visualization. ([#568](https://github.com/open-mmlab/mmclassification/pull/568))
- [CI] Cancel previous runs that are not completed. ([#583](https://github.com/open-mmlab/mmclassification/pull/583))
- [CI] Skip build CI if only configs or docs modification. ([#575](https://github.com/open-mmlab/mmclassification/pull/575))

### Bug Fixes

- Fix test sampler bug. ([#611](https://github.com/open-mmlab/mmclassification/pull/611))
- Try to create a symbolic link, otherwise copy. ([#580](https://github.com/open-mmlab/mmclassification/pull/580))
- Fix a bug for multiple output in swin transformer. ([#571](https://github.com/open-mmlab/mmclassification/pull/571))

### Docs Update

- Update mmcv, torch, cuda version in Dockerfile and docs. ([#594](https://github.com/open-mmlab/mmclassification/pull/594))
- Add analysis&misc docs. ([#525](https://github.com/open-mmlab/mmclassification/pull/525))
- Fix docs build dependency. ([#584](https://github.com/open-mmlab/mmclassification/pull/584))

## v0.18.0(30/11/2021)

### Highlights

- Support MLP-Mixer backbone and provide pre-trained checkpoints.
- Add a tool to visualize the learning rate curve of the training phase. Welcome to use with the [tutorial](https://mmclassification.readthedocs.io/en/latest/tools/visualization.html#learning-rate-schedule-visualization)!

### New Features

- Add MLP Mixer Backbone. ([#528](https://github.com/open-mmlab/mmclassification/pull/528), [#539](https://github.com/open-mmlab/mmclassification/pull/539))
- Support positive weights in BCE. ([#516](https://github.com/open-mmlab/mmclassification/pull/516))
- Add a tool to visualize learning rate in each iterations. ([#498](https://github.com/open-mmlab/mmclassification/pull/498))

### Improvements

- Use CircleCI to do unit tests. ([#567](https://github.com/open-mmlab/mmclassification/pull/567))
- Focal loss for single label tasks. ([#548](https://github.com/open-mmlab/mmclassification/pull/548))
- Remove useless `import_modules_from_string`. ([#544](https://github.com/open-mmlab/mmclassification/pull/544))
- Rename config files according to the config name standard. ([#508](https://github.com/open-mmlab/mmclassification/pull/508))
- Use `reset_classifier` to remove head of timm backbones. ([#534](https://github.com/open-mmlab/mmclassification/pull/534))
- Support passing arguments to loss from head. ([#523](https://github.com/open-mmlab/mmclassification/pull/523))
- Refactor `Resize` transform and add `Pad` transform. ([#506](https://github.com/open-mmlab/mmclassification/pull/506))
- Update mmcv dependency version. ([#509](https://github.com/open-mmlab/mmclassification/pull/509))

### Bug Fixes

- Fix bug when using `ClassBalancedDataset`. ([#555](https://github.com/open-mmlab/mmclassification/pull/555))
- Fix a bug when using iter-based runner with 'val' workflow. ([#542](https://github.com/open-mmlab/mmclassification/pull/542))
- Fix interpolation method checking in `Resize`. ([#547](https://github.com/open-mmlab/mmclassification/pull/547))
- Fix a bug when load checkpoints in mulit-GPUs environment. ([#527](https://github.com/open-mmlab/mmclassification/pull/527))
- Fix an error on indexing scalar metrics in `analyze_result.py`. ([#518](https://github.com/open-mmlab/mmclassification/pull/518))
- Fix wrong condition judgment in `analyze_logs.py` and prevent empty curve. ([#510](https://github.com/open-mmlab/mmclassification/pull/510))

### Docs Update

- Fix vit config and model broken links. ([#564](https://github.com/open-mmlab/mmclassification/pull/564))
- Add abstract and image for every paper. ([#546](https://github.com/open-mmlab/mmclassification/pull/546))
- Add mmflow and mim in banner and readme. ([#543](https://github.com/open-mmlab/mmclassification/pull/543))
- Add schedule and runtime tutorial docs. ([#499](https://github.com/open-mmlab/mmclassification/pull/499))
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
- Add datetime info and saving model using torch\<1.6 format. ([#439](https://github.com/open-mmlab/mmclassification/pull/439))
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
