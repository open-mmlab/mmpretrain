# Changelog (MMPreTrain)

## v1.1.0(12/10/2023)

### New Features

- [Feature] Implement of Zero-Shot CLIP Classifier ([#1737](https://github.com/open-mmlab/mmpretrain/pull/1737))
- [Feature] Add minigpt4 gradio demo and training script. ([#1758](https://github.com/open-mmlab/mmpretrain/pull/1758))

### Improvements

- [Config] New Version of config Adapting MobileNet Algorithm ([#1774](https://github.com/open-mmlab/mmpretrain/pull/1774))
- [Config] Support DINO self-supervised learning in project ([#1756](https://github.com/open-mmlab/mmpretrain/pull/1756))
- [Config] New Version of config Adapting Swin Transformer Algorithm ([#1780](https://github.com/open-mmlab/mmpretrain/pull/1780))
- [Enhance] Add iTPN Supports for Non-three channel image ([#1735](https://github.com/open-mmlab/mmpretrain/pull/1735))
- [Docs] Update dataset download script from opendatalab to openXlab ([#1765](https://github.com/open-mmlab/mmpretrain/pull/1765))
- [Docs] Update COCO-Retrieval dataset docs. ([#1806](https://github.com/open-mmlab/mmpretrain/pull/1806))

### Bug Fix

- Update `train.py` to compat with new config.
- Update OFA module to compat with the latest huggingface.
- Fix pipeline bug in ImageRetrievalInferencer.

## v1.0.2(15/08/2023)

### New Features

- Add MFF ([#1725](https://github.com/open-mmlab/mmpretrain/pull/1725))
- Support training of BLIP2 ([#1700](https://github.com/open-mmlab/mmpretrain/pull/1700))

### Improvements

- New Version of config Adapting MAE Algorithm ([#1750](https://github.com/open-mmlab/mmpretrain/pull/1750))
- New Version of config Adapting ConvNeXt Algorithm ([#1760](https://github.com/open-mmlab/mmpretrain/pull/1760))
- New version of config adapting BeitV2 Algorithm ([#1755](https://github.com/open-mmlab/mmpretrain/pull/1755))
- Update `dataset_prepare.md` ([#1732](https://github.com/open-mmlab/mmpretrain/pull/1732))
- New Version of `config` Adapting Vision Transformer Algorithm ([#1727](https://github.com/open-mmlab/mmpretrain/pull/1727))
- Support Infographic VQA dataset and ANLS metric. ([#1667](https://github.com/open-mmlab/mmpretrain/pull/1667))
- Support IconQA dataset. ([#1670](https://github.com/open-mmlab/mmpretrain/pull/1670))
- Fix typo MIMHIVIT to MAEHiViT ([#1749](https://github.com/open-mmlab/mmpretrain/pull/1749))

## v1.0.1(28/07/2023)

### Improvements

- Add init_cfg with type='pretrained' to downstream tasks ([#1717](https://github.com/open-mmlab/mmpretrain/pull/1717)
- Set 'is_init' in some multimodal methods ([#1718](https://github.com/open-mmlab/mmpretrain/pull/1718)
- Adapt test cases on Ascend NPU ([#1728](https://github.com/open-mmlab/mmpretrain/pull/1728)
- Add GPU Acceleration Apple silicon mac ([#1699](https://github.com/open-mmlab/mmpretrain/pull/1699)
- BEiT refactor ([#1705](https://github.com/open-mmlab/mmpretrain/pull/1705)

### Bug Fixes

- Fix dict update in minigpt4. ([#1709](https://github.com/open-mmlab/mmpretrain/pull/1709)
- Fix nested predict for multi-task prediction ([#1716](https://github.com/open-mmlab/mmpretrain/pull/1716)
- Fix the issue #1711 "GaussianBlur doesn't work" ([#1722](https://github.com/open-mmlab/mmpretrain/pull/1722)
- Just to correct a typo of 'target' ([#1655](https://github.com/open-mmlab/mmpretrain/pull/1655)
- Fix freeze without cls_token in vit ([#1693](https://github.com/open-mmlab/mmpretrain/pull/1693)
- Fix RandomCrop bug ([#1706](https://github.com/open-mmlab/mmpretrain/pull/1706)

### Docs Update

- Fix spelling ([#1689](https://github.com/open-mmlab/mmpretrain/pull/1689)

## v1.0.0(04/07/2023)

### Highlights

- Support inference of more **multi-modal** algorithms, such as **LLaVA**, **MiniGPT-4**, **Otter**, etc.
- Support around **10 multi-modal datasets**!
- Add **iTPN**, **SparK** self-supervised learning algorithms.
- Provide examples of [New Config](https://github.com/open-mmlab/mmpretrain/tree/main/mmpretrain/configs/) and [DeepSpeed/FSDP](https://github.com/open-mmlab/mmpretrain/tree/main/configs/mae/benchmarks/).

### New Features

- Transfer shape-bias tool from mmselfsup ([#1658](https://github.com/open-mmlab/mmpretrain/pull/1685))
- Download dataset by using MIM&OpenDataLab ([#1630](https://github.com/open-mmlab/mmpretrain/pull/1630))
- Support New Configs ([#1639](https://github.com/open-mmlab/mmpretrain/pull/1639), [#1647](https://github.com/open-mmlab/mmpretrain/pull/1647), [#1665](https://github.com/open-mmlab/mmpretrain/pull/1665))
- Support Flickr30k Retrieval dataset ([#1625](https://github.com/open-mmlab/mmpretrain/pull/1625))
- Support SparK ([#1531](https://github.com/open-mmlab/mmpretrain/pull/1531))
- Support LLaVA ([#1652](https://github.com/open-mmlab/mmpretrain/pull/1652))
- Support Otter ([#1651](https://github.com/open-mmlab/mmpretrain/pull/1651))
- Support MiniGPT-4 ([#1642](https://github.com/open-mmlab/mmpretrain/pull/1642))
- Add support for VizWiz dataset ([#1636](https://github.com/open-mmlab/mmpretrain/pull/1636))
- Add support for vsr dataset ([#1634](https://github.com/open-mmlab/mmpretrain/pull/1634))
- Add InternImage Classification project ([#1569](https://github.com/open-mmlab/mmpretrain/pull/1569))
- Support OCR-VQA dataset ([#1621](https://github.com/open-mmlab/mmpretrain/pull/1621))
- Support OK-VQA dataset ([#1615](https://github.com/open-mmlab/mmpretrain/pull/1615))
- Support TextVQA dataset ([#1569](https://github.com/open-mmlab/mmpretrain/pull/1569))
- Support iTPN and HiViT ([#1584](https://github.com/open-mmlab/mmpretrain/pull/1584))
- Add retrieval mAP metric ([#1552](https://github.com/open-mmlab/mmpretrain/pull/1552))
- Support NoCap dataset based on BLIP. ([#1582](https://github.com/open-mmlab/mmpretrain/pull/1582))
- Add GQA dataset ([#1585](https://github.com/open-mmlab/mmpretrain/pull/1585))

### Improvements

- Update fsdp vit-huge and vit-large config ([#1675](https://github.com/open-mmlab/mmpretrain/pull/1675))
- Support deepspeed with flexible runner ([#1673](https://github.com/open-mmlab/mmpretrain/pull/1673))
- Update Otter and LLaVA docs and config. ([#1653](https://github.com/open-mmlab/mmpretrain/pull/1653))
- Add image_only param of ScienceQA ([#1613](https://github.com/open-mmlab/mmpretrain/pull/1613))
- Support to use "split" to specify training set/validation ([#1535](https://github.com/open-mmlab/mmpretrain/pull/1535))

### Bug Fixes

- Refactor \_prepare_pos_embed in ViT ([#1656](https://github.com/open-mmlab/mmpretrain/pull/1656)， [#1679](https://github.com/open-mmlab/mmpretrain/pull/1679))
- Freeze pre norm in vision transformer ([#1672](https://github.com/open-mmlab/mmpretrain/pull/1672))
- Fix bug loading IN1k dataset ([#1641](https://github.com/open-mmlab/mmpretrain/pull/1641))
- Fix sam bug ([#1633](https://github.com/open-mmlab/mmpretrain/pull/1633))
- Fixed circular import error for new transform ([#1609](https://github.com/open-mmlab/mmpretrain/pull/1609))
- Update torchvision transform wrapper ([#1595](https://github.com/open-mmlab/mmpretrain/pull/1595))
- Set default out_type in CAM visualization ([#1586](https://github.com/open-mmlab/mmpretrain/pull/1586))

### Docs Update

- Fix spelling ([#1681](https://github.com/open-mmlab/mmpretrain/pull/1681))
- Fix doc typos ([#1671](https://github.com/open-mmlab/mmpretrain/pull/1671), [#1644](https://github.com/open-mmlab/mmpretrain/pull/1644), [#1629](https://github.com/open-mmlab/mmpretrain/pull/1629))
- Add t-SNE visualization doc ([#1555](https://github.com/open-mmlab/mmpretrain/pull/1555))

## v1.0.0rc8(22/05/2023)

### Highlights

- Support multiple multi-modal algorithms and inferencers. You can explore these features by the [gradio demo](https://github.com/open-mmlab/mmpretrain/tree/main/projects/gradio_demo)!
- Add EVA-02, Dino-V2, ViT-SAM and GLIP backbones.
- Register torchvision transforms into MMPretrain, you can now easily integrate torchvision's data augmentations in MMPretrain.

### New Features

- Support Chinese CLIP. ([#1576](https://github.com/open-mmlab/mmpretrain/pull/1576))
- Add ScienceQA Metrics ([#1577](https://github.com/open-mmlab/mmpretrain/pull/1577))
- Support multiple multi-modal algorithms and inferencers. ([#1561](https://github.com/open-mmlab/mmpretrain/pull/1561))
- add eva02 backbone ([#1450](https://github.com/open-mmlab/mmpretrain/pull/1450))
- Support dinov2 backbone ([#1522](https://github.com/open-mmlab/mmpretrain/pull/1522))
- Support some downstream classification datasets. ([#1467](https://github.com/open-mmlab/mmpretrain/pull/1467))
- Support GLIP ([#1308](https://github.com/open-mmlab/mmpretrain/pull/1308))
- Register torchvision transforms into mmpretrain ([#1265](https://github.com/open-mmlab/mmpretrain/pull/1265))
- Add ViT of SAM ([#1476](https://github.com/open-mmlab/mmpretrain/pull/1476))

### Improvements

- [Refactor] Support to freeze channel reduction and add layer decay function ([#1490](https://github.com/open-mmlab/mmpretrain/pull/1490))
- [Refactor] Support resizing pos_embed while loading ckpt and format output ([#1488](https://github.com/open-mmlab/mmpretrain/pull/1488))

### Bug Fixes

- Fix scienceqa ([#1581](https://github.com/open-mmlab/mmpretrain/pull/1581))
- Fix config of beit ([#1528](https://github.com/open-mmlab/mmpretrain/pull/1528))
- Incorrect stage freeze on RIFormer Model ([#1573](https://github.com/open-mmlab/mmpretrain/pull/1573))
- Fix ddp bugs caused by `out_type`. ([#1570](https://github.com/open-mmlab/mmpretrain/pull/1570))
- Fix multi-task-head loss potential bug ([#1530](https://github.com/open-mmlab/mmpretrain/pull/1530))
- Support bce loss without batch augmentations ([#1525](https://github.com/open-mmlab/mmpretrain/pull/1525))
- Fix clip generator init bug ([#1518](https://github.com/open-mmlab/mmpretrain/pull/1518))
- Fix the bug in binary cross entropy loss ([#1499](https://github.com/open-mmlab/mmpretrain/pull/1499))

### Docs Update

- Update PoolFormer citation to CVPR version ([#1505](https://github.com/open-mmlab/mmpretrain/pull/1505))
- Refine Inference Doc ([#1489](https://github.com/open-mmlab/mmpretrain/pull/1489))
- Add doc for usage of confusion matrix ([#1513](https://github.com/open-mmlab/mmpretrain/pull/1513))
- Update MMagic link ([#1517](https://github.com/open-mmlab/mmpretrain/pull/1517))
- Fix example_project README ([#1575](https://github.com/open-mmlab/mmpretrain/pull/1575))
- Add NPU support page ([#1481](https://github.com/open-mmlab/mmpretrain/pull/1481))
- train cfg: Removed old description ([#1473](https://github.com/open-mmlab/mmpretrain/pull/1473))
- Fix typo in MultiLabelDataset docstring ([#1483](https://github.com/open-mmlab/mmpretrain/pull/1483))

## v1.0.0rc7(07/04/2023)

### Highlights

- Integrated Self-supervised learning algorithms from **MMSelfSup**, such as **MAE**, **BEiT**, etc.
- Support **RIFormer**, a simple but effective vision backbone by removing token mixer.
- Support **LeViT**, **XCiT**, **ViG** and **ConvNeXt-V2** backbone.
- Add t-SNE visualization.
- Refactor dataset pipeline visualization.
- Support confusion matrix calculation and plot.

### New Features

- Support RIFormer. ([#1453](https://github.com/open-mmlab/mmpretrain/pull/1453))
- Support XCiT Backbone. ([#1305](https://github.com/open-mmlab/mmclassification/pull/1305))
- Support calculate confusion matrix and plot it. ([#1287](https://github.com/open-mmlab/mmclassification/pull/1287))
- Support RetrieverRecall metric & Add ArcFace config ([#1316](https://github.com/open-mmlab/mmclassification/pull/1316))
- Add `ImageClassificationInferencer`. ([#1261](https://github.com/open-mmlab/mmclassification/pull/1261))
- Support InShop Dataset (Image Retrieval). ([#1019](https://github.com/open-mmlab/mmclassification/pull/1019))
- Support LeViT backbone. ([#1238](https://github.com/open-mmlab/mmclassification/pull/1238))
- Support VIG Backbone. ([#1304](https://github.com/open-mmlab/mmclassification/pull/1304))
- Support ConvNeXt-V2 backbone. ([#1294](https://github.com/open-mmlab/mmclassification/pull/1294))

### Improvements

- Use PyTorch official `scaled_dot_product_attention` to accelerate `MultiheadAttention`. ([#1434](https://github.com/open-mmlab/mmpretrain/pull/1434))
- Add ln to vit avg_featmap output ([#1447](https://github.com/open-mmlab/mmpretrain/pull/1447))
- Update analysis tools and documentations. ([#1359](https://github.com/open-mmlab/mmclassification/pull/1359))
- Unify the `--out` and `--dump` in `tools/test.py`. ([#1307](https://github.com/open-mmlab/mmclassification/pull/1307))
- Enable to toggle whether Gem Pooling is trainable or not. ([#1246](https://github.com/open-mmlab/mmclassification/pull/1246))
- Update registries of mmcls. ([#1306](https://github.com/open-mmlab/mmclassification/pull/1306))
- Add metafile fill and validation tools. ([#1297](https://github.com/open-mmlab/mmclassification/pull/1297))
- Remove useless EfficientnetV2 config files. ([#1300](https://github.com/open-mmlab/mmclassification/pull/1300))

### Bug Fixes

- Fix precise bn hook ([#1466](https://github.com/open-mmlab/mmpretrain/pull/1466))
- Fix retrieval multi gpu bug ([#1319](https://github.com/open-mmlab/mmclassification/pull/1319))
- Fix error repvgg-deploy base config path. ([#1357](https://github.com/open-mmlab/mmclassification/pull/1357))
- Fix bug in test tools. ([#1309](https://github.com/open-mmlab/mmclassification/pull/1309))

### Docs Update

- Translate some tools tutorials to Chinese. ([#1321](https://github.com/open-mmlab/mmclassification/pull/1321))
- Add Chinese translation for runtime.md.  ([#1313](https://github.com/open-mmlab/mmclassification/pull/1313))

# Changelog (MMClassification)

## v1.0.0rc5(30/12/2022)

### Highlights

- Support EVA, RevViT, EfficientnetV2, CLIP, TinyViT and MixMIM backbones.
- Reproduce the training accuracy of ConvNeXt and RepVGG.
- Support multi-task training and testing.
- Support Test-time Augmentation.

### New Features

- [Feature] Add EfficientnetV2 Backbone. ([#1253](https://github.com/open-mmlab/mmclassification/pull/1253))
- [Feature] Support TTA and add `--tta` in `tools/test.py`. ([#1161](https://github.com/open-mmlab/mmclassification/pull/1161))
- [Feature] Support Multi-task. ([#1229](https://github.com/open-mmlab/mmclassification/pull/1229))
- [Feature] Add clip backbone. ([#1258](https://github.com/open-mmlab/mmclassification/pull/1258))
- [Feature] Add mixmim backbone with checkpoints. ([#1224](https://github.com/open-mmlab/mmclassification/pull/1224))
- [Feature] Add TinyViT for dev-1.x. ([#1042](https://github.com/open-mmlab/mmclassification/pull/1042))
- [Feature] Add some scripts for development. ([#1257](https://github.com/open-mmlab/mmclassification/pull/1257))
- [Feature] Support EVA. ([#1239](https://github.com/open-mmlab/mmclassification/pull/1239))
- [Feature] Implementation of RevViT. ([#1127](https://github.com/open-mmlab/mmclassification/pull/1127))

### Improvements

- [Reproduce] Reproduce RepVGG  Training Accuracy. ([#1264](https://github.com/open-mmlab/mmclassification/pull/1264))
- [Enhance] Support ConvNeXt More Weights. ([#1240](https://github.com/open-mmlab/mmclassification/pull/1240))
- [Reproduce] Update ConvNeXt config files. ([#1256](https://github.com/open-mmlab/mmclassification/pull/1256))
- [CI] Update CI to test PyTorch 1.13.0. ([#1260](https://github.com/open-mmlab/mmclassification/pull/1260))
- [Project] Add ACCV workshop 1st Solution. ([#1245](https://github.com/open-mmlab/mmclassification/pull/1245))
- [Project] Add Example project. ([#1254](https://github.com/open-mmlab/mmclassification/pull/1254))

### Bug Fixes

- [Fix] Fix imports in transforms. ([#1255](https://github.com/open-mmlab/mmclassification/pull/1255))
- [Fix] Fix CAM visualization. ([#1248](https://github.com/open-mmlab/mmclassification/pull/1248))
- [Fix] Fix the requirements and lazy register mmpretrain models. ([#1275](https://github.com/open-mmlab/mmclassification/pull/1275))

## v1.0.0rc4(06/12/2022)

### Highlights

- Upgrade API to get pre-defined models of MMClassification. See [#1236](https://github.com/open-mmlab/mmclassification/pull/1236) for more details.
- Refactor BEiT backbone and support v1/v2 inference. See [#1144](https://github.com/open-mmlab/mmclassification/pull/1144).

### New Features

- Support getting model from the name defined in the model-index file. ([#1236](https://github.com/open-mmlab/mmclassification/pull/1236))

### Improvements

- Support evaluate on both EMA and non-EMA models. ([#1204](https://github.com/open-mmlab/mmclassification/pull/1204))
- Refactor BEiT backbone and support v1/v2 inference. ([#1144](https://github.com/open-mmlab/mmclassification/pull/1144))

### Bug Fixes

- Fix `reparameterize_model.py` doesn't save meta info. ([#1221](https://github.com/open-mmlab/mmclassification/pull/1221))
- Fix dict update in BEiT. ([#1234](https://github.com/open-mmlab/mmclassification/pull/1234))

### Docs Update

- Update install tutorial. ([#1223](https://github.com/open-mmlab/mmclassification/pull/1223))
- Update MobileNetv2 & MobileNetv3 readme. ([#1222](https://github.com/open-mmlab/mmclassification/pull/1222))
- Add version selection in the banner. ([#1217](https://github.com/open-mmlab/mmclassification/pull/1217))

## v1.0.0rc3(21/11/2022)

### Highlights

- Add **Switch Recipe** Hook, Now we can modify training pipeline, mixup and loss settings during training, see [#1101](https://github.com/open-mmlab/mmclassification/pull/1101).
- Add **TIMM and HuggingFace** wrappers. Now you can train/use models in TIMM/HuggingFace directly, see [#1102](https://github.com/open-mmlab/mmclassification/pull/1102).
- Support **retrieval tasks**, see [#1055](https://github.com/open-mmlab/mmclassification/pull/1055).
- Reproduce **mobileone** training accuracy. See [#1191](https://github.com/open-mmlab/mmclassification/pull/1191)

### New Features

- Add checkpoints from EfficientNets NoisyStudent & L2. ([#1122](https://github.com/open-mmlab/mmclassification/pull/1122))
- Migrate CSRA head to 1.x. ([#1177](https://github.com/open-mmlab/mmclassification/pull/1177))
- Support RepLKnet backbone. ([#1129](https://github.com/open-mmlab/mmclassification/pull/1129))
- Add Switch Recipe Hook. ([#1101](https://github.com/open-mmlab/mmclassification/pull/1101))
- Add adan optimizer. ([#1180](https://github.com/open-mmlab/mmclassification/pull/1180))
- Support DaViT. ([#1105](https://github.com/open-mmlab/mmclassification/pull/1105))
- Support Activation Checkpointing for ConvNeXt. ([#1153](https://github.com/open-mmlab/mmclassification/pull/1153))
- Add TIMM and HuggingFace wrappers to build classifiers from them directly. ([#1102](https://github.com/open-mmlab/mmclassification/pull/1102))
- Add reduction for neck ([#978](https://github.com/open-mmlab/mmclassification/pull/978))
- Support HorNet Backbone for dev1.x. ([#1094](https://github.com/open-mmlab/mmclassification/pull/1094))
- Add arcface head. ([#926](https://github.com/open-mmlab/mmclassification/pull/926))
- Add Base Retriever and Image2Image Retriever for retrieval tasks. ([#1055](https://github.com/open-mmlab/mmclassification/pull/1055))
- Support MobileViT backbone. ([#1068](https://github.com/open-mmlab/mmclassification/pull/1068))

### Improvements

- [Enhance] Enhance ArcFaceClsHead. ([#1181](https://github.com/open-mmlab/mmclassification/pull/1181))
- [Refactor] Refactor to use new fileio API in MMEngine. ([#1176](https://github.com/open-mmlab/mmclassification/pull/1176))
- [Enhance] Reproduce mobileone training accuracy. ([#1191](https://github.com/open-mmlab/mmclassification/pull/1191))
- [Enhance] add deleting params info in swinv2. ([#1142](https://github.com/open-mmlab/mmclassification/pull/1142))
- [Enhance] Add more mobilenetv3 pretrains. ([#1154](https://github.com/open-mmlab/mmclassification/pull/1154))
- [Enhancement] RepVGG for YOLOX-PAI for dev-1.x. ([#1126](https://github.com/open-mmlab/mmclassification/pull/1126))
- [Improve] Speed up data preprocessor. ([#1064](https://github.com/open-mmlab/mmclassification/pull/1064))

### Bug Fixes

- Fix the torchserve. ([#1143](https://github.com/open-mmlab/mmclassification/pull/1143))
- Fix configs due to api refactor of `num_classes`. ([#1184](https://github.com/open-mmlab/mmclassification/pull/1184))
- Update mmpretrain2torchserve. ([#1189](https://github.com/open-mmlab/mmclassification/pull/1189))
- Fix for `inference_model` cannot get classes information in checkpoint. ([#1093](https://github.com/open-mmlab/mmclassification/pull/1093))

### Docs Update

- Add not-found page extension. ([#1207](https://github.com/open-mmlab/mmclassification/pull/1207))
- update visualization doc. ([#1160](https://github.com/open-mmlab/mmclassification/pull/1160))
- Support sort and search the Model Summary table. ([#1100](https://github.com/open-mmlab/mmclassification/pull/1100))
- Improve the ResNet model page. ([#1118](https://github.com/open-mmlab/mmclassification/pull/1118))
- update the readme of convnext. ([#1156](https://github.com/open-mmlab/mmclassification/pull/1156))
- Fix the installation docs link in README. ([#1164](https://github.com/open-mmlab/mmclassification/pull/1164))
- Improve ViT and MobileViT model pages. ([#1155](https://github.com/open-mmlab/mmclassification/pull/1155))
- Improve Swin Doc and Add Tabs enxtation. ([#1145](https://github.com/open-mmlab/mmclassification/pull/1145))
- Add MMEval projects link in README. ([#1162](https://github.com/open-mmlab/mmclassification/pull/1162))
- Add runtime configuration docs. ([#1128](https://github.com/open-mmlab/mmclassification/pull/1128))
- Add custom evaluation docs ([#1130](https://github.com/open-mmlab/mmclassification/pull/1130))
- Add custom pipeline docs. ([#1124](https://github.com/open-mmlab/mmclassification/pull/1124))
- Add MMYOLO projects link in MMCLS1.x. ([#1117](https://github.com/open-mmlab/mmclassification/pull/1117))

## v1.0.0rc2(12/10/2022)

### New Features

- [Feature] Support DeiT3. ([#1065](https://github.com/open-mmlab/mmclassification/pull/1065))

### Improvements

- [Enhance] Update `analyze_results.py` for dev-1.x. ([#1071](https://github.com/open-mmlab/mmclassification/pull/1071))
- [Enhance] Get scores from inference api. ([#1070](https://github.com/open-mmlab/mmclassification/pull/1070))

### Bug Fixes

- [Fix] Update requirements. ([#1083](https://github.com/open-mmlab/mmclassification/pull/1083))

### Docs Update

- [Docs] Add 1x docs schedule. ([#1015](https://github.com/open-mmlab/mmclassification/pull/1015))

## v1.0.0rc1(30/9/2022)

### New Features

- Support MViT for MMCLS 1.x ([#1023](https://github.com/open-mmlab/mmclassification/pull/1023))
- Add ViT huge architecture. ([#1049](https://github.com/open-mmlab/mmclassification/pull/1049))
- Support EdgeNeXt for dev-1.x. ([#1037](https://github.com/open-mmlab/mmclassification/pull/1037))
- Support Swin Transformer V2 for MMCLS 1.x. ([#1029](https://github.com/open-mmlab/mmclassification/pull/1029))
- Add efficientformer Backbone for MMCls 1.x. ([#1031](https://github.com/open-mmlab/mmclassification/pull/1031))
- Add MobileOne Backbone For MMCls 1.x.  ([#1030](https://github.com/open-mmlab/mmclassification/pull/1030))
- Support BEiT Transformer layer. ([#919](https://github.com/open-mmlab/mmclassification/pull/919))

### Improvements

- [Refactor] Fix visualization tools. ([#1045](https://github.com/open-mmlab/mmclassification/pull/1045))
- [Improve] Update benchmark scripts ([#1028](https://github.com/open-mmlab/mmclassification/pull/1028))
- [Improve] Update tools to enable `pin_memory` and `persistent_workers` by default. ([#1024](https://github.com/open-mmlab/mmclassification/pull/1024))
- [CI] Update circle-ci and github workflow. ([#1018](https://github.com/open-mmlab/mmclassification/pull/1018))

### Bug Fixes

- Fix verify dataset tool in 1.x. ([#1062](https://github.com/open-mmlab/mmclassification/pull/1062))
- Fix `loss_weight` in `LabelSmoothLoss`. ([#1058](https://github.com/open-mmlab/mmclassification/pull/1058))
- Fix the output position of Swin-Transformer. ([#947](https://github.com/open-mmlab/mmclassification/pull/947))

### Docs Update

- Auto generate model summary table.  ([#1010](https://github.com/open-mmlab/mmclassification/pull/1010))
- Refactor new modules tutorial. ([#998](https://github.com/open-mmlab/mmclassification/pull/998))

## v1.0.0rc0(31/8/2022)

MMClassification 1.0.0rc0 is the first version of MMClassification 1.x, a part of the OpenMMLab 2.0 projects.

Built upon the new [training engine](https://github.com/open-mmlab/mmengine), MMClassification 1.x unifies the interfaces of dataset, models, evaluation, and visualization.

And there are some BC-breaking changes. Please check [the migration tutorial](https://mmclassification.readthedocs.io/en/1.x/migration.html) for more details.

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
- [Docs] fix logo url link from mmocr to mmpretrain. ([#732](https://github.com/open-mmlab/mmclassification/pull/732))

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
- Provide dockerfile to build mmpretrain dev docker image.

### New Features

- Add transformer in transformer backbone and pretrain checkpoints. ([#339](https://github.com/open-mmlab/mmclassification/pull/339))
- Support mim, welcome to use mim to manage your mmpretrain project. ([#376](https://github.com/open-mmlab/mmclassification/pull/376))
- Add Dockerfile. ([#365](https://github.com/open-mmlab/mmclassification/pull/365))
- Add ResNeSt configs. ([#332](https://github.com/open-mmlab/mmclassification/pull/332))

### Improvements

- Use the `presistent_works` option if available, to accelerate training. ([#349](https://github.com/open-mmlab/mmclassification/pull/349))
- Add Chinese ipynb tutorial. ([#306](https://github.com/open-mmlab/mmclassification/pull/306))
- Refactor unit tests. ([#321](https://github.com/open-mmlab/mmclassification/pull/321))
- Support to test mmdet inference with mmpretrain backbone. ([#343](https://github.com/open-mmlab/mmclassification/pull/343))
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
- Check mmcv version when importing mmpretrain to ensure compatibility. (#209)
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
- Support showing and saving painted results in `mmpretrain.apis.test` and `tools/test.py`, matching the behavior of other open-mmlab projects. (#162)

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
