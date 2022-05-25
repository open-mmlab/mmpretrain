# 0.x 相关兼容性问题

## MMClassification 0.20.1

### MMCV 兼容性

在 Twins 骨干网络中，我们使用了 MMCV 提供的 `PatchEmbed` 模块，该模块是在 MMCV 1.4.2 版本加入的，因此我们需要将 MMCV 依赖版本升至 1.4.2。
