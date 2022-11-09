# Pytorch 转 TorchScript （试验性的）

<!-- TOC -->

- [如何将 PyTorch 模型转换至 TorchScript](#如何将-pytorch-模型转换至-torchscript)
  - [使用方法](#使用方法)
- [提示](#提示)
- [常见问题](#常见问题)

<!-- TOC -->

## 如何将 PyTorch 模型转换至 TorchScript

### 使用方法

```bash
python tools/deployment/pytorch2torchscript.py \
    ${CONFIG_FILE} \
    --checkpoint ${CHECKPOINT_FILE} \
    --output-file ${OUTPUT_FILE} \
    --shape ${IMAGE_SHAPE} \
    --verify \
```

所有参数的说明：

- `config` : 模型配置文件的路径。
- `--checkpoint` : 模型权重文件的路径。
- `--output-file`: TorchScript 模型的输出路径。如果没有指定，默认为当前脚本执行路径下的 `tmp.pt`。
- `--shape`: 模型输入的高度和宽度。如果没有指定，默认为 `224 224`。
- `--verify`: 是否验证导出模型的正确性。如果没有指定，默认为`False`。

示例：

```bash
python tools/deployment/pytorch2torchscript.py \
    configs/resnet/resnet18_8xb16_cifar10.py \
    --checkpoint checkpoints/resnet/resnet18_b16x8_cifar10.pth \
    --output-file checkpoints/resnet/resnet18_b16x8_cifar10.pt \
    --verify \
```

注：

- *所有模型基于 Pytorch==1.8.1 通过了转换测试*

## 提示

- 由于 `torch.jit.is_tracing()` 只在 PyTorch 1.6 之后的版本中得到支持，对于 PyTorch 1.3-1.5 的用户，我们建议手动提前返回结果。
- 如果你在本仓库的模型转换中遇到问题，请在 GitHub 中创建一个 issue，我们会尽快处理。

## 常见问题

- 无
