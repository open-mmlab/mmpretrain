from mmcls.models.backbones import CSPResNeXt

if __name__ == '__main__':
    import torch
    model = CSPResNeXt(depth=50)
    model.eval()
    print(model)
    inputs = torch.rand(1, 3, 224, 224)
    level_outputs = model.forward(inputs)
    for i, level_out in enumerate(level_outputs):
        print(i, tuple(level_out.shape))
