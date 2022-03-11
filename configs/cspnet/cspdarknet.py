from mmcls.models.backbones import CSPDarknet

if __name__ == '__main__':
    import torch
    model = CSPDarknet(depth=53)
    model.eval()
    inputs = torch.rand(1, 3, 224, 224)
    level_outputs = model.forward(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))
