from mmcls.models.backbones.levit import get_LeViT_model
import torch

if __name__ == '__main__':
    from torchsummary import summary

    model = get_LeViT_model('LeViT_256')

    params = torch.load('./LeViT-256-13b5763e.pth')

    origin = params['model']
    new = model.state_dict()
    keys = []
    keys1 = []
    # print(len(origin), len(new))
    # print(origin.items())
    for key, _ in origin.items():
        if ('weight' in key or 'bias' in key) and 'bn.weight' not in key and 'head.bn.bias' not in key and 'head_dist' \
                                                                                                           '.bn.bias' \
                not in key:
            keys.append(key)
            # print(key)
    # print('------------------------------------------')
    for key, _ in new.items():
        if ('weight' in key or 'bias' in key) and 'bn.weight' not in key:
            keys1.append(key)
            # print(key)
    change_dict = {}
    for i in range(len(keys)):
        # print('\"%s\": \"%s\",' % (keys[i], keys1[i]))
        change_dict[keys1[i]] = keys[i]
    print(change_dict)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(origin[change_dict[name]])

    torch.save(model.state_dict(),'./new_params.pth')
    print('success save')

