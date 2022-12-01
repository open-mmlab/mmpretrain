from mmcls.models.backbones.levit import get_LeViT_model
import torch

if __name__ == '__main__':
    from torchsummary import summary

    model = get_LeViT_model('LeViT_256')
    params1 = torch.load('./levit-256-p16.pth')
    model.load_state_dict(params1)
    model.eval()
    x = torch.ones((1, 3, 224, 224))
    x = model(x)
    print(x.size())

    # params = torch.load('./params.pth')
    #
    # origin = params
    # new = model.state_dict()
    # keys = []
    # keys1 = []
    # # print(len(origin), len(new))
    # # print(origin.items())
    # for key, _ in origin.items():
    #     if ('weight' in key or 'bias' in key) and 'bn.weight' not in key and 'head.bn.bias' not in key and 'head_dist' \
    #                                                                                                        '.bn.bias' \
    #             not in key:
    #         keys.append(key)
    #         # print(key)
    # # print('------------------------------------------')
    # for key, _ in new.items():
    #     # if ('weight' in key or 'bias' in key) and 'bn.weight' not in key:
    #         keys1.append(key)
    #         # print(key)
    # # print(len(keys1))
    #
    # change_dict = {}
    # for i in range(len(keys)):
    #     # print('\"%s\": \"%s\",' % (keys[i], keys1[i]))
    #     change_dict[keys1[i]] = keys[i]
    # print(change_dict)
    # print(len(change_dict))
    # with torch.no_grad():
    #     for name, param in model.named_parameters():
    #         param.copy_(origin[change_dict[name]])
    #
    # torch.save(model.state_dict(),'./new_params.pth')
    # print('success save')
