from mmcls.models.backbones.levit import get_LeViT_model
import torch

if __name__ == '__main__':
    from torchsummary import summary

    model = get_LeViT_model('LeViT_384')
    # params1 = torch.load('./new_params.pth')
    # model.load_state_dict(params1)
    # x = torch.ones((1, 3, 224, 224), device='cuda:0')
    # # x.to("cuda:0")
    # model.to("cuda:0")
    # model.eval()
    # x = model(x)
    # print(x.size())

    params = torch.load('./source_model_path/LeViT-384-9bdaf2e2.pth')

    origin = params['model']
    new = model.state_dict()
    keys = []
    keys1 = []
    print(len(origin), len(new))
    # print(origin.items())
    for key, _ in origin.items():
        keys.append(key)
        print(key)
    print('------------------------------------------')
    for key, _ in new.items():
        keys1.append(key)
        print(key)

    change_dict = {}
    for i in range(len(keys)):
        # print('\"%s\": \"%s\",' % (keys[i], keys1[i]))
        change_dict[keys1[i]] = keys[i]

    for i in change_dict:
        print("%s\t------------->\t%s" % (change_dict[i], i))
    with torch.no_grad():
        for name, param in new.items():
            param.copy_(origin[change_dict[name]])

    torch.save(new, './converters_model_path/LeViT-384.pth')
    print('success save')
