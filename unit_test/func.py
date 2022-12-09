import torch


def tensor_test(model_src, state_src, model_dst, state_dst):
    model_src.eval()
    model_dst.eval()

    state_src = torch.load(state_src)
    model_src.load_state_dict(state_src)

    state_dst = torch.load(state_dst)
    model_dst.load_state_dict(state_dst)

    for i in range(10):
        input = torch.randn((1, 3, 224, 224))
        output_src = model_src(input)
        output_dst = model_dst(input)

        if False in output_src == output_dst:
            print(f'测试失败{i}')
            break
        else:
            print(f'测试成功{i}')
