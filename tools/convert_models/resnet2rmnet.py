import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcls
from mmcls.apis import init_model

'''
A PyTorch implementation  of : `RMNet: Equivalently Removing Residual Connection from Networks <https://arxiv.org/abs/2111.00687>`_
'''

def RMOperation(block):
    block.eval()
    conv_bn1=nn.utils.fuse_conv_bn_eval(block.conv1,block.bn1)
    conv_bn2=nn.utils.fuse_conv_bn_eval(block.conv2,block.bn2)
    idconv1=nn.Conv2d(in_channels=conv_bn1.in_channels,
                      out_channels=conv_bn1.in_channels+conv_bn1.out_channels,
                      kernel_size=conv_bn1.kernel_size,
                      stride=conv_bn1.stride,
                      padding=conv_bn1.padding,
                      dilation=conv_bn1.dilation,
                      groups=conv_bn1.groups)
    idconv2=nn.Conv2d(in_channels=conv_bn1.in_channels+conv_bn2.in_channels,
                      out_channels=conv_bn2.out_channels,
                      kernel_size=conv_bn2.kernel_size,
                      stride=conv_bn2.stride,
                      padding=conv_bn2.padding,
                      dilation=conv_bn2.dilation,
                      groups=conv_bn2.groups)
    
    nn.init.dirac_(idconv1.weight.data[:conv_bn1.in_channels])
    nn.init.zeros_(idconv1.bias.data[:conv_bn1.in_channels])
    idconv1.weight.data[conv_bn1.in_channels:]=conv_bn1.weight.data
    idconv1.bias.data[conv_bn1.in_channels:]=conv_bn1.bias.data
    
    idconv2.weight.data[:,conv_bn1.in_channels:]=conv_bn2.weight.data
    idconv2.bias.data=conv_bn2.bias.data
    if block.downsample is None:
        nn.init.dirac_(idconv2.weight.data[:,:conv_bn1.in_channels])
    else:
        conv_bn_downsample=nn.utils.fuse_conv_bn_eval(block.downsample[0],block.downsample[1])
        idconv2.weight.data[:,:conv_bn1.in_channels]=F.pad(conv_bn_downsample.weight.data,[1,1,1,1])
        idconv2.bias.data+=conv_bn_downsample.bias.data
        
    return nn.Sequential(idconv1,nn.ReLU(inplace=True),idconv2,nn.ReLU(inplace=True))

def resnet2vgg(config_path, checkpoint_path, save_path):
    model = init_model(config_path, checkpoint=checkpoint_path).cpu()
    print('Converting...')
    
    backbone=[]
    for m in model.backbone.layer1.modules():
        if isinstance(m,mmcls.models.backbones.resnet.BasicBlock):
            backbone.append(RMOperation(m))
    model.backbone.layer1=nn.Sequential(*backbone)

    backbone=[]
    for m in model.backbone.layer2.modules():
        if isinstance(m,mmcls.models.backbones.resnet.BasicBlock):
            backbone.append(RMOperation(m))
    model.backbone.layer2=nn.Sequential(*backbone)

    backbone=[]
    for m in model.backbone.layer3.modules():
        if isinstance(m,mmcls.models.backbones.resnet.BasicBlock):
            backbone.append(RMOperation(m))
    model.backbone.layer3=nn.Sequential(*backbone)

    backbone=[]
    for m in model.backbone.layer4.modules():
        if isinstance(m,mmcls.models.backbones.resnet.BasicBlock):
            backbone.append(RMOperation(m))
    model.backbone.layer4=nn.Sequential(*backbone)
    
    torch.save(model.state_dict(), save_path)

    print('Done! Save at path "{}"'.format(save_path))
    
    return model

def main():
    parser = argparse.ArgumentParser(
        description='Convert the parameters of the repvgg block '
        'from training mode to deployment mode.')
    parser.add_argument(
        'config_path',
        help='The path to the configuration file of the network '
        'containing the repvgg block.')
    parser.add_argument(
        'checkpoint_path',
        help='The path to the checkpoint file corresponding to the model.')
    parser.add_argument(
        'save_path',
        help='The path where the converted checkpoint file is stored.')
    args = parser.parse_args()

    save_path = Path(args.save_path)
    if save_path.suffix != '.pth':
        print('The path should contain the name of the pth format file.')
        exit()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(resnet2vgg(args.config_path, args.checkpoint_path, args.save_path))


if __name__ == '__main__':
    main()
