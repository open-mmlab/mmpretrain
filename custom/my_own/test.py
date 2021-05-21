import torch
# torch.cuda.set_device(0)
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
show = ToPILImage()   #0-1 转 0-255
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.autograd import Variable
#读取MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),  # 0-255  转为  0-1
    transforms.Normalize((0.5,),(0.5,))  # 0-1  转为  -1-1
])
trainset=torchvision.datasets.MNIST(
    root='/home/n/Github/mmclassification0110/mmclassification/data/',
    train=True,
    download=False,
    transform=transform
)
trainloader=torch.utils.data.DataLoader(
    trainset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)
testset=torchvision.datasets.MNIST(
    root='/home/n/Github/mmclassification0110/mmclassification/data/',
    train=False,
    download=False,
    transform=transform
)
testloader=torch.utils.data.DataLoader(
    testset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

data_iter=iter(testloader)
images, labels = data_iter.next() #一个batch多个图片  输入 1 1 28 28
print("label:"+str(labels))
# img=transforms.ToPILImage(images)
img=images.squeeze(0)
# img=torchvision.utils.make_grid(images).numpy()
plt.imshow(np.transpose(img,(1,2,0)),cmap='gray')
plt.show()