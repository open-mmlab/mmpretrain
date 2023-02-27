# MaxViT Models


MaxViT is a family of hybrid (CNN + ViT) image classification models, that achieves better performances across the board
for both parameter and FLOPs efficiency than both SoTA ConvNets and Transformers. 
They can also scale well on large dataset sizes like ImageNet-21K. 
Notably, due to the linear-complexity of the grid attention used, 
MaxViT is able to ''see'' globally throughout the entire network, even in earlier, high-resolution stages.



