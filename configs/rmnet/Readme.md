# How to Use:
## Training phase (normally train a ResNet):
`./tools/dist_train.sh configs/resnet/resnet18_b32x8_imagenet.py ${GPU_NUM} [optional arguments]`

## convert ResNet to RMNet(presently only support ResNet18 and ResNet34):
`python tools/convert_models/resnet2rmnet.py configs/resnet/resnet18_b32x8_imagenet.py ${checkpoint_path} ${save_path}`

## inference phase:
`./tools/dist_test.sh configs/rmnet/rmnet18_b32x8_imagenet.py ${CHECKPOINT_FILE} ${GPU_NUM} [--metrics ${METRICS}] [--out ${RESULT_FILE}]`
