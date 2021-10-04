_base_ = './logos_dataset_detection.py'
data_root = '/home/ubuntu/data/logo_dataset/'

data = dict(
    train=dict(
        ann_file=data_root + 'ImageSets/Main/train_micro.txt',
    ),
    val=dict(
        ann_file=data_root + 'ImageSets/Main/validation_micro.txt',
    ),
    test=dict(
        ann_file=data_root + 'ImageSets/Main/test_micro.txt',
    ),
)
