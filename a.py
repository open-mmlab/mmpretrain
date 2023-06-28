from mmpretrain.datasets import VisDial

test_pipeline = [
    dict(type='mmpretrain.LoadImageFromFile'),
    dict(
        type='mmpretrain.ResizeEdge',
        scale=224,
        interpolation='bicubic',
        backend='pillow'),
    dict(type='mmpretrain.CenterCrop', crop_size=(224, 224)),
    dict(
        type='mmpretrain.PackInputs',
        algorithm_keys=['question', 'gt_answer', 'sub_set'],
        meta_keys=['image_id'],
    ),
]

dataset = VisDial(
    data_root='data/visualdialogue',
    data_prefix='VisualDialog_val2018',
    ann_file='visdial_1.0_val.json',
    pipeline=test_pipeline)

# dataset = ChartQA(
#     data_root='data/chartqa/train',
#     data_prefix='png',
#     ann_file=['train_human.json', ],
#     pipeline=test_pipeline)

print('a')
