from mmpretrain.datasets import ChartQA


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


dataset = ChartQA(
    data_root='data/chartqa/test',
    data_prefix='png',
    ann_file=['test_human.json', 'test_augmented.json'],
    pipeline=test_pipeline)

# dataset = ChartQA(
#     data_root='data/chartqa/train',
#     data_prefix='png',
#     ann_file=['train_human.json', ],
#     pipeline=test_pipeline)


print("a")