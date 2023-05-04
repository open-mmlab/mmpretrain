_base_ = [
    '../_base_/datasets/coco_vqa.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='OFA',
    task='vqa',
    vocab_size=59457,
    embedding_dim=768,
    encoder_cfg=dict(
        embed_images=dict(type='OFAResNet', depth=101),
        num_layers=6,
        num_heads=12,
    ),
    decoder_cfg=dict(
        num_layers=6,
        num_heads=12,
    ),
    generation_cfg=dict(
        num_beams=20,
        max_new_tokens=200,
        length_penalty=0.,  # VQA doesn't require longer answer.
        use_cache=True,
    ),
    tokenizer=dict(type='OFATokenizer', name_or_path='OFA-Sys/OFA-base'),
)

# data settings
data_preprocessor = dict(
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    to_rgb=True,
)

train_dataloader = None  # Eval only

# schedule settings
train_cfg = None
val_cfg = dict()
test_cfg = dict()
