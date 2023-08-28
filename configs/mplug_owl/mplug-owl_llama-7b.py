_base_ = [
    '../_base_/datasets/coco_caption.py',
    '../_base_/default_runtime.py',
]

# dataset settings
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(224, 224),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='PackInputs', meta_keys=['image_id']),
]

val_dataloader = dict(batch_size=1, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# model settings
model = dict(
    type='MplugOwlForConditionalGeneration',
    vision_encoder=dict(
        type='MplugOwlVisionModel',
        hidden_size=1024, 
        image_size=224, 
        patch_size=14, 
        intermediate_size=4096, 
        num_attention_heads=16, 
        attention_dropout=0.0,
        layer_norm_eps=1e-6, 
        num_hidden_layers=24,
        pretrained=  # noqa
        ''  # noqa
    ),
    abstractor_model=dict(
        type='MplugOwlVisualAbstractorModel',
        language_hidden_size=4096, 
        num_hidden_layers=6, 
        hidden_size=1024,
        num_attention_heads=16,
        intermediate_size=4096,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-6,
        encoder_hidden_size=1024,
        pretrained=  # noqa
        ''  # noqa
    ),
    lang_encoder=dict(
        type='AutoModelForCausalLM', name_or_path='YOUR_PATH_TO_LLAMA'),
    tokenizer=dict(type='LlamaTokenizer', name_or_path='YOUR_PATH_TO_LLAMA'),
    task='caption',
    prompt_template="The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <image>\nHuman: how many cats are there?\nAI: ",
    # raw_prompts=[
    #     '<Img><ImageHere></Img> Describe this image in detail.',
    #     '<Img><ImageHere></Img> Take a look at this image and describe what you notice.',  # noqa
    #     '<Img><ImageHere></Img> Please provide a detailed description of the picture.',  # noqa
    #     '<Img><ImageHere></Img> Could you describe the contents of this image for me?',  # noqa
    # ],
    # max_txt_len=160,
    # end_sym='###'
    )

# schedule settings
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=1e-5, weight_decay=0.05))

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=5,
    )
]

train_cfg = dict(by_epoch=True, max_epochs=5)
val_cfg = dict()
test_cfg = dict()
