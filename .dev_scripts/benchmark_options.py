third_part_libs = [
    'pip install -r ../requirements.txt',
]

default_floating_range = 0.2
model_floating_ranges = {
    'blip/blip-base_8xb32_retrieval.py': 1.0,
    'blip2/blip2-opt2.7b_8xb32_caption.py': 1.0,
    'ofa/ofa-base_finetuned_caption.py': 1.0,
}
