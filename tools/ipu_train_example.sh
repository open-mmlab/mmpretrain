
   
# get SOTA accuracy 81.2 for 224 input ViT fine-tuning, reference is below:
# https://github.com/google-research/vision_transformer#available-vit-models
# cfg: vit-base-p16_ft-4xb512_in1k-224_ipu_half train model in fp16 precision, faster and the accuracy is 80.8 now
cfg_name=vit-base-p16_ft-4xb512_in1k-224_ipu_half
python3 tools/train.py configs/vision_transformer/${cfg_name}.py --ipu_replicas 4 --no-validate &&
python3 tools/test.py configs/vision_transformer/${cfg_name}.py work_dirs/${cfg_name}/latest.pth --metrics accuracy --device ipu
