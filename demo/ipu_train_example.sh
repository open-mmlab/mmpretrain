

# get SOTA accuracy 81.2 for 224 input ViT fine-tuning, reference is below:
# https://github.com/google-research/vision_transformer#available-vit-models
# cfg: vit-base-p16_ft-4xb544_in1k-224_ipu train model in fp16 precision
# 8 epoch, 2176 batch size, 16 IPUs, 4 replicas, model Tput = 5600 images, training time 0.6 hour roughly
cfg_name=vit-base-p16_ft-4xb544_in1k-224_ipu
python3 tools/train.py configs/vision_transformer/${cfg_name}.py --ipu-replicas 4 --no-validate &&
python3 tools/test.py configs/vision_transformer/${cfg_name}.py work_dirs/${cfg_name}/latest.pth --metrics accuracy --device ipu
