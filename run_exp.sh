# args setting
seed=0
manual_fold=0
num_gpus=2
save_dir=effnet_b5_food_dataset

# 그냥 돌리는법 (single gpu)
# python3 tools/train.py configs/efficientnet/efficientnet-b5_8xb32-01norm_food.py --work-dir work_dirs/effnet_b0_test --seed 0 --deterministic

# 그냥 돌리는법 2 (multi gpu), ...food.py 뒤에 숫자는 gpu 개수
# bash tools/dist_train.sh configs/efficientnet/efficientnet-b5_8xb32-01norm_food.py 2 --work-dir work_dirs/effnet_b0_multi_test --seed 0 --deterministic

# single-gpu
# python3 tools/train.py \
# configs/efficientnet/efficientnet-b5_8xb32-01norm_food.py \
# --work-dir work_dirs/${save_dir}_manual_fold${manual_fold} \
# --seed ${seed} \
# --deterministic \
# --cfg-options data.train.fold=${manual_fold} data.val.fold=${manual_fold} data.test.fold=${manual_fold}

# multi-gpu
bash tools/dist_train.sh \
configs/efficientnet/efficientnet-b5_8xb32-01norm_food.py ${num_gpus} \
--work-dir work_dirs/${save_dir}_manual_fold${manual_fold} \
--seed ${seed} \
--deterministic \
--cfg-options data.train.fold=${manual_fold} data.val.fold=${manual_fold} data.test.fold=${manual_fold}

# multi-gpu folds
# folds=(0 1 2 3 4)
# for fold in ${folds[@]}
#     do
#     bash tools/dist_train.sh \
#     configs/efficientnet/efficientnet-b5_8xb32-01norm_food.py ${num_gpus} \ # number of gpus
#     --work-dir work_dirs/${save_dir}_fold${fold} \
#     --seed ${seed} \
#     --deterministic \
#     --cfg-options data.train.fold=${fold} data.val.fold=${fold} data.test.fold=${fold}
#     done