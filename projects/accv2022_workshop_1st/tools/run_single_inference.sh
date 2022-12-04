!/bin/sh

declare -A dict
dict=(['l-448-arc-rounda3-0.4.pth']="7473" \
      ['l-448-arc-rounda3-0.5.pth']="7418" \
      ['l-448-arc-roundb1.pth']='7520' \
      ['l-448-arc-roundb2.pth']='7570' \
      ['l-448-arc-roundb3.pth']='7620' \
      ['l-448-ce-rounda3-0.4.pth']='7471' \
      ['l-448-ce-rounda3-0.5.pth']='7405' \
      ['l-448-ce-roundb1.pth']='7450' \
      ['l-448-ce-roundb2.pth']='7500' \
      ['l-448-ce-roundb3.pth']='7550' \
      ['b-384-arc-roundb2.pth']='7460' \
      ['b-384-arc-roundb1.pth']='7410' \
      ['b-384-arc-roundb3.pth']='7510' \
      ['l-384-arc-round2.pth']='7510' \
      ['l-384-arc-rounda3.pth']='7400' \
      ['l-384-arc-roundb1.pth']='7460' \
      ['l-384-arc-roundb3.pth']='7560')


NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
GPUS=${GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-32}
PY_ARGS=${@:4}


python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/infer_folder.py \
    $1 \
    $2 \
    ./data/ACCV_workshop/testb \
    --dump $3 \
    --tta \
    --cfg-option test_dataloader.batch_size=$BATCH_SIZE \
    --launcher pytorch \
    $PY_ARGS
