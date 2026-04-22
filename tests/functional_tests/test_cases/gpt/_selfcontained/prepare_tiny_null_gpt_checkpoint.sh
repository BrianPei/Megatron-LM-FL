#!/bin/bash

set -euo pipefail

TARGET_CKPT_DIR=${1:?target checkpoint directory is required}
MASTER_PORT=${2:-6701}

if [[ -f "${TARGET_CKPT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "Tiny GPT checkpoint already exists at ${TARGET_CKPT_DIR}"
    exit 0
fi

rm -rf "${TARGET_CKPT_DIR}"
mkdir -p "${TARGET_CKPT_DIR}"

uv run --no-sync python -m torch.distributed.run \
    --nproc_per_node 1 \
    --nnodes 1 \
    --master_addr localhost \
    --master_port "${MASTER_PORT}" \
    --node_rank 0 \
    pretrain_gpt.py \
    --use-mcore-models \
    --num-layers 2 \
    --hidden-size 64 \
    --num-attention-heads 4 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --seq-length 32 \
    --max-position-embeddings 64 \
    --train-iters 1 \
    --exit-interval 1 \
    --save-interval 1 \
    --eval-interval 1000 \
    --eval-iters 1 \
    --lr 1.0e-4 \
    --min-lr 1.0e-4 \
    --lr-decay-style constant \
    --lr-decay-iters 1 \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --log-interval 1 \
    --seed 1234 \
    --save "${TARGET_CKPT_DIR}" \
    --load "${TARGET_CKPT_DIR}" \
    --tensorboard-dir "${TARGET_CKPT_DIR}/tensorboard" \
    --tokenizer-type NullTokenizer \
    --vocab-size 4095 \
    --mock-data \
    --split 1,1,1 \
    --distributed-backend nccl \
    --transformer-impl local \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --deterministic-mode \
    --no-gradient-accumulation-fusion \
    --no-save-optim \
    --no-save-rng \
    --ckpt-format torch \
    --bf16
