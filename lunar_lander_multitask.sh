#! /bin/bash

python main.py \
    --env-name LunarLanderMultitask-v0 \
    --num-workers 16 \
    --fast-lr 0.1 \
    --max-kl 0.01 \
    --fast-batch-size 20 \
    --meta-batch-size 40 \
    --num-layers 2 \
    --hidden-size 100 \
    --num-batches 1000 \
    --gamma 0.99 \
    --tau 1.0 \
    --cg-damping 1e-5 \
    --ls-max-steps 15 \
    --output-folder maml-lunarlander-multitask-easy4 \
    --device cuda