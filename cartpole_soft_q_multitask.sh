#! /bin/bash

NUM_JOBS=4

OUTPUT_DIR='maml-soft-q-cartpole-multitask-2goals'
parallel -j $NUM_JOBS \
    'CUDA_VISIBLE_DEVICES=$(({%} % 4))' python main.py \
        --env-name CartPoleMultitask-v0 \
        --num-workers 16 \
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
        --inner_q \
        --device cuda \
        --inner-q-residue-gradient \
        --inner-q-soft \
        --fast-lr {1} \
        --inner-q-soft-temp {2} \
        --output-folder $OUTPUT_DIR/'res_grad_true_inner_lr_{1}_soft_temp_{2}' \
    ::: 0.3 0.1 0.03 0.01 0.003 0.001 \
    ::: 1.0 100.0 10000.0
