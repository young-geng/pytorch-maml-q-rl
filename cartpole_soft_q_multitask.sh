#! /bin/bash

NUM_JOBS=20

OUTPUT_DIR='maml-soft-q-cartpole-multitask-2goals-1'
parallel -j $NUM_JOBS \
    'CUDA_VISIBLE_DEVICES=$(({%} % 4))' python main.py \
        --env-name CartPoleMultitask-v0 \
        --num-workers 16 \
        --max-kl 0.01 \
        --fast-batch-size 20 \
        --meta-batch-size 40 \
        --num-layers 2 \
        --hidden-size 100 \
        --num-batches 300 \
        --gamma 0.99 \
        --tau 1.0 \
        --cg-damping 1e-5 \
        --ls-max-steps 15 \
        --inner-q true \
        --device cuda \
        --inner-steps 8 \
        --inner-q-soft true \
        --fast-lr {1} \
        --inner-q-soft-temp {2} \
        --inner-q-residue-gradient {3} \
        --seed {4} \
        --output-folder $OUTPUT_DIR/'inner_lr_{1}_soft_temp_{2}_res_grad_{3}_seed_{4}' \
    ::: 0.3 0.1 0.03 0.01 0.003 0.001 \
    ::: 1.0 100.0 10000.0 \
    ::: true false \
    ::: 42 24 79 87
