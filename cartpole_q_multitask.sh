#! /bin/bash


NUM_JOBS=1

    
OUTPUT_DIR='maml-q-cartpole-multitask-2goals-5'    
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
        --fast-lr {1} \
        --inner-q-residue-gradient {2} \
        --seed {3} \
        --output-folder $OUTPUT_DIR/'inner_lr_{1}_res_grad_{2}_seed_{3}' \
    ::: 0.03 0.015 0.01 0.007 \
    ::: true false \
    ::: 42 24 79 87
    
