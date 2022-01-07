#!/bin/bash

# EXAMPLE USAGE:
# CUDA_VISIBLE_DEVICES=0 ./emotion_generative.sh 0 4 cuda
# CUDA_VISIBLE_DEVICES=1 ./emotion_generative.sh 5 9 cuda
# CUDA_VISIBLE_DEVICES=-1 ./emotion_generative.sh 10 14 cpu
# CUDA_VISIBLE_DEVICES=-1 ./emotion_generative.sh 15 19 cpu
# CUDA_VISIBLE_DEVICES=-1 ./emotion_generative.sh 20 24 cpu
# CUDA_VISIBLE_DEVICES=-1 ./emotion_generative.sh 25 29 cpu
# CUDA_VISIBLE_DEVICES=-1 ./emotion_generative.sh 30 34 cpu
# CUDA_VISIBLE_DEVICES=-1 ./emotion_generative.sh 35 39 cpu

PYTHON=python3.7

K=10
SEQ_LEN=2048
VOCAB_SIZE=391
N_LAYERS=8
EMOTIONS=(0 1 2 3)
LM=../trained/language_model_43.pth
CLF=../trained/emotion_classifier_1.pth
SAVE_TO=results/generative/mcts_piece
MCTS=../generate_mcts.py

if [ -z "$1" ]; then
    echo "Provide pieces range start."
    exit 1
fi

if [ -z "$2" ]; then
    echo "Provide pieces range end."
    exit 1
fi

if [ -z "$3" ]; then
    echo "Provide device's name."
    exit 1
fi

FIRST_PIECE_IDX=$1
LAST_PIECE_IDX=$2
DEVICE=$3

for emotion in ${EMOTIONS[@]}; do
    echo "-> Emotion $emotion"
    for i in `seq $FIRST_PIECE_IDX $LAST_PIECE_IDX`; do
        echo "--- Generating piece $i"
        $PYTHON $MCTS --lm $LM --clf $CLF --emotion $emotion --vocab_size $VOCAB_SIZE \
                        --seq_len $SEQ_LEN --n_layers $N_LAYERS --k $K \
                        --save_to ${SAVE_TO}_${emotion}_${i}.mid --device $DEVICE

    done
done
