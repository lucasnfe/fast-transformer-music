#!/bin/bash

# EXAMPLE USAGE:
# CUDA_VISIBLE_DEVICES=0 nohup ./emotion_generative.sh 0 4 cuda > emotion_generative_0_4_cuda.out &
# CUDA_VISIBLE_DEVICES=1 nohup ./emotion_generative.sh 5 9 cuda > emotion_generative_5_9_cuda.out &
# CUDA_VISIBLE_DEVICES=-1 nohup ./emotion_generative.sh 10 14 cpu > emotion_generative_10_14_cpu.out &
# CUDA_VISIBLE_DEVICES=-1 nohup ./emotion_generative.sh 15 19 cpu > emotion_generative_15_19_cpu.out &
# CUDA_VISIBLE_DEVICES=-1 nohup ./emotion_generative.sh 20 24 cpu > emotion_generative_20_24_cpu.out &
# CUDA_VISIBLE_DEVICES=-1 nohup ./emotion_generative.sh 25 29 cpu > emotion_generative_25_29_cpu.out &
# CUDA_VISIBLE_DEVICES=-1 nohup ./emotion_generative.sh 30 34 cpu > emotion_generative_30_34_cpu.out &
# CUDA_VISIBLE_DEVICES=-1 nohup ./emotion_generative.sh 35 39 cpu > emotion_generative_35_39_cpu.out &

PYTHON=python3.7

K=8
C=2.0
SEQ_LEN=2048
VOCAB_SIZE=391
N_LAYERS=8
ROLL_STEPS=128
EMOTIONS=(0 1 2 3)
LM=../trained/language_model_43.pth
CLF=../trained/emotion_classifier_128prefix_7.pth
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
                        --seq_len $SEQ_LEN --gen_len ${GEN_LEN} --n_layers $N_LAYERS --k $K --c $C \
                        --save_to ${SAVE_TO}_${emotion}_${i}.mid --device $DEVICE \
                        --roll_steps ${ROLL_STEPS}

    done
done
