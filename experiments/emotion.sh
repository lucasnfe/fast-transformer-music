#!/bin/bash

PYTHON=python3.7

SEQ_LEN=2048
VOCAB_SIZE=391
N_LAYERS=8
BATCH_SIZE=16
LR=1e-4
PREFIX=64
EPOCHS=100

TRAINER=../train_emotion_classifier.py
DATA=../vgmidi
MODEL=../trained/language_model_43.pth
SAVE_TO=../trained/

if [ -z "$1" ]; then
    echo "Provide an experiment range."
    exit 1
fi

if [ -z "$2" ]; then
    echo "Provide a file path to save models."
    exit 1
fi

DATA_RANGE=$1
OUTPUT_PATH=$2

for i in `seq 0 $DATA_RANGE`; do
    TEST_SET=${DATA}/vgmidi_labelled_${i}_test.csv
    TRAIN_SET=${DATA}/vgmidi_labelled_${i}_train.csv

    echo "Experiment $i: $TRAIN_SET, $TEST_SET"

    $PYTHON $TRAINER --train $TRAIN_SET --test $TEST_SET --epochs $EPOCHS \
                     --seq_len $SEQ_LEN --vocab_size $VOCAB_SIZE \
                     --model $MODEL --n_layers $N_LAYERS --batch_size $BATCH_SIZE \
                     --lr $LR --prefix $PREFIX --save_to ${SAVE_TO}/${OUTPUT_PATH}_${i}.pth
done
