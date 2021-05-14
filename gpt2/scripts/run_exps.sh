#!/bin/bash

mkdir -p logs

for DROPOUT in 0.1 0.2 0.3 0.4 0.5
do
sbatch \
    --gres=gpu:1 \
    --partition tiger-lo \
    --output logs/teacher_${DROPOUT}.log \
    --mem 32g \
    scripts/train_teacher.sh $DROPOUT
done
