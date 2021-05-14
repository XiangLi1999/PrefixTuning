#!/bin/bash
CACHE=/scr/eix/cache
mkdir -p $CACHE
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE

CACHE_DIR=/tiger/u/eix/prefix-tuning/cache
mkdir -p $CACHE_DIR

source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh
conda activate eix-prefix

DROPOUT=$1

python train_e2e.py \
    --mode webnlg \
    --tuning_mode finetune \
    --cache_dir $CACHE_DIR \
    --use_custom_teacher_dropout yes \
    --dropout $DROPOUT \
    --bsz 5
