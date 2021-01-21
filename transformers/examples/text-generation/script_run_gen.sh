#/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/output_medium1/checkpoint-65000
/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/output2_medium_data3/checkpoint-45000

export TRAIN_FILE=/u/scr/xlisali/contrast_LM/data/test.txt
export TEST_FILE=/u/scr/xlisali/contrast_LM/data/dev.txt
#export MODEL_FILE=/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/output_medium1
#export MODEL_FILE=/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/output2_medium_data3/checkpoint-45000
# The following is the story data
#export MODEL_FILE=/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/output2_medium_data3

# The following is the general data
#export MODEL_FILE=/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/output2_medium_data3/checkpoint-100000

# another set of problem: generation based on embedding matching.
# this is for the Saturday code.
#export MODEL_FILE=/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/temp_medium_matching_secondtrial
# this is for the Sunday clean bert code.
#/checkpoint-25000
export MODEL_FILE=/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/temp_medium_matching_cleanbert
#export MODEL_FILE=/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/temp_medium_matching_layer5/checkpoint-50000
export Token_FILE=/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/temp_medium_matching_cleanbert
# this is for the Sunday 5-layer code.


#
#export TRAIN_FILE=/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/debug_train.txt
#export TEST_FILE=/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/debug_dev.txt
#export MODEL_FILE=/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/output

python run_generation.py \
    --model_type=gpt2 \
    --length 100 \
    --model_name_or_path=$MODEL_FILE \
    --num_return_sequences 5 \
    --stop_token [EOS] \
    --tokenizer_name=$Token_FILE \

#echo $COMMANDLINE
#nlprun -a lisa-base-torch -n medium-finetune -m jagupard27 '$COMMANDLINE'

