import os, sys

# example: python train_run.py keyword temp_keyword _
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

if __name__ == '__main__':
    mode = sys.argv[1]
    model_file = sys.argv[2]
    old_model = sys.argv[3]
    bsz = sys.argv[4]
    dataless = sys.argv[5]
    load_prefix_model = sys.argv[6]
    tuning_mode = sys.argv[7]
    use_big = sys.argv[8]

    assert tuning_mode in ['finetune', 'prefixtune', 'finetune-top']

    if dataless == 'yes':
        print('A BIG WARNING, dataless=True')
        print('WARNING '*50 )

    if mode == 'embMatch':
        TRAIN_FILE='/u/scr/xlisali/contrast_LM/data_api/dataset/matching_train_small.txt'
        TEST_FILE='/u/scr/xlisali/contrast_LM/data_api/dataset/matching_dev_small.txt'

        # TRAIN_FILE = "/u/scr/xlisali/contrast_LM/data_api/dataset/matching_debug_0.txt"
        # TEST_FILE = "/u/scr/xlisali/contrast_LM/data_api/dataset/matching_debug_0.txt"
        if model_file:
            Model_FILE = model_file
        else:
            Model_FILE="medium_matching"

        if old_model != '_':
            OLD_MODEL=old_model
        else:
            OLD_MODEL="gpt2-medium"

    if mode == 'keyword':
        # TRAIN_FILE = "/u/scr/xlisali/contrast_LM/data_api/dataset/matching_debug_0.txt"
        # TEST_FILE = "/u/scr/xlisali/contrast_LM/data_api/dataset/matching_debug_0.txt"

        TRAIN_FILE='/u/scr/xlisali/contrast_LM/data_api/dataset/matching_train_small.txt'
        TEST_FILE='/u/scr/xlisali/contrast_LM/data_api/dataset/matching_dev_small.txt'
        if model_file:
            Model_FILE = model_file
        else:
            Model_FILE="medium_matching"

        if old_model != '_':
            OLD_MODEL=old_model
        else:
            OLD_MODEL="gpt2-medium"


    if mode == 'multiple_keywords':
        # TRAIN_FILE = "/u/scr/xlisali/contrast_LM/data_api/dataset/matching_debug_0.txt"
        # TEST_FILE = "/u/scr/xlisali/contrast_LM/data_api/dataset/matching_debug_0.txt"

        TRAIN_FILE='/u/scr/xlisali/contrast_LM/data_api/dataset/matching_train_small.txt'
        TEST_FILE='/u/scr/xlisali/contrast_LM/data_api/dataset/matching_dev_small.txt'
        if model_file:
            Model_FILE = model_file
        else:
            Model_FILE="medium_matching"

        if old_model != '_':
            OLD_MODEL=old_model
        else:
            OLD_MODEL="gpt2-medium"

    elif mode == 'topic':
        TRAIN_FILE="/u/scr/xlisali/contrast_LM/transformers/examples/text-classification/glue_data/AG-news/train1.tsv"
        TEST_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/text-classification/glue_data/AG-news/dev1.tsv"
        if model_file:
            Model_FILE = model_file
        else:
            Model_FILE="medium_topic"

        if old_model != '_':
            OLD_MODEL=old_model
        else:
            OLD_MODEL="gpt2-medium"

    elif mode == 'length':
        # TRAIN_FILE = '/u/scr/xlisali/contrast_LM/data_api/dataset/matching_train_small.txt'
        # TEST_FILE = '/u/scr/xlisali/contrast_LM/data_api/dataset/matching_dev_small.txt'
        TRAIN_FILE = "/u/scr/xlisali/contrast_LM/data_api/dataset/matching_debug_0.txt"
        TEST_FILE = "/u/scr/xlisali/contrast_LM/data_api/dataset/matching_debug_0.txt"
        if model_file:
            Model_FILE = model_file
        else:
            Model_FILE="medium_topic"

        if old_model != '_':
            OLD_MODEL=old_model
        else:
            OLD_MODEL="gpt2-medium"

    elif mode == 'debug':
        TRAIN_FILE="/u/scr/xlisali/contrast_LM/data_api/dataset/matching_debug_0.txt"
        TEST_FILE="/u/scr/xlisali/contrast_LM/data_api/dataset/matching_debug_0.txt"

        if model_file:
            Model_FILE = model_file
        else:
            Model_FILE="medium_debug"

        if old_model != '_':
            OLD_MODEL=old_model
        else:
            OLD_MODEL="gpt2-medium"

    elif mode =='data2text':
        TRAIN_FILE = "/u/scr/xlisali/e2e_data/src1_train.txt"
        TEST_FILE = "/u/scr/xlisali/e2e_data/src1_valid.txt"
        if model_file:
            Model_FILE = model_file
        else:
            Model_FILE="medium_data2text"

        if old_model != '_':
            OLD_MODEL=old_model
        else:
            OLD_MODEL="gpt2-medium"

        # when we have input-based AND activation level AND infix.
        # app = "--optim_prefix {} --preseqlen {} --prefix_mode {} --format_mode {} " \
        #       "--gradient_accumulation_steps 1 ".format('no', 15, 'activation', 'infix')
        # Try to replicate the best performing version of cat.
        app = "--optim_prefix {} --preseqlen {} --prefix_mode {} --format_mode {} " \
              "--gradient_accumulation_steps 1 ".format('no', 15, 'activation', 'cat')

        # app = "--optim_prefix {} --preseqlen {} --prefix_mode {} --format_mode {} " \
        #       "--gradient_accumulation_steps 1 ".format('peek', 15, 'activation', 'cat')

        # We have instruction based AND embedding AND peek.
        # app = "--optim_prefix {} --preseqlen {} --prefix_mode {} --format_mode {}".format('yes', 20, 'embedding', 'peek')

        # We have instruction based AND activation AND peek.
        # app = "--optim_prefix {} --preseqlen {} --prefix_mode {} --format_mode {}".format('yes', 15, 'activation', 'peek')

    elif mode =='lemma2text':
        TRAIN_FILE = "/u/scr/xlisali/ud_en/gpt2-train.txt"
        TEST_FILE = "/u/scr/xlisali/ud_en/gpt2-dev.txt"
        if model_file:
            Model_FILE = model_file
        else:
            Model_FILE="medium_data2text"

        if old_model != '_':
            OLD_MODEL=old_model
        else:
            OLD_MODEL="gpt2-medium"

    elif mode =='text2data':
        TRAIN_FILE = "/u/scr/xlisali/e2e_data/src1_train.txt"
        TEST_FILE = "/u/scr/xlisali/e2e_data/src1_valid.txt"
        if model_file:
            Model_FILE = model_file
        else:
            Model_FILE="medium_text2data"

        if old_model != '_':
            OLD_MODEL=old_model
        else:
            OLD_MODEL="gpt2-medium"

    elif mode == 'dataless':

        TRAIN_FILE = None
        TEST_FILE = None
        if model_file:
            Model_FILE = model_file
        else:
            Model_FILE = "medium_dataless"

        if old_model != '_':
            OLD_MODEL = old_model
        else:
            OLD_MODEL = "gpt2-medium"

    elif mode == 'gen_data':

        if 'topic' in model_file:
            TRAIN_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/len_trial-data-topic/gptgen_sentiment_train.txt"
            # TRAIN_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/len_trial-data-topic/gptgen_sentiment_valid.txt"
            # TRAIN_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/len_trial-data/gptgen_sentiment.txt"
            TEST_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/len_trial-data-topic/gptgen_sentiment_valid.txt"
        elif 'sent' in model_file:
            TRAIN_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/len_trial-data/gptgen_sentiment_train.txt"
            TEST_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/len_trial-data/gptgen_sentiment_valid.txt"

        if model_file:
            Model_FILE = model_file
        else:
            Model_FILE = "medium_topic_gen"

        if old_model != '_':
            OLD_MODEL = old_model
        else:
            OLD_MODEL = "gpt2-medium"

    elif mode == 'generate':
        TRAIN_FILE = '_'
        TEST_FILE = '_'

        if model_file:
            Model_FILE = model_file
        else:
            Model_FILE = "medium_topic_gen"

        if old_model != '_':
            OLD_MODEL = old_model
        else:
            OLD_MODEL = "gpt2-medium"

    controlprefix = ('yes' if tuning_mode == 'prefixtune' else 'no')

    COMMANDLINE="python run_language_modeling.py \
        --output_dir={} \
        --model_type=gpt2 \
        --model_name_or_path={} \
        --tokenizer_name=gpt2-medium \
        --per_device_train_batch_size {} \
        --per_device_eval_batch_size {} \
        --save_steps 5000 \
        --num_train_epochs 5 \
        --do_train \
        --train_data_file={} \
        --do_eval \
        --line_by_line \
        --save_total_limit 0 \
        --overwrite_output_dir \
        --task_mode {} \
        --eval_data_file={}  \
        --dataless {} --tuning_mode {} --controlprefix {} \
        --train_embs no ".format(Model_FILE, OLD_MODEL, bsz, bsz, TRAIN_FILE, mode, TEST_FILE,
                                 dataless, tuning_mode, controlprefix)

    COMMANDLINE += app
    # LOAD_TRAIN_PREFIX = None
    # LOAD_TRAIN_PREFIX = '/u/scr/xlisali/contrast_LM/transformers/examples/control/len_trial/checkpoint-1000'
    if load_prefix_model == 'yes':
        LOAD_TRAIN_PREFIX = '/u/scr/xlisali/contrast_LM/transformers/examples/control/med_topic_gen'
        COMMANDLINE += '--prefixModel_name_or_path {} '.format(LOAD_TRAIN_PREFIX)

    DATALESS = (dataless =='yes')

    if DATALESS:
        # use gumbel
        COMMANDLINE += '--dataless_sample_size {} ' \
                       '--dataless_sample_length {} ' \
                       '--dataless_usebaseline {} ' \
                       '--dataless_control_type {} ' \
                       '--gradient_accumulation_steps 1 ' \
                       '--gumbel {} ' \
                       '--replay_buffer {} ' \
                       '--training_obj {} ' \
                       '--dataless_discri_model_path {}'.format(5, 50, 'yes', 0, 'yes', 'yes', 0, 'textattack/roberta-base-imdb') # 2 for sentiment;; 3 for length.
                       # '--dataless_discri_model_path {}'.format(4, 60, 'yes', 2, 'no', 'yes', 0, 'textattack/roberta-base-ag-news')  # 2 for sentiment;; 3 for length.
    # '--dataless_discri_model_path {}'.format(8, 60, 'yes', 2, 'no', 'yes', 0, 'textattack/roberta-base-imdb') # 2 for sentiment;; 3 for length.
    os.system(COMMANDLINE) # textattack/roberta-base-ag-news # textattack/roberta-base-imdb
    # #
    # print(use_big)
    # if use_big == 'no':
    #     full_command = "nlprun -a lisa-base-torch -g 1 -n {} -x jagupard4,jagupard5,jagupard6,jagupard7,jagupard8 \'{}\'".format(Model_FILE, COMMANDLINE)
    # elif True:
    #     full_command = "nlprun -p high -a lisa-base-torch -g 1 -n {} -x jagupard4,jagupard5,jagupard6,jagupard7,jagupard8," \
    #                    "jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18," \
    #                    "jagupard19,jagupard20,jagupard21,jagupard22,jagupard23," \
    #                    "jagupard24,jagupard25 \'{}\'".format(Model_FILE, COMMANDLINE)
    # else:
    #     full_command = "nlprun -a lisa-base-torch -m jagupard26 -p high -g 1 -n {} \'{}\'".format(Model_FILE, COMMANDLINE)
    #
    # print(full_command)
    # os.system(full_command)




#
#nlprun -a lisa-base-torch -g 1 --p high -n 5top-finetune -x jagupard4,jagupard5,jagupard6,jagupard7,jagupard8,jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25 '$COMMANDLINE'

#nlprun -a lisa-base-torch -g 1 -n topk-medium-finetune -m jagupard26 '$COMMANDLINE'










################
# python train_run.py generate topichaha _ 10 no no _
##############
# keyword training.
# python train_run.py keyword keywordhaha _ 10 no no _
##############
# Embedding training.
# python train_run.py embMatch embMatchhaha _ 10 no no _
##############
# Selective Classifier training.
# python train_run.py gen_data topichaha_s _ 10 no no no

##############
# Finetune keyword training.
# python train_run.py keyword keywordfinetune _ 1 no no yes
# python train_run.py embMatch embMatchfinetune _ 10 no no no



###########
# FINETUNE Data2Text
# python train_run.py data2text data2textfinetune _ 10 no no finetune no

# python train_run.py data2text data2textprefixtune10 _ 10 no no prefixtune no

# python train_run.py dataless dataless2 _ 10 yes no prefixtune no


###############
# python train_run.py topic topicprefixtune _ 3 no no prefixtune no
# python train_run.py topic topicfinetune _ 2 no no finetune no

# python train_run.py keyword keywordfinetune _ 4 no no finetune no
# python train_run.py keyword keywordprefixtune _ 4 no no prefixtune no


###########################

## python train_run.py data2text data2textfinetune-top _ 10 no no finetune-top no

## python train_run.py data2text data2textprefixtune-emb _ 10 no no prefixtune no (cat)

## python train_run.py data2text data2textprefixtune-emb-peek _ 10 no no prefixtune no (peek)

# ## python train_run.py data2text data2textprefixtune-emb-nopeek _ 10 no no prefixtune no (nopeek)

# ## python train_run.py data2text data2textprefixtune-emb-para _ 10 no no prefixtune no (instruction)

## python train_run.py data2text data2textprefixtune15 _ 10 no no prefixtune no (instruction based activation)

# python train_run.py data2text data2textprefixtune-infix-emb-cat2 _ 10 no no prefixtune no (peek)
# python train_run.py data2text data2textprefixtune-infix-latent-cat2 _ 10 no no prefixtune no


###########################
# python train_run.py data2text data2textprefixtune-no-act-cat-rep _ 10 no no prefixtune no (cat, epoch=6)
# python train_run.py data2text data2textprefixtune-no-act-cat-rep2 _ 10 no no prefixtune no (cat, epoch=5)

















    #
    #
    # if False:
    #     COMMANDLINE += '--dataless_sample_size {} ' \
    #                    '--dataless_sample_length {} ' \
    #                    '--dataless_usebaseline {} ' \
    #                    '--dataless_control_type {} ' \
    #                    '--gradient_accumulation_steps 20 ' \
    #                    '--gumbel {} ' \
    #                    '--replay_buffer {} ' \
    #                    '--training_obj {} ' \
    #                    '--dataless_discri_model_path {}'.format(8, 60, 'yes', 2, 'no', 'yes', 0, 'textattack/roberta-base-imdb') # 2 for sentiment;; 3 for length.
    #                    # '--dataless_discri_model_path {}'.format(4, 60, 'yes', 2, 'no', 'yes', 0, 'textattack/roberta-base-ag-news')  # 2 for sentiment;; 3 for length.
