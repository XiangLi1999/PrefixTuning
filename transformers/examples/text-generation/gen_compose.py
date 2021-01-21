import os, sys

# example: python train_run.py keyword temp_keyword _
if __name__ == '__main__':
    mode = sys.argv[1]
    control_mode = sys.argv[2]
    use_prefixtuning = (sys.argv[3] == 'yes')
    model_file = None
    old_model = None
    MODEL_FILE = sys.argv[4]
    MODEL_FILE_second = sys.argv[5]
    split_file = sys.argv[6]

    if mode == 'embMatch' and not use_prefixtuning:
        MODEL_FILE="/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/temp_medium_matching_cleanbert"
        Token_FILE="/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/temp_medium_matching_cleanbert"
        SENT_FILE = '/u/scr/xlisali/contrast_LM/data_api/dataset/matching_train_small.txt'
        SENT_FILE='/u/scr/xlisali/contrast_LM/data_api/dataset/matching_dev_small.txt'
        if model_file:
            Model_FILE = model_file
        else:
            Model_FILE="medium_matching"

        if old_model != '_':
            OLD_MODEL=old_model
        else:
            OLD_MODEL="gpt2-medium"

    if mode == 'keyword' and not use_prefixtuning:
        if control_mode == 'no':
            MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/temp_keyword"
            Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/temp_keyword"
        else:
            # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/keyword_temp"
            # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/keyword_temp"

            # mid
            MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/keyword_temp2/checkpoint-90000"
            Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/temp_keyword"
            #
            # MODEL_FILE = "gpt2-medium"
            # Token_FILE = "gpt2-medium"

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
        if control_mode == 'no':
            pass
            # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/temp_topic"
            # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/temp_topic"
        else:
            # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/topic_temp"
            # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/topic_temp"

            # MODEL_FILE = "gpt2-medium"
            # Token_FILE = "gpt2-medium"

            # mid
            # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/topic_temp2"
            # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/topic_temp2"

            MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/topicprefixtune"
            Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/topicprefixtune"


        TRAIN_FILE = None
        TEST_FILE = None


        if 'finetune' in MODEL_FILE:
            tuning_mode = 'finetune'
            app = ''
        elif 'prefixtune' in MODEL_FILE:
            tuning_mode = 'prefixtune'
            # if 'inputpara' not in MODEL_FILE and "-emb-" not in MODEL_FILE:
            #     app = '--optim_prefix {} --preseqlen {}'.format('yes', 10)
            # else:
            app = '--optim_prefix {} --preseqlen {} '.format('no', 10)
            if "-emb" in MODEL_FILE:
                app += "--prefix_mode embedding "
            MODEL_FILE2 = MODEL_FILE
            MODEL_FILE = 'gpt2-medium'


    elif mode =='data2text':


        Token_FILE = MODEL_FILE

        if 'finetune' in MODEL_FILE:
            tuning_mode = 'finetune'
            app = ''

        elif 'prefixtune' in MODEL_FILE:
            tuning_mode = 'prefixtune'
            if "_y_20" in MODEL_FILE:
                app = '--optim_prefix {} --preseqlen {} '.format('yes', 20)
            else:
                app = '--optim_prefix {} --preseqlen {} '.format('no', 20)
            if "_emb" in MODEL_FILE:
                app += "--prefix_mode embedding "
            elif "_act" in MODEL_FILE:
                app += "--prefix_mode activation "
            if "_inf" in MODEL_FILE or 'infix' in MODEL_FILE:
                app += " --format_mode infix "
            elif "_cat" in MODEL_FILE:
                app += " --format_mode cat "
            elif "_pee" in MODEL_FILE:
                app += " --format_mode peek "

            MODEL_FILE2 = MODEL_FILE
            MODEL_FILE2_second = MODEL_FILE_second
            MODEL_FILE = 'gpt2-medium'




    elif mode == 'triples':
        Token_FILE = MODEL_FILE

        if 'finetune' in MODEL_FILE:
            tuning_mode = 'finetune'
            app = ''
        elif 'prefixtune' in MODEL_FILE:
            tuning_mode = 'prefixtune'
            if "tune_y_" in MODEL_FILE:
                app = '--optim_prefix {} --preseqlen {} '.format('yes', 20)
            else:
                app = '--optim_prefix {} --preseqlen {} '.format('no', 20)
            if "_emb" in MODEL_FILE:
                app += "--prefix_mode embedding "
            elif "_act" in MODEL_FILE:
                app += "--prefix_mode activation "
            if "_inf" in MODEL_FILE or 'infix' in MODEL_FILE:
                app += " --format_mode infix "
            elif "_cat" in MODEL_FILE:
                app += " --format_mode cat "
            elif "_pee" in MODEL_FILE:
                app += " --format_mode peek "

            MODEL_FILE2 = MODEL_FILE
            MODEL_FILE = 'gpt2-medium'

    elif mode == 'webnlg':
        Token_FILE = MODEL_FILE

        if 'finetune' in MODEL_FILE:
            tuning_mode = 'finetune'
            app = ''
        elif 'prefixtune' in MODEL_FILE:
            tuning_mode = 'prefixtune'
            if "tune_y_" in MODEL_FILE:
                app = '--optim_prefix {} --preseqlen {} '.format('yes', 20)
            else:
                app = '--optim_prefix {} --preseqlen {} '.format('no', 20)
            if "_emb" in MODEL_FILE:
                app += "--prefix_mode embedding "
            elif "_act" in MODEL_FILE:
                app += "--prefix_mode activation "
            if "_inf" in MODEL_FILE or 'infix' in MODEL_FILE:
                app += " --format_mode infix "
            elif "_cat" in MODEL_FILE:
                app += " --format_mode cat "
            elif "_pee" in MODEL_FILE:
                app += " --format_mode peek "

            MODEL_FILE2 = MODEL_FILE
            MODEL_FILE = 'gpt2-medium'


    COMMANDLINE = "python run_compose.py \
        --model_type=gpt2 \
        --length 100 \
        --model_name_or_path={} \
        --num_return_sequences 5 \
        --stop_token [EOS] \
        --tokenizer_name={} \
        --task_mode={} \
        --control_mode={} --tuning_mode {} --split_file {}\
    ".format(MODEL_FILE, Token_FILE, mode, control_mode, tuning_mode, split_file)

    COMMANDLINE += app

    if tuning_mode == 'prefixtune':
        COMMANDLINE += ' --prefixModel_name_or_path {} --prefixModel_name_or_path2 {} '.format(MODEL_FILE2, MODEL_FILE2_second)
    else:
        COMMANDLINE += ' --model_name_or_path2 {} '.format(MODEL_FILE_second)


    os.system(COMMANDLINE)
    # name = os.path.basename(MODEL_FILE2)
    # name = 'e2e_results_new/{}'.format(name)
    # full_command = "nlprun -a lisa-base-torch -g 1 -n {} -x jagupard4,jagupard5,jagupard6,jagupard7,jagupard8 \'{}\'".format(name,COMMANDLINE)
    # print(full_command)
    # os.system(full_command)






