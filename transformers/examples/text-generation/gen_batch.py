import os, sys

# example: python train_run.py keyword temp_keyword _
if __name__ == '__main__':
    mode = sys.argv[1]
    control_mode = sys.argv[2]
    use_prefixtuning = (sys.argv[3] == 'yes')
    model_file = None
    old_model = None
    MODEL_FILE = sys.argv[4]
    submit_job = (sys.argv[5] == 'yes')

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

    elif mode == 'topic_old':
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

    elif mode == 'length' and not use_prefixtuning:
        if control_mode == 'no':
            MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/temp_length"
            Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/temp_length"
        else:
            # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/length_temp"
            # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/length_temp"
            # mid
            MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/length_temp2/checkpoint-90000"
            Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/length_temp"

            # MODEL_FILE = "gpt2-medium"
            # Token_FILE = "gpt2-medium"


        TRAIN_FILE = '/u/scr/xlisali/contrast_LM/data_api/dataset/matching_train_small.txt'
        TEST_FILE = '/u/scr/xlisali/contrast_LM/data_api/dataset/matching_dev_small.txt'
        if model_file:
            Model_FILE = model_file
        else:
            Model_FILE="medium_length"

        if old_model != '_':
            OLD_MODEL=old_model
        else:
            OLD_MODEL="gpt2-medium"

    elif mode == 'debug' and not use_prefixtuning:
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
        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/temp_data2text"
        # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/temp_data2text"

        ################## finetune full models.
        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textfinetune"
        # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textfinetune"

        ################## finetune only the top layers.
        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textfinetune-top"
        # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textfinetune-top"

        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textfinetune-top1"
        # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textfinetune-top1"

        # MODEL_FILE = "/u/scr/xlisal/i/contrast_LM/transformers/examples/control/data2textfinetune-top2"
        # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textfinetune-top2"

        ################## instruction based prefix tuning.
        # preseqlen = 15
        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune{}".format(preseqlen)
        # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune{}".format(preseqlen)




        ################## prefix Tuning different prefix modes.
        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune-inputpara-nopeek"
        # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune-inputpara-nopeek"

        # BEST one!!!
        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune-inputpara-cat2"
        # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune-inputpara-cat2"

        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune-inputpara-peek"
        # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune-inputpara-peek"

        ############### prefix Embedding Tuning with different prefix modes.
        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune-emb-cat2"
        # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune-emb-cat2"

        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune-emb-peek"
        # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune-emb-peek"

        ############### infix
        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune-infix-latent-cat2"
        # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune-infix-latent-cat2"

        ############ replication.  5 epochs + dropouts.
        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune-no-act-cat-rep"
        # Token_FILE = "/u/scr/xlisali/contrast_LM/transformepythors/examples/control/data2textprefixtune-no-act-cat-rep"

        ########## replication. 5 epochs.
        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune-no-act-cat-rep2"
        # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune-no-act-cat-rep2"

        ######## hyperparam tuning (batch size)
        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune_n_20_act_cat_b=20-e=5"
        # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune_n_20_act_cat_b=20-e=5"

        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune_n_20_act_cat_b=10-e=10"
        # Token_FILE = MODEL_FILE
        #
        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune_n_20_act_cat_b=10-e=5"
        # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune_n_20_act_cat_b=10-e=5"

        #
        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune_n_20_act_cat_b=30-e=5"
        # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune_n_20_act_cat_b=30-e=5"

        ######## hyperparam tuning (peek)
        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune_n_20_act_inf_b=10-e=5"
        # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune_n_20_act_inf_b=10-e=5"


        ######### hyper-param tuning (learning rate.)
        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune_n_20_act_cat_b=10-e=5_lr=0.001_w=1e-05_s=101"
        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune_n_20_act_cat_b=10-e=5_lr=0.0005_w=1e-05_s=101"

        #########
        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune_n_20_act_pee_b=10-e=5"
        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune_n_20_act_pee_b=30-e=5"
        # Token_FILE = MODEL_FILE

        # /u/scr/xlisali/contrast_LM/transformers/examples/control/save_e2e_models

        Token_FILE = MODEL_FILE

        gen_dir = 'e2e_results_conv'
        sub_model_name = os.path.basename(MODEL_FILE)
        if 'checkpoint-' in sub_model_name:
            sub_model_name =  MODEL_FILE

        if 'finetune' in sub_model_name:
            tuning_mode = 'finetune'
            app = ''
        elif 'prefixtune' in sub_model_name:
            tuning_mode = 'prefixtune'
            if "_y_20" in sub_model_name:
                app = '--optim_prefix {} --preseqlen {} '.format('yes', 20)
            else:
                app = '--optim_prefix {} --preseqlen {} '.format('no', 20)
            if "_emb" in sub_model_name:
                app += "--prefix_mode embedding "
            elif "_act" in sub_model_name:
                app += "--prefix_mode activation "
            if "_inf" in sub_model_name or 'infix' in sub_model_name:
                app += " --format_mode infix "
            elif "_cat" in sub_model_name:
                app += " --format_mode cat "
            elif "_pee" in sub_model_name:
                app += " --format_mode peek "

            MODEL_FILE2 = MODEL_FILE

            if 'large' in sub_model_name:
                MODEL_FILE = 'gpt2-large'
            else:
                MODEL_FILE = 'gpt2-medium'

        elif 'bothtune' in sub_model_name:
            tuning_mode = 'bothtune'
            if "_y_20" in sub_model_name:
                app = '--optim_prefix {} --preseqlen {} '.format('yes', 20)
            else:
                app = '--optim_prefix {} --preseqlen {} '.format('no', 20)
            if "_emb" in sub_model_name:
                app += "--prefix_mode embedding "
            elif "_act" in sub_model_name:
                app += "--prefix_mode activation "
            if "_inf" in sub_model_name or 'infix' in sub_model_name:
                app += " --format_mode infix "
            elif "_cat" in sub_model_name:
                app += " --format_mode cat "
            elif "_pee" in sub_model_name:
                app += " --format_mode peek "

            MODEL_FILE2 = MODEL_FILE
            MODEL_FILE = os.path.join(MODEL_FILE, 'gpt2')

        elif 'adaptertune' in sub_model_name:
            tuning_mode = 'adaptertune'
            app = ''
            # if "_y_20" in sub_model_name:
            #     app = '--optim_prefix {} --preseqlen {} '.format('yes', 20)
            # else:
            #     app = '--optim_prefix {} --preseqlen {} '.format('no', 20)
            # if "_emb" in sub_model_name:
            #     app += "--prefix_mode embedding "
            # elif "_act" in sub_model_name:
            #     app += "--prefix_mode activation "
            # if "_inf" in sub_model_name or 'infix' in sub_model_name:
            #     app += " --format_mode infix "
            # elif "_cat" in sub_model_name:
            #     app += " --format_mode cat "
            # elif "_pee" in sub_model_name:
            #     app += " --format_mode peek "



        # if 'finetune' in MODEL_FILE:
        #     tuning_mode = 'finetune'
        #     app = ''
        # elif 'prefixtune' in MODEL_FILE:
        #     tuning_mode = 'prefixtune'
        #     if "_y_20" in MODEL_FILE:
        #         app = '--optim_prefix {} --preseqlen {}'.format('yes', 20)
        #     elif 'inputpara' not in MODEL_FILE and "-emb-" not in MODEL_FILE and 'infix' not in MODEL_FILE \
        #             and '-no' not in MODEL_FILE and  '_n' not in MODEL_FILE:
        #         app = '--optim_prefix {} --preseqlen {}'.format('yes', preseqlen)
        #     else:
        #         app = '--optim_prefix {} --preseqlen {} '.format('no', 20)
        #     if "-emb" in MODEL_FILE:
        #         app += "--prefix_mode embedding "
        #     if "_inf" in MODEL_FILE or 'infix' in MODEL_FILE:
        #         app += " --format_mode infix"
        #     MODEL_FILE2 = MODEL_FILE
        #     MODEL_FILE = 'gpt2-medium'



        # TRAIN_FILE = "/u/scr/xlisali/e2e_data/src1_train.txt"
        # TEST_FILE = "/u/scr/xlisali/e2e_data/src1_valid.txt"
        # if model_file:
        #     Model_FILE = model_file
        # else:
        #     Model_FILE="medium_data2text"
        #
        # if old_model != '_':
        #     OLD_MODEL=old_model
        # else:
        #     OLD_MODEL="gpt2-medium"

    elif mode == 'writingPrompts' or mode == 'sentiment' or mode == 'topic':
        Token_FILE = MODEL_FILE
        if mode == 'writingPrompts':
            gen_dir = 'wp_results'
        else:
            gen_dir = 'class_conditional_results'

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


    elif mode == 'classify-sentiment' or mode == 'classify-topic':

        Token_FILE = MODEL_FILE
        sub_model_name = os.path.basename(MODEL_FILE)

        gen_dir = 'classify_results'

        if 'finetune' in sub_model_name:
            tuning_mode = 'finetune'
            app = ''
        elif 'prefixtune' in sub_model_name:
            tuning_mode = 'prefixtune'
            if "_y_20" in sub_model_name:
                app = '--optim_prefix {} --preseqlen {} '.format('yes', 20)
            else:
                app = '--optim_prefix {} --preseqlen {} '.format('no', 20)
            if "_emb" in sub_model_name:
                app += "--prefix_mode embedding "
            elif "_act" in sub_model_name:
                app += "--prefix_mode activation "
            if "_inf" in sub_model_name or 'infix' in sub_model_name:
                app += " --format_mode infix "
            elif "_cat" in sub_model_name:
                app += " --format_mode cat "
            elif "_pee" in sub_model_name:
                app += " --format_mode peek "

            MODEL_FILE2 = MODEL_FILE
            MODEL_FILE = 'gpt2-medium'




    elif mode == 'triples':
        Token_FILE = MODEL_FILE

        gen_dir = 'triples_results'
        sub_model_name = os.path.basename(MODEL_FILE)

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

            if 'large' in sub_model_name:
                MODEL_FILE = 'gpt2-large'
            else:
                MODEL_FILE = 'gpt2-medium'

        elif 'adaptertune' in sub_model_name:
            tuning_mode = 'adaptertune'
            app = ''

    elif mode == 'webnlg':
        Token_FILE = MODEL_FILE
        gen_dir = 'webNLG_results'

        sub_model_name = os.path.basename(MODEL_FILE)

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

            if 'large' in sub_model_name:
                MODEL_FILE = 'gpt2-large'
            else:
                MODEL_FILE = 'gpt2-medium'

            # MODEL_FILE = 'gpt2-medium'
        elif 'adaptertune' in sub_model_name:
            tuning_mode = 'adaptertune'
            app = ''


    elif mode == 'cnndm' or mode == 'xsum':
        Token_FILE = MODEL_FILE
        gen_dir = 'xsum_results2'

        sub_model_name = os.path.basename(MODEL_FILE)

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

            if 'large' in sub_model_name:
                MODEL_FILE = 'gpt2-large'
            else:
                MODEL_FILE = 'gpt2-medium'

            # MODEL_FILE = 'gpt2-medium'
        elif 'adaptertune' in sub_model_name:
            tuning_mode = 'adaptertune'
            app = ''



    elif mode =='lemma2text':
        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/temp_data2text"
        # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/temp_data2text"

        MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/lemma2textfinetune"
        Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/lemma2textfinetune"
        # Token_FILE = "gpt2-medium"

        # preseqlen = 20
        # MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune{}".format(preseqlen)
        # Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/control/data2textprefixtune{}".format(preseqlen)
        if 'finetune' in MODEL_FILE:
            tuning_mode = 'finetune'
            app = ''
        elif 'prefixtune' in MODEL_FILE:
            tuning_mode = 'prefixtune'
            MODEL_FILE2 = MODEL_FILE
            MODEL_FILE = 'gpt2-medium'

            app = '--optim_prefix {} --preseqlen {}'.format('yes', preseqlen)

    elif mode =='text2data' and not use_prefixtuning:
        MODEL_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/temp_text2data"
        Token_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/language-modeling/temp_text2data"

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


    elif mode == 'dataless' and use_prefixtuning:
        MODEL_FILE = "gpt2-medium"
        # MODEL_FILE2 = "/u/scr/xlisali/contrast_LM/transformers/examples/control/len_trial-nb/checkpoint-12000"
        # MODEL_FILE2 = "/u/scr/xlisali/contrast_LM/transformers/examples/control/len_trial-b/checkpoint-10000"
        # MODEL_FILE2 = "/u/scr/xlisali/contrast_LM/transformers/examples/control/embMatchhaha"
        MODEL_FILE2 = "/u/scr/xlisali/contrast_LM/transformers/examples/control/dataless2/checkpoint-5000"
        Token_FILE = "gpt2-medium"

        if 'finetune' in MODEL_FILE2:
            tuning_mode = 'finetune'
            app = ''
        elif 'prefixtune' in MODEL_FILE2 or 'dataless' in MODEL_FILE2:
            tuning_mode = 'prefixtune'

            app = '--optim_prefix {} --preseqlen {}'.format('no', 0)

    elif mode == 'embMatch' and use_prefixtuning:
        MODEL_FILE = "gpt2-medium"
        # MODEL_FILE2 = "/u/scr/xlisali/contrast_LM/transformers/examples/control/len_trial-nb/checkpoint-12000"
        # MODEL_FILE2 = "/u/scr/xlisali/contrast_LM/transformers/examples/control/len_trial-b/checkpoint-10000"
        MODEL_FILE2 = "/u/scr/xlisali/contrast_LM/transformers/examples/control/embMatchhaha"
        # MODEL_FILE2 = "/u/scr/xlisali/contrast_LM/transformers/examples/control/keywordhaha"
        Token_FILE = "gpt2-medium"

    elif mode == 'keyword' and use_prefixtuning:
        MODEL_FILE = "gpt2-medium"
        # MODEL_FILE2 = "/u/scr/xlisali/contrast_LM/transformers/examples/control/len_trial-nb/checkpoint-12000"
        # MODEL_FILE2 = "/u/scr/xlisali/contrast_LM/transformers/examples/control/len_trial-b/checkpoint-10000"
        # MODEL_FILE2 = "/u/scr/xlisali/contrast_LM/transformers/examples/control/embMatchhaha"
        MODEL_FILE2 = "/u/scr/xlisali/contrast_LM/transformers/examples/control/keywordhaha"
        Token_FILE = "gpt2-medium"





    COMMANDLINE = "python /juice/scr/xlisali/contrast_LM/transformers/examples/text-generation/run_generation_batch.py \
        --model_type=gpt2 \
        --length 100 \
        --model_name_or_path={} \
        --num_return_sequences 5 \
        --stop_token [EOS] \
        --tokenizer_name={} \
        --task_mode={} \
        --control_mode={} --tuning_mode {} --gen_dir {} \
    ".format(MODEL_FILE, Token_FILE, mode, control_mode, tuning_mode, gen_dir)

    COMMANDLINE += app

    if tuning_mode == 'prefixtune' or tuning_mode == 'bothtune':
        COMMANDLINE += ' --prefixModel_name_or_path {}'.format(MODEL_FILE2)
        name = os.path.basename(MODEL_FILE2)
    else:
        name = os.path.basename(MODEL_FILE)


    if MODEL_FILE == 'gpt2-large':
        COMMANDLINE += ' --cache_dir /u/scr/xlisali/contrast_LM/transformers/examples/control/gpt2-large-s3 '



    if not submit_job:
        os.system(COMMANDLINE)
    else:
        # name = 'e2e_results_lowdata/{}'.format(name)
        # name = 'e2e_results_lowdata_finetune/{}'.format(name)
        name = os.path.join(gen_dir, name)
        full_command = "nlprun -a lisa-base-torch -g 1 -n {} -x jagupard4,jagupard5,jagupard6,jagupard7,jagupard8 \'{}\'".format(name,COMMANDLINE)
        print(full_command)
        os.system(full_command)


    # name = 'e2e_results_new/{}'.format(name) # prev large search result generation files.
    # name = 'e2e_results_infix/{}'.format(name)

    # name = 'webNLG_results/{}'.format(name)


    #########################
    # name = 'e2e_results_lowdata/{}'.format(name)
    # full_command = "nlprun -a lisa-base-torch -g 1 -n {} -x jagupard4,jagupard5,jagupard6,jagupard7,jagupard8 \'{}\'".format(name,COMMANDLINE)
    # print(full_command)
    # os.system(full_command)


#
#nlprun -a lisa-base-torch -g 1 --p high -n 5top-finetune -x jagupard4,jagupard5,jagupard6,jagupard7,jagupard8,jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25 '$COMMANDLINE'

#nlprun -a lisa-base-torch -g 1 -n topk-medium-finetune -m jagupard26 '$COMMANDLINE'





    # full_command = "nlprun -p high -a lisa-base-torch -g 1 -n {} -x jagupard4,jagupard5,jagupard6,jagupard7,jagupard8," \
    #                "jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18," \
    #                "jagupard19,jagupard20,jagupard21,jagupard22,jagupard23," \
    #                "jagupard24,jagupard25 \'{}\'".format(Model_FILE, COMMANDLINE)



