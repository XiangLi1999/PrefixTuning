import os, sys

# example: python train_run.py keyword temp_keyword _
if __name__ == '__main__':
    mode = sys.argv[1]
    control_mode = sys.argv[2]
    eval_split = sys.argv[3]
    model_file = None
    old_model = None
    MODEL_FILE = sys.argv[4]
    submit_job = (sys.argv[5] == 'yes')


    if mode =='data2text':

        Token_FILE = MODEL_FILE

        # gen_dir = 'e2e_results_conv'
        gen_dir = 'e2e_results_conv2'

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
        # gen_dir = 'webNLG_results'
        gen_dir = 'webNLG_results2'

        sub_model_name = os.path.basename(MODEL_FILE)

        if 'o=' in sub_model_name:
            o_idx = sub_model_name.index('o=')
            num_idx = sub_model_name[o_idx+2]
            print(num_idx)



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



    COMMANDLINE = "python run_generation.py \
        --model_type=gpt2 \
        --length 100 \
        --model_name_or_path={} \
        --num_return_sequences 5 \
        --stop_token [EOS] \
        --tokenizer_name={} \
        --task_mode={} \
        --control_mode={} --tuning_mode {} --gen_dir {} --eval_dataset {} \
    ".format(MODEL_FILE, Token_FILE, mode, control_mode, tuning_mode, gen_dir, eval_split)

    COMMANDLINE += app

    if tuning_mode == 'prefixtune' or tuning_mode == 'bothtune':
        COMMANDLINE += ' --prefixModel_name_or_path {}'.format(MODEL_FILE2)
        name = os.path.basename(MODEL_FILE2)
    else:
        name = os.path.basename(MODEL_FILE)


    if MODEL_FILE == 'gpt2-large':
        COMMANDLINE += ' --cache_dir /u/scr/xlisali/contrast_LM/transformers/examples/control/gpt2-large-s3 '

    if MODEL_FILE == 'gpt2-medium':
        COMMANDLINE += ' --cache_dir /u/scr/xlisali/contrast_LM/transformers/examples/control/gpt2-medium-s3 '


    print(COMMANDLINE)

    if not submit_job:
        os.system(COMMANDLINE)
    else:
        # name = 'e2e_results_lowdata/{}'.format(name)
        # name = 'e2e_results_lowdata_finetune/{}'.format(name)
        name = os.path.join(gen_dir, name)
        full_command = "nlprun -a lisa-base-torch -g 1 -n {} -x jagupard4,jagupard5,jagupard6,jagupard7,jagupard8,jagupard28,jagupard29 \'{}\'".format(name,COMMANDLINE)
        print(full_command)
        os.system(full_command)

