import os, sys
import argparse

# example: python train_run.py keyword temp_keyword _
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='data2text E2E training args.')
    parser.add_argument('--mode', type=str, default='data2text', help='')
    parser.add_argument('--tuning_mode', type=str, default='prefixtune', help='')
    parser.add_argument('--optim_prefix', type=str, default='yes', help='')
    parser.add_argument('--preseqlen', type=int, default=10, help='')
    parser.add_argument('--prefix_mode', type=str, default='activation', help='')
    parser.add_argument('--format_mode', type=str, default='cat', help='')

    parser.add_argument('--dir_name', type=str, default=None, help='')
    parser.add_argument('--notes', type=str, default=None, help='')
    parser.add_argument('--lowdata_token', type=str, default='summarize', help='')
    parser.add_argument('--use_lowdata_token', type=str, default='yes', help='')


    parser.add_argument('--parametrize_emb', type=str, default='MLP', help='')
    parser.add_argument('--adapter_design', type=int, default=1, help='')
    parser.add_argument('--top_layers', type=int, default=1, help='')

    parser.add_argument('--objective_mode', type=int, default=1, help='')


    # training parameters.
    parser.add_argument('--use_dropout', type=str, default='no', help='')
    parser.add_argument('--seed', type=int, default=101, help='') # old is 42
    parser.add_argument('--bsz', type=int, default=10, help='')
    parser.add_argument('--use_big', type=str, default='no', help='')
    parser.add_argument('--epoch', type=int, default=5, help='')
    parser.add_argument('--max_steps', type=int, default=400, help='')
    parser.add_argument('--eval_steps', type=int, default=50, help='')
    parser.add_argument('--warmup_steps', type=int, default=100, help='')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='')
    parser.add_argument('--learning_rate', type=float, default=5e-05, help='')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--mid_dim', type=int, default=512, help='')
    parser.add_argument('--init_random', type=str, default='no', help='')

    parser.add_argument('--prefix_model_path', type=str, default=None, help='')
    parser.add_argument('--submit', type=str, default='no', help='')

    args = parser.parse_args()

    assert args.optim_prefix in ['yes', 'no']
    if args.optim_prefix == 'yes':
        assert args.preseqlen is not None
    assert args.prefix_mode in ['embedding', 'activation']
    assert args.format_mode in ['cat', 'infix', 'peek', 'nopeek']
    assert args.tuning_mode in ['prefixtune', 'finetune', 'finetune-top', 'bothtune', 'adaptertune']
    if args.prefix_model_path is not None:
        load_prefix_model = True
    else:
        load_prefix_model = False

    assert  args.mode in ['data2text', 'triples', 'webnlg', 'writingPrompts', 'cnndm', 'xsum', 'sentiment', 'topic',
                          'classify-sentiment', 'classify-topic']

    assert args.objective_mode in [0, 1]
    # 0 means the regular token level objective, which is sum / output_len
    # 1 means the sentence level objective, which is sum
    # 2 means our buggy version which is sum/max_batch(input_len +output_len)
    # 3 means our buggy version which is sum/max_batch(output_len)
    # 4 means our buggy version which is sum/(input_len +output_len)


    if args.tuning_mode == 'adaptertune':
        folder_name = 'baseline_adapter/'
        if args.notes is None:
            args.notes = 'a={}_b={}'.format(args.adapter_design, args.adapter_bottleneck)
        else:
            args.notes = args.notes + '_a={}_b={}'.format(args.adapter_design, args.adapter_bottleneck)

    if args.tuning_mode == 'finetune-top':
        if args.notes is None:
            args.notes = 'l={}'.format(args.top_layers)
        else:
            args.notes = args.notes + '_l={}'.format(args.top_layers)


    if args.mode == 'data2text':

        TRAIN_FILE = "/u/scr/xlisali/e2e_data/src1_train.txt"
        TEST_FILE = "/u/scr/xlisali/e2e_data/src1_valid.txt"
        # folder_name = 'save_e2e_models_infix/'
        folder_name = 'save_e2e_models_convcheck/'

        if args.prefix_mode == 'embedding':
            folder_name = 'ablation_e2e_emb_models/'

            if args.notes is None:
                args.notes = args.parametrize_emb
            else:
                args.notes = args.notes + '_p={}'.format(args.parametrize_emb)

        if args.format_mode == 'infix':
            folder_name = 'ablation_e2e_infix_models/'



        if args.notes is not None and 'lowdata' in args.notes:
            _, temp_seed, temp_size = args.notes.split('_')
            TRAIN_FILE = "/juice/u/xlisali/e2e_lowdata/lowdata_{}_{}_train.txt".format(temp_seed, temp_size)
            TEST_FILE = "/juice/u/xlisali/e2e_lowdata/lowdata_{}_{}_valid.txt".format(temp_seed, temp_size)
            # folder_name = 'e2e_lowdata_models_new/' #100
            # folder_name = 'e2e_lowdata_models_finetune/'
            folder_name = 'e2e_lowdata_models_prefixtune/' # 50, 200

            if temp_size == '10':
                pass
                # args.max_steps = 100
                # args.eval_steps = 15

            if temp_size == '100':
                pass
                # args.max_steps = 300
                # args.max_steps = 150

                # args.max_steps = 400
                # args.eval_steps = 50


            app_special = ' --max_steps {} --eval_steps {} --save_steps -1 ' \
                          '--evaluate_during_training --per_device_eval_batch_size 32 ' \
                          '--warmup_steps {} --lowdata_token {} ' \
                          '--use_lowdata_token {} '.format(args.max_steps, args.eval_steps,
                                                           args.warmup_steps, args.lowdata_token,
                                                           args.use_lowdata_token)

            args.notes = args.notes + 'st={}_ev={}_ws={}_t={}'.format(args.max_steps, args.eval_steps,
                                                                      args.warmup_steps, args.lowdata_token)


        if args.notes in ['Type', 'near', 'customer', 'food', 'area', 'price']:
            TRAIN_FILE = "/u/scr/xlisali/e2e_data/missing_{}_cleaned_train_e2e.txt".format(args.notes[:4])
            TEST_FILE = "/u/scr/xlisali/e2e_data/missing_{}_cleaned_dev_e2e.txt".format(args.notes[:4])
            folder_name = "compose_control_cleaned/"

        # if args.notes == 'Type':
        #     TRAIN_FILE = "/u/scr/xlisali/e2e_data/missing_Type_src1_train.txt"
        #     TEST_FILE = "/u/scr/xlisali/e2e_data/missing_Type_src1_valid.txt"
        #     folder_name = "compose_control/"
        # elif args.notes == 'near':
        #     TRAIN_FILE = "/u/scr/xlisali/e2e_data/missing_near_src1_train.txt"
        #     TEST_FILE = "/u/scr/xlisali/e2e_data/missing_near_src1_valid.txt"
        #     folder_name = "compose_control/"

        print(TRAIN_FILE)
        print(TEST_FILE)

    elif args.mode == 'triples':
        TRAIN_FILE = "/u/scr/xlisali/DART/dart/data/v1.1.1/dart-v1.1.1-full-train.json"
        TEST_FILE = "/u/scr/xlisali/DART/dart/data/v1.1.1/dart-v1.1.1-full-dev.json"
        folder_name = "triples_models/"


    elif args.mode == 'webnlg':
        # 2017 Challeng Version.
        TRAIN_FILE = "/u/scr/xlisali/WebNLG/webnlg-dataset/webnlg_challenge_2017/train.json"
        TEST_FILE = "/u/scr/xlisali/WebNLG/webnlg-dataset/webnlg_challenge_2017/dev.json"

        # v2
        # TRAIN_FILE = "/u/scr/xlisali/WebNLG/webnlg-dataset/release_v2/json/webnlg_release_v2_train.json"
        # TEST_FILE =  "/u/scr/xlisali/WebNLG/webnlg-dataset/release_v2/json/webnlg_release_v2_dev.json"
        folder_name = "webnlg_models/"

    elif args.mode == 'writingPrompts':
        TRAIN_FILE = "/juice/u/xlisali/WritingPrompts/writingPrompts/train_small.txt"
        TEST_FILE = "/juice/u/xlisali/WritingPrompts/writingPrompts/valid_small.txt"
        folder_name = "wp_models/"

    elif args.mode == 'cnndm':
        # TRAIN_FILE = "/juice/u/xlisali/WritingPrompts/summarization/finished_files/test.txt"
        # TEST_FILE = "/juice/u/xlisali/WritingPrompts/summarization/finished_files/val.txt"
        TRAIN_FILE = '/u/scr/xlisali/contrast_LM/transformers/examples/seq2seq/cnn_dm/train.source'
        TEST_FILE = '/u/scr/xlisali/contrast_LM/transformers/examples/seq2seq/cnn_dm/val.source'

        max_source_length = 512
        max_target_length = 56
        val_max_target_length = 142
        test_max_target_length = 142

        cnndm_app = ' --max_source_length {} --train_max_target_length {} ' \
                   '--val_max_target_length {} --dataloader_num_workers 4 '.format(max_source_length, max_target_length,
                                                         val_max_target_length, )

        folder_name = "cnndm_models/"
        assert args.optim_prefix == 'yes'

    elif args.mode == 'xsum':
        TRAIN_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/seq2seq/xsum/train.source"
        TEST_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/seq2seq/xsum/val.source"

        max_source_length = 512
        max_target_length = 60
        val_max_target_length = 60
        test_max_target_length = 100

        xsum_app = ' --max_source_length {} --train_max_target_length {} ' \
                   '--val_max_target_length {} --dataloader_num_workers 4 '.format(max_source_length, max_target_length,
                                                         val_max_target_length, )

        folder_name = "xsum_models/"
        assert args.optim_prefix == 'yes'

    elif args.mode == 'sentiment':
        TRAIN_FILE = "/u/scr/xlisali/IMDB/train.txt"
        TEST_FILE = "/u/scr/xlisali/IMDB/dev.txt"
        folder_name = "sentiment_models/"

    elif args.mode == 'topic':
        TRAIN_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/text-classification/glue_data/AG-news/train1.tsv"
        TEST_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/text-classification/glue_data/AG-news/dev1.tsv"
        folder_name = "topic_models/"

    elif args.mode == 'classify-sentiment':
        TRAIN_FILE = "/u/scr/xlisali/IMDB/train.txt"
        TEST_FILE = "/u/scr/xlisali/IMDB/dev.txt"
        folder_name = "classification_models/"
        assert args.optim_prefix == 'yes'

    elif args.mode == 'classify-topic':
        TRAIN_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/text-classification/glue_data/AG-news/train1.tsv"
        TEST_FILE = "/u/scr/xlisali/contrast_LM/transformers/examples/text-classification/glue_data/AG-news/dev1.tsv"
        folder_name = "classification_models/"
        assert args.optim_prefix == 'yes'




    batch_size = args.gradient_accumulation_steps * args.bsz
    # print(args.mode + args.tuning_mode + '_' + args.optim_prefix[:1] + '_' + args.preseqlen)
    # print('_' + args.prefix_mode[:2] + '_' + args.format_mode[:2] + '_')

    if args.notes is not None:
        args.notes = args.notes + '_o={}'.format(args.objective_mode)
    else:
        args.notes = 'o={}'.format(args.objective_mode)

    if args.dir_name is None:
        Model_FILE = args.mode + args.tuning_mode + '_' + args.optim_prefix[:1] + '_' + str(args.preseqlen) + \
                     '_' + args.prefix_mode[:3] + '_' + args.format_mode[:3] + '_' + \
                     'b={}-'.format(batch_size) + 'e={}_'.format(args.epoch) + 'd={}_'.format(args.dropout) + \
                     'u={}_'.format(args.use_dropout) + 'lr={}_'.format(args.learning_rate) \
                     + 'w={}_'.format(args.weight_decay) + 's={}'.format(args.seed) + '_r={}'.format(args.init_random[:1]) +\
                     '_m={}'.format(args.mid_dim)
    else:
        Model_FILE = dir_name

    if args.notes is not None:
        Model_FILE += '_{}'.format(args.notes)

    # Model_FILE = 'save_e2e_models/{}'.format(Model_FILE)

    logging_dir = os.path.join(folder_name, 'runs', Model_FILE)
    Model_FILE = '{}{}'.format(folder_name, Model_FILE)
    print(Model_FILE)


    if args.notes is not None and 'large' in args.notes:
        OLD_MODEL = "gpt2-large"
    else:
        OLD_MODEL = "gpt2-medium"

    app = "--optim_prefix {} --preseqlen {} --prefix_mode {} --format_mode {} " \
          "--gradient_accumulation_steps {} --learning_rate {} --weight_decay {} --seed {} --disable_tqdm " \
          "--mid_dim {} --init_random {} --use_dropout {} --prefix_dropout {} --objective_mode {} ".\
        format(args.optim_prefix, args.preseqlen, args.prefix_mode, args.format_mode,
               args.gradient_accumulation_steps, args.learning_rate, args.weight_decay, args.seed,
               args.mid_dim, args.init_random, args.use_dropout, args.dropout, args.objective_mode)

    if args.prefix_mode == 'embedding':
        app += ' --parametrize_emb {} '.format(args.parametrize_emb)

    if args.tuning_mode == 'adaptertune':
        app += ' --adapter_design {} '.format(args.adapter_design)

    # temp for logging of the evals.
    if args.notes is not None and 'lowdata' in args.notes:
        app += app_special
    else:
        app += '--evaluate_during_training --eval_steps 5000 '

    if OLD_MODEL == 'gpt2-large':
        app += ' --cache_dir /u/scr/xlisali/contrast_LM/transformers/examples/control/gpt2-large-s3 '

    if args.tuning_mode == 'finetune-top':
        app += ' --top_layers {} '.format(args.top_layers)


    if args.mode == 'xsum':
        app += xsum_app

    if args.mode == 'cnndm':
        app += cnndm_app



    # when we have input-based AND activation level AND infix.
    # app = "--optim_prefix {} --preseqlen {} --prefix_mode {} --format_mode {} " \
    #       "--gradient_accumulation_steps 1 ".format('no', 15, 'activation', 'infix')
    # Try to replicate the best performing version of cat.
    # app = "--optim_prefix {} --preseqlen {} --prefix_mode {} --format_mode {} " \
    #       "--gradient_accumulation_steps 1 ".format('no', 15, 'activation', 'cat')

    # app = "--optim_prefix {} --preseqlen {} --prefix_mode {} --format_mode {} " \
    #       "--gradient_accumulation_steps 1 ".format('peek', 15, 'activation', 'cat')

    # We have instruction based AND embedding AND peek.
    # app = "--optim_prefix {} --preseqlen {} --prefix_mode {} --format_mode {}".format('yes', 20, 'embedding', 'peek')

    # We have instruction based AND activation AND peek.
    # app = "--optim_prefix {} --preseqlen {} --prefix_mode {} --format_mode {}".format('yes', 15, 'activation', 'peek')



    controlprefix = ('yes' if args.tuning_mode == 'prefixtune' else 'no')

    COMMANDLINE="python run_language_modeling.py \
        --output_dir={} \
        --model_type=gpt2 \
        --model_name_or_path={} \
        --tokenizer_name={} \
        --per_device_train_batch_size {} \
        --per_device_eval_batch_size {} \
        --save_steps 500000 \
        --num_train_epochs {} \
        --do_train \
        --train_data_file={} \
        --do_eval \
        --line_by_line \
        --save_total_limit 1 \
        --overwrite_output_dir \
        --task_mode {} \
        --eval_data_file={}  \
        --dataless no --tuning_mode {} --logging_dir {} \
        --train_embs no ".format(Model_FILE, OLD_MODEL, OLD_MODEL, args.bsz, args.bsz, args.epoch, TRAIN_FILE, args.mode, TEST_FILE,
                                 args.tuning_mode, logging_dir)

    COMMANDLINE += app

    if load_prefix_model:
        LOAD_TRAIN_PREFIX = '/u/scr/xlisali/contrast_LM/transformers/examples/control/med_topic_gen'
        COMMANDLINE += '--prefixModel_name_or_path {} '.format(LOAD_TRAIN_PREFIX)



    with open(Model_FILE + '.sh', 'w') as f:
        print(COMMANDLINE, file=f)


    print(COMMANDLINE)
    if args.submit == 'no':
        os.system(COMMANDLINE) # textattack/roberta-base-ag-news # textattack/roberta-base-imdb
    # #
    elif args.submit == 'yes':
        if args.use_big == 'no':
            full_command = "nlprun -a lisa-base-torch -g 1 -n {} -x jagupard4,jagupard5,jagupard6,jagupard7,jagupard8,jagupard14 \'{}\'".format(Model_FILE, COMMANDLINE)
            if args.mode == 'cnndm':
                full_command ="nlprun -a lisa-base-torch -r 20GB -g 1 -n {} -x jagupard4,jagupard5,jagupard6,jagupard7,jagupard8 \'{}\'".format(Model_FILE, COMMANDLINE)
        elif True:
            full_command = "nlprun -p high -a lisa-base-torch -g 1 -n {} -x jagupard4,jagupard5,jagupard6,jagupard7,jagupard8," \
                           "jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18," \
                           "jagupard19,jagupard20,jagupard21,jagupard22,jagupard23," \
                           "jagupard24,jagupard25 \'{}\'".format(Model_FILE, COMMANDLINE)
        else:
            full_command = "nlprun -a lisa-base-torch -m jagupard26 -p high -g 1 -n {} \'{}\'".format(Model_FILE, COMMANDLINE)
        print(full_command)
        os.system(full_command)




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
