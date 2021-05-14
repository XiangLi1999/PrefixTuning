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
    parser.add_argument('--preseqlen', type=int, default=5, help='')
    parser.add_argument('--prefix_mode', type=str, default='activation', help='')
    parser.add_argument('--format_mode', type=str, default='cat', help='')

    parser.add_argument('--dir_name', type=str, default=None, help='')
    parser.add_argument('--notes', type=str, default=None, help='')
    parser.add_argument('--lowdata_token', type=str, default='summarize', help='')
    parser.add_argument('--use_lowdata_token', type=str, default='yes', help='')


    parser.add_argument('--parametrize_emb', type=str, default='MLP', help='')
    parser.add_argument('--adapter_design', type=int, default=1, help='')
    parser.add_argument('--top_layers', type=int, default=1, help='')

    parser.add_argument('--do_train', type=str, default='yes', help='')

    parser.add_argument('--fp16', type=str, default='no', help='')


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


    parser.add_argument('--label_smoothing', type=float, default=0.0, help='')
    parser.add_argument('--length_pen', type=float, default=1.0, help='')
    parser.add_argument('--mid_dim', type=int, default=512, help='')
    parser.add_argument('--use_deep', type=str, default='no', help='')

    parser.add_argument('--prefix_model_path', type=str, default=None, help='')
    parser.add_argument('--finetune_model_path', type=str, default=None, help='')
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

    assert  args.mode in ['e2e', 'cnn_dm', 'webnlg', 'triples', 'xsum', 'xsum_news', 'xsum_news_sport']



    if args.mode == 'e2e':

        data_dir= 'e2e'
        folder_name = 'save_e2e_models/'


    elif args.mode == 'triples':
        TRAIN_FILE = "/u/scr/xlisali/DART/dart/data/v1.1.1/dart-v1.1.1-full-train.json"
        TEST_FILE = "/u/scr/xlisali/DART/dart/data/v1.1.1/dart-v1.1.1-full-dev.json"
        folder_name = "triples_models/"


    elif args.mode == 'webnlg':
        # 2017 Challeng Version.
        TRAIN_FILE = "/u/scr/xlisali/WebNLG/webnlg-dataset/webnlg_challenge_2017/train.json"
        TEST_FILE = "/u/scr/xlisali/WebNLG/webnlg-dataset/webnlg_challenge_2017/dev.json"
        folder_name = "webnlg_models/"

    elif args.mode == 'writingPrompts':
        TRAIN_FILE = "/juice/u/xlisali/WritingPrompts/writingPrompts/train_small.txt"
        TEST_FILE = "/juice/u/xlisali/WritingPrompts/writingPrompts/valid_small.txt"
        folder_name = "wp_models/"

    elif args.mode == 'cnn_dm':
        data_dir = 'cnn_dm'
        folder_name = "cnndm_models/"
        max_source_length = 512
        max_target_length = 56
        val_max_target_length = 142
        test_max_target_length = 142

        cnndm_app = ' --max_source_length {} --max_target_length {} --val_max_target_length {} ' \
                    '--test_max_target_length {} '.format(max_source_length, max_target_length,
                                                         val_max_target_length, test_max_target_length)

        if args.fp16 == 'yes':
            cnndm_app += ' --fp16 --fp16_opt_level O1 '

        assert args.optim_prefix == 'yes'


    elif args.mode == 'xsum':
        data_dir = 'xsum'
        folder_name = "xsum_models/"
        max_source_length = 1024
        max_target_length = 60
        val_max_target_length = 60
        test_max_target_length = 100

        xsum_app = ' --max_source_length {} --max_target_length {} --val_max_target_length {} ' \
                    '--test_max_target_length {} '.format(max_source_length, max_target_length,
                                                         val_max_target_length, test_max_target_length)

        if args.fp16 == 'yes':
            xsum_app += ' --fp16 --fp16_opt_level O1 '

        assert args.optim_prefix == 'yes'

    elif args.mode == 'xsum_news':
        data_dir = '/data/xsum_news'
        folder_name = "/data/xsum_news_models/"
        max_source_length = 512
        max_target_length = 60
        val_max_target_length = 60
        test_max_target_length = 100

        xsum_app = ' --max_source_length {} --max_target_length {} --val_max_target_length {} ' \
                    '--test_max_target_length {} '.format(max_source_length, max_target_length,
                                                         val_max_target_length, test_max_target_length)

        if args.fp16 == 'yes':
            xsum_app += ' --fp16 --fp16_opt_level O1 '

    elif args.mode == 'xsum_news_sport':
        data_dir = '/data/xsum_topic-news-sports'
        folder_name = "/data/xsum_news_sport_models/"
        max_source_length = 512
        max_target_length = 60
        val_max_target_length = 60
        test_max_target_length = 100
    
        xsum_app = ' --max_source_length {} --max_target_length {} --val_max_target_length {} ' \
                    '--test_max_target_length {} '.format(max_source_length, max_target_length,
                                                         val_max_target_length, test_max_target_length)

        if args.fp16 == 'yes':
            xsum_app += ' --fp16 --fp16_opt_level O1 '

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


    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)




    batch_size = args.gradient_accumulation_steps * args.bsz
    # print(args.mode + args.tuning_mode + '_' + args.optim_prefix[:1] + '_' + args.preseqlen)
    # print('_' + args.prefix_mode[:2] + '_' + args.format_mode[:2] + '_')
    if args.dir_name is None:
        Model_FILE = args.mode + args.tuning_mode + '_' + args.optim_prefix[:1] + '_' + str(args.preseqlen) + \
                     '_' + args.prefix_mode[:3] + '_' + args.format_mode[:3] + '_' + \
                     'b={}-'.format(batch_size) + 'e={}_'.format(args.epoch) + 'd={}_'.format(args.dropout) + \
                     'l={}_'.format(args.label_smoothing) + 'lr={}_'.format(args.learning_rate) \
                     + 'w={}_'.format(args.weight_decay) + 's={}'.format(args.seed) + '_d={}'.format(args.use_deep[:1]) +\
                     '_m={}'.format(args.mid_dim)
    else:
        Model_FILE = dir_name

    if args.notes is not None:
        Model_FILE += '_{}'.format(args.notes)

    # Model_FILE = 'save_e2e_models/{}'.format(Model_FILE)

    logging_dir = os.path.join(folder_name, 'runs', Model_FILE)
    Model_FILE = '{}{}'.format(folder_name, Model_FILE)
    print(Model_FILE)


    OLD_MODEL = 'facebook/bart-large'

    app = "--optim_prefix {} --preseqlen {} --prefix_mode {} --format_mode {} " \
          "--gradient_accumulation_steps {} --learning_rate {} --weight_decay {} --seed {} " \
          "--mid_dim {} --use_dropout {} --prefix_dropout {} ".\
        format(args.optim_prefix, args.preseqlen, args.prefix_mode, args.format_mode,
               args.gradient_accumulation_steps, args.learning_rate, args.weight_decay, args.seed,
               args.mid_dim, args.use_dropout, args.dropout)

    if args.prefix_mode == 'embedding':
        app += ' --parametrize_emb {} '.format(args.parametrize_emb)

    if args.tuning_mode == 'adaptertune':
        app += ' --adapter_design {} '.format(args.adapter_design)

    if args.mode == 'cnn_dm':
        app += cnndm_app

    if args.mode == 'xsum' or args.mode == 'xsum_news' or args.mode == 'xsum_news_sport': 
        app += xsum_app


    if OLD_MODEL == 'gpt2-large':
        app += ' --cache_dir /u/scr/xlisali/contrast_LM/transformers/examples/control/gpt2-large-s3 '

    if args.tuning_mode == 'finetune-top':
        app += ' --top_layers {} '.format(args.top_layers)




    controlprefix = ('yes' if args.tuning_mode == 'prefixtune' else 'no')


    if args.do_train == 'yes':
        COMMANDLINE = 'python finetune.py ' \
                      '--model_name_or_path {} ' \
                      '--output_dir {} ' \
                      '--data_dir {} ' \
                      '--tuning_mode {} ' \
                      '--preseqlen {} ' \
                      '--do_train ' \
                      '--label_smoothing {} ' \
                      '--use_deep {} ' \
                      '--gpus 1 ' \
                      '--learning_rate {} ' \
                      '--train_batch_size {} ' \
                      '--eval_batch_size {} ' \
                      '--num_train_epochs {} '.format(OLD_MODEL, Model_FILE, data_dir, args.tuning_mode, args.preseqlen, args.label_smoothing, args.use_deep,
                                                      args.learning_rate, args.bsz, args.bsz, args.epoch)
    else:
        if args.tuning_mode == 'finetune':
            assert args.finetune_model_path is not None
            print('loading from the finetune model {}'.format(args.finetune_model_path))
            Model_FILE = args.finetune_model_path + '_decode_eval' + '_{}'.format(args.length_pen)
            print('writing the decoded results to {}'.format(Model_FILE))
            COMMANDLINE = 'python finetune.py ' \
                          '--model_name_or_path {} ' \
                          '--output_dir {} ' \
                          '--data_dir {} ' \
                          '--tuning_mode {} ' \
                          '--preseqlen {} ' \
                          '--do_predict ' \
                          '--use_deep {} ' \
                          '--gpus 1 ' \
                          '--train_batch_size {} ' \
                          '--eval_batch_size {} ' \
                          '--length_penalty {} ' \
                          '--num_train_epochs {} '.format(args.finetune_model_path, Model_FILE, data_dir,
                                                          args.tuning_mode, args.preseqlen,  args.use_deep,
                                                          10, 10, args.length_pen, args.epoch)
        else:
            assert args.prefix_model_path is not None
            print('loading from the prefix model {}'.format(args.prefix_model_path))
            print('loading from the main model {}'.format(OLD_MODEL))
            Model_FILE = args.prefix_model_path + '_decode_eval' + '_{}'.format(args.length_pen)
            print('writing the decoded results to {}'.format(Model_FILE))
            COMMANDLINE = 'python finetune.py ' \
                          '--model_name_or_path {} ' \
                          '--prefixModel_name_or_path {} ' \
                          '--output_dir {} ' \
                          '--data_dir {} ' \
                          '--tuning_mode {} ' \
                          '--preseqlen {} ' \
                          '--do_predict ' \
                          '--use_deep {} ' \
                          '--gpus 1 ' \
                          '--train_batch_size {} ' \
                          '--eval_batch_size {} ' \
                          '--seed {} ' \
                          '--length_penalty {} ' \
                          '--num_train_epochs {} '.format(OLD_MODEL, args.prefix_model_path, Model_FILE, data_dir,
                                                          args.tuning_mode, args.preseqlen, args.use_deep,
                                                          8, 8, args.seed, args.length_pen, args.epoch)


    # COMMANDLINE="python run_language_modeling.py \
    #     --output_dir={} \
    #     --model_type=gpt2 \
    #     --model_name_or_path={} \
    #     --tokenizer_name={} \
    #     --per_device_train_batch_size {} \
    #     --per_device_eval_batch_size {} \
    #     --save_steps 500000 \
    #     --num_train_epochs {} \
    #     --do_train \
    #     --train_data_file={} \
    #     --do_eval \
    #     --line_by_line \
    #     --save_total_limit 1 \
    #     --overwrite_output_dir \
    #     --task_mode {} \
    #     --eval_data_file={}  \
    #     --dataless no --tuning_mode {} --logging_dir {} \
    #     --train_embs no ".format(Model_FILE, OLD_MODEL, OLD_MODEL, args.bsz, args.bsz, args.epoch, TRAIN_FILE, args.mode, TEST_FILE,
    #                              args.tuning_mode, logging_dir)

    COMMANDLINE += app



    with open(Model_FILE + '.sh', 'w') as f:
        print(COMMANDLINE, file=f)


    print(COMMANDLINE)
    if args.submit == 'no':
        os.system(COMMANDLINE) # textattack/roberta-base-ag-news # textattack/roberta-base-imdb
    # #
    elif args.submit == 'yes':
        if args.use_big == 'no':
            full_command = "nlprun -a lisa_apex_latest -g 1 -n {} -x jagupard4,jagupard5,jagupard6,jagupard7,jagupard8 \'{}\'".format(Model_FILE, COMMANDLINE)
        elif args.use_big == 'yes':
            full_command = "nlprun -p high -a lisa_apex_latest -g 1 -n {} -x jagupard4,jagupard5,jagupard6,jagupard7,jagupard8," \
                           "jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18," \
                           "jagupard19,jagupard20,jagupard21,jagupard22,jagupard23," \
                           "jagupard24,jagupard25,jagupard28,jagupard29 \'{}\'".format(Model_FILE, COMMANDLINE)

        elif args.use_big == 'yes2':
            full_command = "nlprun -p high -a lisa_apex_latest2 -g 1 -n {} -x jagupard4,jagupard5,jagupard6,jagupard7,jagupard8," \
                           "jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18," \
                           "jagupard19,jagupard20,jagupard21,jagupard22,jagupard23," \
                           "jagupard24,jagupard25,jagupard26,jagupard27 \'{}\'".format(Model_FILE, COMMANDLINE)
        else:
            full_command = "nlprun -a lisa_apex_latest -m jagupard26 -p high -g 1 -n {} \'{}\'".format(Model_FILE, COMMANDLINE)
        print(full_command)
        os.system(full_command)


