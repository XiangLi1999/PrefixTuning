import os, sys
import argparse
from pathlib import Path

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
    parser.add_argument('--adapter_bottleneck', type=int, default=100, help='')

    parser.add_argument('--top_layers', type=int, default=1, help='')

    parser.add_argument('--objective_mode', type=int, default=1, help='')

    parser.add_argument('--init_shallow', type=str, default='no', help='')
    parser.add_argument('--init_shallow_word', type=str, default='summarize', help='')



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


    # DISTILLATION
    parser.add_argument('--distill', type=str, default='no', help='')
    parser.add_argument('--finetuned_model_path', type=str,
                        default='/u/scr/xlisali/contrast_LM/transformers/examples/full/full/webnlgfinetune_n_20_act_ca'
                                't_b=6-e=10_d=0.0_u=no_lr=1e-05_w=0.0_s=101_r=n_m=512_earlystop', help='')
    parser.add_argument('--matching_objective', type=str, default='kl', help='kl or logits')

    # Added by MX
    parser.add_argument('--cache_dir', type=str, default='/u/scr/xlisali/contrast_LM/transformers/examples/control', help='cache dir')
    parser.add_argument('--use_custom_teacher_dropout', type=str, default='no', help='')



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

    assert args.objective_mode in [0, 1, 2, 3, 4]
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

    # added by MX
    if args.tuning_mode == 'finetune':
        key = f'uctd={args.use_custom_teacher_dropout}'
        if args.notes is None:
            args.notes = key
        else:
            args.notes = args.notes + f'_{key}'


    if args.mode == 'data2text':

        TRAIN_FILE = "../data/e2e_data/src1_train.txt"
        TEST_FILE = "../data/e2e_data/src1_valid.txt"
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
            folder_name = 'e2e_lowdata_models_prefixtune/' # 50, 200



            app_special = ' --max_steps {} --eval_steps {} --save_steps -1 ' \
                          '--evaluate_during_training --per_device_eval_batch_size 32 ' \
                          '--warmup_steps {} --lowdata_token {} ' \
                          '--use_lowdata_token {} '.format(args.max_steps, args.eval_steps,
                                                           args.warmup_steps, args.lowdata_token,
                                                           args.use_lowdata_token)

            args.notes = args.notes + 'st={}_ev={}_ws={}_t={}'.format(args.max_steps, args.eval_steps,
                                                                      args.warmup_steps, args.lowdata_token)


        if args.notes is not None and 'datalevels' in args.notes:
            # for example, notes = 'datalevels-1-10
            _, temp_seed, temp_size = args.notes.split('_')
            TRAIN_FILE = "/juice/u/xlisali/e2e_datalevels/datalevels_{}_{}_train.txt".format(temp_seed, temp_size)
            TEST_FILE = "/juice/u/xlisali/e2e_datalevels/datalevels_{}_{}_valid.txt".format(temp_seed, temp_size)
            # folder_name = 'e2e_lowdata_models_new/' #100
            # folder_name = 'e2e_lowdata_models_finetune/'
            folder_name = 'e2e_datalevels_models/' # 50, 200




            app_special_levels = ' --eval_steps {} --save_steps -1 ' \
                          '--evaluate_during_training --per_device_eval_batch_size 32 ' \
                          '--warmup_steps {} --lowdata_token {} ' \
                          '--use_lowdata_token {} '.format(args.eval_steps,
                                                           args.warmup_steps, args.lowdata_token,
                                                           args.use_lowdata_token)

            args.notes = args.notes + 'ev={}_ws={}_t={}_u={}'.format(args.eval_steps,
                                                                      args.warmup_steps, args.lowdata_token,
                                                                      args.use_lowdata_token)


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

    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)



    batch_size = args.gradient_accumulation_steps * args.bsz

    if args.init_shallow == 'yes':
        if args.notes is not None:
            args.notes = args.notes + '_s={}_w={}'.format(args.init_shallow, args.init_shallow_word)
        else:
            args.notes ='s={}_w={}'.format(args.init_shallow, args.init_shallow_word)

    if args.notes is not None:
        args.notes = args.notes + '_o={}'.format(args.objective_mode)
    else:
        args.notes = 'o={}'.format(args.objective_mode)


    if args.distill == 'yes':
        if args.notes is not None:
            args.notes = args.notes + f'_distill_o={args.matching_objective}'
        else:
            args.notes =f'distill_o={args.matching_objective}'

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
        app += ' --adapter_design {} --adapter_bottleneck {} '.format(args.adapter_design, args.adapter_bottleneck)

    # temp for logging of the evals.
    if args.notes is not None and 'lowdata' in args.notes:
        app += app_special
    elif args.notes is not None and 'datalevels' in args.notes:
        app += app_special_levels
    else:
        app += '--evaluate_during_training --eval_steps 5000 '

    if OLD_MODEL == 'gpt2-large':
        app += f' --cache_dir {Path(args.cache_dir) / "gpt2-large-s3"} '

    elif OLD_MODEL == 'gpt2-medium':
        app += f' --cache_dir {Path(args.cache_dir) / "gpt2-medium-s3"} '

    if args.tuning_mode == 'finetune-top':
        app += ' --top_layers {} '.format(args.top_layers)


    if args.mode == 'xsum':
        app += xsum_app

    if args.mode == 'cnndm':
        app += cnndm_app

    if args.init_shallow == 'yes':
        app += ' --init_shallow {} --init_shallow_word {} '.format(args.init_shallow, args.init_shallow_word)


    if args.distill == 'yes':
        app += f' --distill {args.distill} --finetuned_model_path {args.finetuned_model_path}  --matching_objective {args.matching_objective}'

    if args.use_custom_teacher_dropout == 'yes':
        app += f' --use_custom_teacher_dropout {args.use_custom_teacher_dropout}'
        app += f' --teacher_dropout {args.dropout}'

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
        --tuning_mode {} --logging_dir {} \
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
            full_command = "nlprun -a lisa-base-torch -g 1 -n {} -x jagupard4,jagupard5,jagupard6,jagupard7,jagupard8,jagupard28,jagupard29,jagupard11,jagupard12,jagupard10 \'{}\'".format(Model_FILE, COMMANDLINE)
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


