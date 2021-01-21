from collections import defaultdict
import numpy as np
import os

result_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
# path1 = '../results_lowdata_tune.txt'
# path1 = '../results_lowdata_finetune100.txt'
# path1 = '../temp_tune100.txt'
# path1 = '/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/e2e_results_conv/prefixtune_lowdata_100_results'
# path1 = '/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/e2e_results_conv/finetune_lowdata_100_results'
# path1 = '/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/e2e_results_conv/temp_10_prefix'
# path1 = '/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/e2e_results_conv/temp_10_finetune'
# path1 = '/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/e2e_results_conv/finetune_50_results'
# path1 = '/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/e2e_results_conv/prefixtune_50_results'
# path1 = '/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/e2e_results_conv/prefixtune_200_results'
# path1 = '/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/e2e_results_conv/finetune_200_results'
# path1 = '/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/e2e_results_conv/finetune_500_results'
# path1 = '/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/e2e_results_conv/prefixtune_500_results'
# path1 = '/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/no_init_temp1'
path1 = '/u/scr/xlisali/contrast_LM/transformers/examples/text-generation/init_temp1'
with open(path1, 'r') as f:
    for line in f:
        if line[:2] == 'Fi':
            continue

        name, bleu, nist, meteor, rougel, cider = line.split()

        if 'finetune' not in name and 'prefixtune' not in name:
            continue

        base_name = os.path.basename(name)

        base_name = base_name.split('_')
        print(base_name)

        train_label = base_name[14] # seeded dataset
        model_name = base_name[-4].split('-')[0] #table or summary
        steps = base_name[15] # max_steps
        seed = base_name[10] # seeded training.
        lr = base_name[8]
        ws = base_name[17]
        y = base_name[2]
        print(y, train_label, model_name, steps, seed, lr, ws)
        train_num = 100

        steps = (y, steps, lr, ws)

        result_dict[model_name][train_num][steps].append(
            (float(bleu), float(nist), float(meteor), float(rougel), float(cider)))

        #
        # base_name = base_name.split('_act_cat_b=10-e=5_d=0.0_u=no_lr=5e-05_w=0.0_s=101_r=n_m=512_lowdata_')
        # assert len(base_name) == 2
        # print(base_name[0], base_name[1])
        # model_name = base_name[0]
        # param_lst = base_name[1].split('_')
        # train_seed = int(param_lst[0])
        # train_num, steps = param_lst[1].split('st=')
        # train_num = int(train_num)
        # steps = int(steps)

        # result_dict[model_name][train_num][steps].append((float(bleu), float(nist), float(meteor), float(rougel), float(cider)))

print(result_dict)
# print(result_dict['data2textfinetune_n_20'][100])

metrics_name = 'BLEU NIST METEOR ROUGE_L CIDEr'.split(' ')
for model_name, model_result in result_dict.items():
    print(model_name)
    for key,val in model_result[100].items():
        for i, metrics in enumerate(list(zip(*val))):
            metrics = np.array(metrics)
            # assert len(metrics) == 3
            print(key, metrics_name[i], round(metrics.mean(), 4),  round(metrics.std(),4), len(metrics))

        print()
    print('-'*20)


        # if 'finetune' in name:
        #     # print(name, train_num, train_seed, 'finetune')
        #     result_dict['finetune'][train_num].append((float(bleu), float(nist), float(meteor), float(rougel), float(cider)))
        # elif 'prefixtune' in name:
        #     if train_num == 1000:
        #         print(name, train_num, train_seed, 'prefixtune')
        #     result_dict['prefixtune'][train_num].append((float(bleu), float(nist), float(meteor), float(rougel), float(cider)))

#
# for k, v in result_dict['prefixtune'].items():
#     print(k, 'mean', 'prefix')
#     v = result_dict['prefixtune'][k]
#     bleu, nist, meteor, rougel, cider = zip(*v)
#     print(np.array(bleu).mean())
#     print(np.array(nist).mean())
#     print(np.array(meteor).mean())
#     print(np.array(rougel).mean())
#     print(np.array(cider).mean())
#     print()
#
#     if k in result_dict['finetune']:
#         print(k, 'mean', 'finetune')
#         v = result_dict['finetune'][k]
#         bleu, nist, meteor, rougel, cider = zip(*v)
#         print(np.array(bleu).mean())
#         print(np.array(nist).mean())
#         print(np.array(meteor).mean())
#         print(np.array(rougel).mean())
#         print(np.array(cider).mean())
#         print()
#
#     print(k, 'max', 'prefix')
#     v = result_dict['prefixtune'][k]
#     bleu, nist, meteor, rougel, cider = zip(*v)
#     print(np.array(bleu).max())
#     print(np.array(nist).max())
#     print(np.array(meteor).max())
#     print(np.array(rougel).max())
#     print(np.array(cider).max())
#     print()
#
#     if k in result_dict['finetune']:
#         print(k, 'max', 'finetune')
#         v = result_dict['finetune'][k]
#         bleu, nist, meteor, rougel, cider = zip(*v)
#         print(np.array(bleu).max())
#         print(np.array(nist).max())
#         print(np.array(meteor).max())
#         print(np.array(rougel).max())
#         print(np.array(cider).max())
#         print()
#
#     print(k, 'min', 'prefix')
#     v = result_dict['prefixtune'][k]
#     bleu, nist, meteor, rougel, cider = zip(*v)
#     print(np.array(bleu).min())
#     print(np.array(nist).min())
#     print(np.array(meteor).min())
#     print(np.array(rougel).min())
#     print(np.array(cider).min())
#     print()
#
#     if k in result_dict['finetune']:
#         print(k, 'min', 'finetune')
#         v = result_dict['finetune'][k]
#         bleu, nist, meteor, rougel, cider = zip(*v)
#         print(np.array(bleu).min())
#         print(np.array(nist).min())
#         print(np.array(meteor).min())
#         print(np.array(rougel).min())
#         print(np.array(cider).min())
#         print()


# for k, v in result_dict['finetune'].items():
#     print(k)
#     bleu, nist, meteor, rougel, cider = zip(*v)
#     print(np.array(bleu).mean())
#     print(np.array(nist).mean())
#     print(np.array(meteor).mean())
#     print(np.array(rougel).mean())
#     print(np.array(cider).mean())
#     print()

# let's rank by BLEU first.























# from collections import defaultdict
# import numpy as np
#
# result_dict = {'finetune':defaultdict(list), 'prefixtune':defaultdict(list)}
# with open('results_lowdata3.txt', 'r') as f:
#     for line in f:
#         if line[:2] == 'Fi':
#             continue
#
#         name, bleu, nist, meteor, rougel, cider = line.split()
#
#         if 'finetune' not in name and 'prefixtune' not in name:
#             continue
#
#         if 'e=5' in name:
#             print('out of this time')
#             continue
#
#
#         temp = name.split('_')
#         train_num = int(temp[-3])
#         train_seed = int(temp[-4])
#         if 'finetune' in name:
#             # print(name, train_num, train_seed, 'finetune')
#             result_dict['finetune'][train_num].append((float(bleu), float(nist), float(meteor), float(rougel), float(cider)))
#         elif 'prefixtune' in name:
#             if train_num == 1000:
#                 print(name, train_num, train_seed, 'prefixtune')
#             result_dict['prefixtune'][train_num].append((float(bleu), float(nist), float(meteor), float(rougel), float(cider)))
#
# print(result_dict['prefixtune'])
#
# for k, v in result_dict['prefixtune'].items():
#     print(k, 'mean', 'prefix')
#     v = result_dict['prefixtune'][k]
#     bleu, nist, meteor, rougel, cider = zip(*v)
#     print(np.array(bleu).mean())
#     print(np.array(nist).mean())
#     print(np.array(meteor).mean())
#     print(np.array(rougel).mean())
#     print(np.array(cider).mean())
#     print()
#
#     if k in result_dict['finetune']:
#         print(k, 'mean', 'finetune')
#         v = result_dict['finetune'][k]
#         bleu, nist, meteor, rougel, cider = zip(*v)
#         print(np.array(bleu).mean())
#         print(np.array(nist).mean())
#         print(np.array(meteor).mean())
#         print(np.array(rougel).mean())
#         print(np.array(cider).mean())
#         print()
#
#     print(k, 'max', 'prefix')
#     v = result_dict['prefixtune'][k]
#     bleu, nist, meteor, rougel, cider = zip(*v)
#     print(np.array(bleu).max())
#     print(np.array(nist).max())
#     print(np.array(meteor).max())
#     print(np.array(rougel).max())
#     print(np.array(cider).max())
#     print()
#
#     if k in result_dict['finetune']:
#         print(k, 'max', 'finetune')
#         v = result_dict['finetune'][k]
#         bleu, nist, meteor, rougel, cider = zip(*v)
#         print(np.array(bleu).max())
#         print(np.array(nist).max())
#         print(np.array(meteor).max())
#         print(np.array(rougel).max())
#         print(np.array(cider).max())
#         print()
#
#     print(k, 'min', 'prefix')
#     v = result_dict['prefixtune'][k]
#     bleu, nist, meteor, rougel, cider = zip(*v)
#     print(np.array(bleu).min())
#     print(np.array(nist).min())
#     print(np.array(meteor).min())
#     print(np.array(rougel).min())
#     print(np.array(cider).min())
#     print()
#
#     if k in result_dict['finetune']:
#         print(k, 'min', 'finetune')
#         v = result_dict['finetune'][k]
#         bleu, nist, meteor, rougel, cider = zip(*v)
#         print(np.array(bleu).min())
#         print(np.array(nist).min())
#         print(np.array(meteor).min())
#         print(np.array(rougel).min())
#         print(np.array(cider).min())
#         print()
#
#
# # for k, v in result_dict['finetune'].items():
# #     print(k)
# #     bleu, nist, meteor, rougel, cider = zip(*v)
# #     print(np.array(bleu).mean())
# #     print(np.array(nist).mean())
# #     print(np.array(meteor).mean())
# #     print(np.array(rougel).mean())
# #     print(np.array(cider).mean())
# #     print()
#
# # let's rank by BLEU first.
#
#
#
#
#
