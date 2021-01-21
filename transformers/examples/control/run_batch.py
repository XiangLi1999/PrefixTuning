import os
from itertools import product
batch_lst = [8, 10, 13]
epoch_lst = [3, 5, 10]
dropout_lst = [0.0, 0.2, 0.05]
gpt2_drop_lst = ['yes', 'no']
random_init_lst = ['yes', 'no']
format_lst = ['peek', 'cat']
mid_dim_lst = [20, 100, 200, 512]
lr_lst = [0.001, 0.0005, 0.0001, 0.00005]

# os.system("python train_e2e.py --init_random yes --submit yes")
total_experiment_count = 0
#
# for (format, mid_dim) in product(format_lst, mid_dim_lst):
#     os.system("python train_e2e.py --format_mode {} --mid_dim {} --submit yes".format(format, mid_dim))
#     total_experiment_count+= 1


# check for epoch&bsz on peek/cat.
# for (bsz, epoch, format) in product(batch_lst, epoch_lst, format_lst):
#     os.system("python train_e2e.py --format_mode {} --bsz {} --epoch {} --submit yes".format(format, bsz, epoch))
#     total_experiment_count += 1
#
# # dropout on peek/cat
# for (drop_out, use_dropout, format) in product(dropout_lst, gpt2_drop_lst, format_lst):
#     os.system("python train_e2e.py --dropout {} --use_dropout {}  --format_mode {} --submit yes".format(drop_out, use_dropout, format))
#     total_experiment_count += 1
#
# # lr for peek/cat
# for (lr_rate, format) in product(lr_lst, format_lst):
#     os.system("python train_e2e.py --learning_rate {} --format_mode {} --submit yes".format(lr_rate, format))
#     total_experiment_count += 1


# init random for peek/cat to be compared with optim_prefix=yes
# for (init_random, format) in product(random_init_lst, format_lst):
#     os.system("python train_e2e.py --init_random {} --format_mode {} --submit yes --notes low".format(init_random, format))
#     total_experiment_count += 1
#
#
# for (drop_out, use_dropout) in product(dropout_lst, gpt2_drop_lst):
#     os.system("python train_e2e.py --optim_prefix yes --submit yes --format_mode peek --dropout {} --use_dropout {} --notes low".format(drop_out, use_dropout))
#     total_experiment_count += 1


##################### infix #######################################
# epoch_lst = [5, 20]
# for epoch in epoch_lst:
#     os.system("python train_e2e.py  --mode data2text --bsz 10 --epoch {} --format_mode infix --submit yes ".format(epoch))
#     total_experiment_count += 1
#
#
# lr_lst = [0.001, 0.0005, 0.0001]
# for lr in lr_lst:
#     os.system("python train_e2e.py  --mode data2text --learning_rate {} --bsz 10 --epoch 10 --format_mode infix --submit yes".format(lr))
#     total_experiment_count += 1

# epoch_lst = [5, 20, 10]
# for epoch in epoch_lst:
#     os.system("python train_e2e.py  --mode data2text --tuning_mode finetune --bsz 10 --epoch {} --submit yes".format(epoch))
#     total_experiment_count += 1
#
# lr_lst = [0.001, 0.0005, 0.0001]
# for lr in lr_lst:
#     os.system("python train_e2e.py  --mode data2text --tuning_mode finetune --learning_rate {} --bsz 10 --epoch 10 --submit yes".format(lr))
#     total_experiment_count += 1


################## EMBEDDING #######
# epoch_lst = [5, 20, 10]
# for epoch in epoch_lst:
#     os.system("python train_e2e.py  --mode data2text --prefix_mode embedding --bsz 10 --epoch {} --submit yes".format(epoch))
#     os.system(
#         "python train_e2e.py  --mode data2text --format_mode peek --prefix_mode embedding --bsz 10 --epoch {} --submit yes".format(epoch))
#     total_experiment_count += 2
#
# lr_lst = [0.001, 0.0005, 0.0001]
# for lr in lr_lst:
#     os.system(
#         "python train_e2e.py  --mode data2text --prefix_mode embedding --bsz 10 --epoch 10 --learning_rate {} --submit yes".format(
#             lr))
#
#     os.system(
#         "python train_e2e.py  --mode data2text --format_mode peek --prefix_mode embedding --bsz 10 --epoch 10 --learning_rate {} --submit yes".format(
#             lr))
#     total_experiment_count += 2
#
#
# for (init_random, format) in product(random_init_lst, format_lst):
#     os.system("python train_e2e.py --prefix_mode embedding  --epoch 10 --init_random {} --format_mode {} "
#               "--submit yes --notes low".format(init_random, format))
#     total_experiment_count += 1
#
# # size of the mid layer.
# mid_dim_lst = [200, 512, 1000, 2000, 5000]
# for (format, mid_dim) in product(format_lst, mid_dim_lst):
#     os.system("python train_e2e.py --format_mode {} --mid_dim {} --submit yes --prefix_mode embedding --epoch 10 ".format(format, mid_dim))
#     total_experiment_count+= 1
#
# # use_optim verions yes and no.
# seqlen_lst = [1, 20, 5, 10]
# for (format, seqlen) in product(format_lst, seqlen_lst):
#     os.system("python train_e2e.py --prefix_mode embedding --epoch 10 --optim_prefix yes --submit yes "
#               "--format_mode cat --format_mode {} --preseqlen {}".format(format, seqlen))
#     total_experiment_count += 1


################## cleaned composition training #######
# notes_lst = ['Type', 'customer', 'food', 'area', 'price'] # near
# for note in notes_lst:
#     os.system("python train_e2e.py --epoch 10 --submit yes --notes {}".format(note))
#     os.system("python train_e2e.py --optim_prefix yes --epoch 10 --preseqlen 1 --submit yes --notes {}".format(note))
#     os.system("python train_e2e.py --optim_prefix yes --epoch 10 --preseqlen 10 --submit yes --notes {}".format(note))
#     total_experiment_count += 3
#


###################Low data regime ##############
# notes_lst = ['Type', 'customer', 'food', 'area', 'price'] # near
# sent_len_lst = [10, 100, 1000, 5000, 10000]
# num_label_lst = list(range(10))
# for (label, sent_len) in product(num_label_lst, sent_len_lst):
#     note = 'lowdata_{}_{}'.format(label, sent_len)
#     # os.system("python train_e2e.py --epoch 10 --submit yes --notes {}".format(note))
#     os.system("python train_e2e.py --tuning_mode finetune --epoch 10 --submit yes --notes {}".format(note))
#     # os.system("python train_e2e.py --optim_prefix yes --epoch 10 --preseqlen 1 --submit yes --notes {}".format(note))
#     # os.system("python train_e2e.py --optim_prefix yes --epoch 10 --preseqlen 10 --submit yes --notes {}".format(note))
#     total_experiment_count += 1



# #################### low data for datasize=100 #########################
# # num_label_lst = list(range(3,10))
# num_label_lst = list(range(6))
# max_steps_lst = [100, 200, 300, 400, 500, 600]
# datasize_lst = [50, 200]
# for (label, max_steps, datasize) in product(num_label_lst, max_steps_lst, datasize_lst):
#     note = 'lowdata_{}_{}'.format(label, datasize)
#     eval_steps = max_steps // 10
#     # os.system("python train_e2e.py --submit yes --max_steps {} --eval_steps {} --notes {}".format(max_steps, eval_steps, note))
#
#
#     os.system("python train_e2e.py --optim_prefix yes --preseqlen 1 --submit yes --max_steps {} --eval_steps {}"
#               " --notes {}".format(max_steps, eval_steps, note))
#     os.system("python train_e2e.py --optim_prefix yes --preseqlen 5 --submit yes --max_steps {} --eval_steps {}"
#               " --notes {}".format(max_steps, eval_steps, note))
#     os.system("python train_e2e.py --optim_prefix yes --preseqlen 10 --submit yes --max_steps {} --eval_steps {}"
#               " --notes {}".format(max_steps, eval_steps, note))
#     os.system("python train_e2e.py --optim_prefix yes --preseqlen 20 --submit yes --max_steps {} --eval_steps {}"
#               " --notes {}".format(max_steps, eval_steps, note))
#
#     # os.system("python train_e2e.py --submit yes --tuning_mode finetune"
#     #           " --max_steps {} --eval_steps {} --notes {}".format(max_steps, eval_steps, note))
#
#     total_experiment_count += 1
#
#
# print('submitted {} jobs in total'.format(total_experiment_count))



#################### low data for datasize=100 #########################
# num_label_lst = list(range(3,10))
# num_label_lst = range(5)
# max_steps_lst = [400, 600]
# datasize_lst = [100]
# mid_dim_lst = [600]
# for (label, max_steps, datasize, mid_dim) in product(num_label_lst, max_steps_lst, datasize_lst, mid_dim_lst):
#     note = 'lowdata_{}_{}'.format(label, datasize)
#     # eval_steps = max_steps // 10
#     # warmup_steps_temp = max_steps // 4
#     # os.system("python train_e2e.py --submit yes --max_steps {} --eval_steps {} --notes {}".format(max_steps, eval_steps, note))
#
# #python train_e2e.py --optim_prefix yes --preseqlen 1 --notes trlowdata_9_100 --max_steps 800 --eval_steps 50 --mid_dim 600 --submit yes
#     # os.system("python train_e2e.py --optim_prefix yes --preseqlen 1 --submit yes --max_steps {} --eval_steps {}"
#     #           " --notes hap{} --mid_dim {} --warmup_steps {}".format(max_steps, 50, note, mid_dim, warmup_steps_temp))
#     # os.system("python train_e2e.py --optim_prefix yes --preseqlen 3 --submit yes --max_steps {} --eval_steps {}"
#     #           " --notes hap{} --mid_dim {} --warmup_steps {}".format(max_steps, 50, note, mid_dim, warmup_steps_temp ))
#
#
#     # os.system("python train_e2e.py --optim_prefix yes --preseqlen 10 --submit yes --max_steps {} --eval_steps {}"
#     #           " --notes {}".format(max_steps, eval_steps, note))
#     # os.system("python train_e2e.py --optim_prefix yes --preseqlen 20 --submit yes --max_steps {} --eval_steps {}"
#     #           " --notes {}".format(max_steps, eval_steps, note))
#
#     # os.system("python train_e2e.py --submit yes --tuning_mode finetune"
#     #           " --max_steps {} --eval_steps {} --notes {}".format(max_steps, eval_steps, note))
#
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen 52 --notes tempnew{} --max_steps {} --seed 200 --submit yes ".format(note, max_steps)
#     command2 = "python train_e2e.py --optim_prefix yes --preseqlen 52 --notes tempnew{} --max_steps {} --seed 101 --submit yes ".format(note, max_steps)
#
#     os.system(command1)
#     os.system(command2)
#
#
#     total_experiment_count += 2


############################################################################
###################### FOR 100 ####################################
# total_experiment_count = 0
# num_label_lst = list(range(5))
# max_steps_lst = [600, 200, 400]
# prompt_lst = ['summarize', 'table-to-text:']
# random_seed_lst = [200, 9]
# # for (label, max_steps, seed, prompt) in product(num_label_lst, max_steps_lst, random_seed_lst, prompt_lst):
# #     note = 'Alowdata_{}_{}'.format(label, 100)
# #     command1 = "python train_e2e.py --optim_prefix yes --preseqlen 51 --notes {} --max_steps {} --seed {} " \
# #                "--submit yes --lowdata_token {} --warmup_steps 0".format(note, max_steps, seed, prompt )
# #     print(command1)
# #     os.system(command1)
# #     total_experiment_count += 1
# # print('submitted {} jobs in total'.format(total_experiment_count))
#
#
# # set the seed to 400 for finetune.
# total_experiment_count = 0
# max_steps_lst = [100]
# for (label, max_steps, seed, prompt) in product(num_label_lst, max_steps_lst, random_seed_lst, prompt_lst):
#     note = 'Alowdata_{}_{}'.format(label, 100)
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen 51 --notes {} --max_steps {} --seed {} " \
#                "--submit yes --lowdata_token {} --warmup_steps 0 --tuning_mode finetune".format(note, max_steps, seed, prompt)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))



############################################################################
###################### FOR 10 ####################################
# total_experiment_count = 0
# num_label_lst = list(range(5))
# max_steps_lst = [600, 200, 400]
# prompt_lst = ['summarize', 'table-to-text:']
# random_seed_lst = [200, 9]
# for (label, max_steps, seed, prompt) in product(num_label_lst, max_steps_lst, random_seed_lst, prompt_lst):
#     note = 'Blowdata_{}_{}'.format(label, 10)
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen 51 --notes {} --max_steps {} --seed {} " \
#                "--submit yes --lowdata_token {} --warmup_steps 0".format(note, max_steps, seed, prompt )
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))
#
#
# # set the seed to 400 for finetune.
# total_experiment_count = 0
# max_steps_lst = [100, 200]
# for (label, max_steps, seed, prompt) in product(num_label_lst, max_steps_lst, random_seed_lst, prompt_lst):
#     note = 'Blowdata_{}_{}'.format(label, 10)
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen 51 --notes {} --max_steps {} --seed {} " \
#                "--submit yes --lowdata_token {} --warmup_steps 0 --tuning_mode finetune".format(note, max_steps, seed, prompt)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))



############################################################################
###################### FOR 50 ####################################

# total_experiment_count = 0
# num_label_lst = list(range(5))
# max_steps_lst = [600, 200, 400]
# prompt_lst = ['summarize', 'table-to-text:']
# random_seed_lst = [200, 9]
# for (label, max_steps, seed, prompt) in product(num_label_lst, max_steps_lst, random_seed_lst, prompt_lst):
#     note = 'Clowdata_{}_{}'.format(label, 50)
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen 51 --notes {} --max_steps {} --seed {} " \
#                "--submit yes --lowdata_token {} --warmup_steps 0".format(note, max_steps, seed, prompt )
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))
#
#
# # set the seed to 400 for finetune.
# total_experiment_count = 0
# max_steps_lst = [100, 200]
# for (label, max_steps, seed, prompt) in product(num_label_lst, max_steps_lst, random_seed_lst, prompt_lst):
#     note = 'Clowdata_{}_{}'.format(label, 50)
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen 51 --notes {} --max_steps {} --seed {} " \
#                "--submit yes --lowdata_token {} --warmup_steps 0 --tuning_mode finetune".format(note, max_steps, seed, prompt)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))


############################################################################
###################### FOR 200 ####################################


# total_experiment_count = 0
# num_label_lst = list(range(5))
# max_steps_lst = [600, 200, 400]
# prompt_lst = ['summarize', 'table-to-text:']
# random_seed_lst = [200, 9]
# for (label, max_steps, seed, prompt) in product(num_label_lst, max_steps_lst, random_seed_lst, prompt_lst):
#     note = 'Dlowdata_{}_{}'.format(label, 200)
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen 51 --notes {} --max_steps {} --seed {} " \
#                "--submit yes --lowdata_token {} --warmup_steps 0".format(note, max_steps, seed, prompt )
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))
#
#
# # set the seed to 400 for finetune.
# total_experiment_count = 0
# max_steps_lst = [200, 400]
# for (label, max_steps, seed, prompt) in product(num_label_lst, max_steps_lst, random_seed_lst, prompt_lst):
#     note = 'Dlowdata_{}_{}'.format(label, 200)
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen 51 --notes {} --max_steps {} --seed {} " \
#                "--submit yes --lowdata_token {} --warmup_steps 0 --tuning_mode finetune".format(note, max_steps, seed, prompt)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))


############################################################################
###################### FOR 500 ####################################

# total_experiment_count = 0
# num_label_lst = list(range(5))
# max_steps_lst = [600, 200, 400]
# prompt_lst = ['summarize', 'table-to-text:']
# random_seed_lst = [200, 9]
# for (label, max_steps, seed, prompt) in product(num_label_lst, max_steps_lst, random_seed_lst, prompt_lst):
#     note = 'Elowdata_{}_{}'.format(label, 500)
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen 51 --notes {} --max_steps {} --seed {} " \
#                "--submit yes --lowdata_token {} --warmup_steps 0".format(note, max_steps, seed, prompt )
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))
#
#
# # set the seed to 400 for finetune.
# total_experiment_count = 0
# max_steps_lst = [200, 400]
# for (label, max_steps, seed, prompt) in product(num_label_lst, max_steps_lst, random_seed_lst, prompt_lst):
#     note = 'Elowdata_{}_{}'.format(label, 500)
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen 51 --notes {} --max_steps {} --seed {} " \
#                "--submit yes --lowdata_token {} --warmup_steps 0 --tuning_mode finetune".format(note, max_steps, seed, prompt)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))


############################ FOR 100 without initialization trick. #####################################################
# total_experiment_count = 0
# num_label_lst = list(range(5))
# max_steps_lst = [600, 200, 400]
# preseqlen_lst = [1, 6]
# random_seed_lst = [200, 9]
# for (label, max_steps, seed, preseqlen) in product(num_label_lst, max_steps_lst, random_seed_lst, preseqlen_lst):
#     note = 'Flowdata_{}_{}'.format(label, 100)
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} --notes {} --max_steps {} --seed {} " \
#                "--submit yes --warmup_steps 0 --use_lowdata_token no ".format(preseqlen, note, max_steps, seed )
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))


############################ Ablation Study with Embeddings #################################
# total_experiment_count = 0
# preseqlen_lst = [1, 5, 10, 20]
# reparam_method_lst = ['MLP', 'Emb']
# for (parametrize_emb, preseqlen) in product(reparam_method_lst, preseqlen_lst):
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} --prefix_mode embedding " \
#                " --submit yes --parametrize_emb {} --epoch 10 ".format(preseqlen, parametrize_emb)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))
#
# total_experiment_count = 0
# mid_dim_lst = [256, 1024]
# for (mid_dim, preseqlen) in product(mid_dim_lst, preseqlen_lst):
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} --prefix_mode embedding " \
#                " --submit yes --mid_dim {} --epoch 10 ".format(preseqlen, mid_dim)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))

########################### INFIX v.s. Prefix ##############################
# total_experiment_count = 0
# preseqlen_lst = [1, 5, 10, 20]
# format_lst = ['infix']
# for (format, preseqlen) in product(format_lst, preseqlen_lst):
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} " \
#                " --submit yes  --epoch 10 --format_mode {} ".format(preseqlen, format)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))

########################### WebNLG-medium debugged ##############################
# total_experiment_count = 0
# preseqlen_lst = [5, 20, 30, 10]
# lr_lst = [0.00001, 0.00003, 0.00005]
# epoch_lst = [5, 10]
# for (epoch, preseqlen, lr) in product(epoch_lst, preseqlen_lst, lr_lst):
#     if epoch == 10 and lr == 0.00001:
#         continue
#
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} " \
#                " --submit yes  --epoch {} --learning_rate {} --bsz 5 --mode webnlg --notes earlystop3".format(preseqlen, epoch, lr)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))




########################### Xsum debugged ##############################
# total_experiment_count = 0
# preseqlen_lst = [40]
# lr_lst = [0.00003, 0.00005, 0.00008]
# grad_accumulation_lst = [10, 5, 3]
# epoch_lst = [5]
# for (grad_accumulation, preseqlen, lr) in product(grad_accumulation_lst, preseqlen_lst, lr_lst):
#
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} " \
#                " --submit yes  --epoch 5 --learning_rate {} --gradient_accumulation_steps {} --bsz 2 " \
#                " --mode xsum --notes newearlystop2 ".format(preseqlen, lr, grad_accumulation)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))


########################### E2E-medium debugged ##############################
# total_experiment_count = 0
# preseqlen_lst = [10, 20, 1, 5]
# lr_lst = [0.00001, 0.00003, 0.00005, 0.00008]
# epoch_lst = [5] # 10 already tried 10, todo try 5.
# for (preseqlen, lr, epoch) in product(preseqlen_lst, lr_lst, epoch_lst):
#
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} " \
#                " --submit yes  --epoch {} --learning_rate {} --mode data2text --notes newearlystop2".format(preseqlen, epoch, lr)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))


########################## WebNLG-large debugged ##############################
# total_experiment_count = 0
# preseqlen_lst = [5, 20, 30, 10]
# lr_lst = [0.00003, 0.00005,  0.00008]
# epoch_lst = [5]
# for (epoch, preseqlen, lr) in product(epoch_lst, preseqlen_lst, lr_lst):
#
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} " \
#                " --submit yes  --epoch {} --learning_rate {} --bsz 2 --gradient_accumulation_steps 3 --mode webnlg --notes large_earlystop4".format(preseqlen, epoch, lr)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))

########## TODO #############
########################## E2E-large debugged ##############################
# total_experiment_count = 0
# preseqlen_lst = [10, 20, 1, 5]
# lr_lst = [0.00001, 0.00003, 0.00005, 0.00008]
# epoch_lst = [5] # 10 already tried 10, todo try 5.
# for (preseqlen, lr, epoch) in product(preseqlen_lst, lr_lst, epoch_lst):
#
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} " \
#                " --submit yes  --epoch {} --learning_rate {} --mode data2text --notes large_newearlystop3".format(preseqlen, epoch, lr)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))


########## TODO #############
########################## DART-medium debugged ##############################
# total_experiment_count = 0
# preseqlen_lst = [20, 30, 10, 40]
# lr_lst = [0.00008, 0.00003, 0.00005]
# epoch_lst = [5]
# for (epoch, preseqlen, lr) in product(epoch_lst, preseqlen_lst, lr_lst):
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} " \
#                " --submit yes  --epoch {} --learning_rate {} --bsz 5 --mode triples --notes earlystop3".format(preseqlen, epoch, lr)
#     print(command1)
#     # os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))


########################## WebNLG-large debugged 2 DONE ##############################
# total_experiment_count = 0
# preseqlen_lst = [20, 30, 10]
# lr_lst = [0.00008, 0.00005,  0.00003]
# epoch_lst = [10]
# for (epoch, lr, preseqlen) in product(epoch_lst, lr_lst, preseqlen_lst):
#
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} " \
#                " --submit yes  --epoch {} --learning_rate {} --bsz 2 --gradient_accumulation_steps 3 --mode webnlg --notes large_earlystop4".format(preseqlen, epoch, lr)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))



########################## WebNLG-medium full objective version... DONE ##############################
# total_experiment_count = 0
# preseqlen_lst = [20]
# lr_lst = [ 0.00005, 0.00008,  0.00003]
# objective_mode_lst = [0, 1, 2, 3, 4]
# epoch_lst = [10]
# for (epoch, lr, preseqlen, objective_mode) in product(epoch_lst, lr_lst, preseqlen_lst, objective_mode_lst):
#
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} " \
#                " --submit yes  --epoch {} --learning_rate {} --bsz 5 --mode webnlg --notes earlystopA " \
#                "--objective_mode {} ".format(preseqlen, epoch, lr, objective_mode)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))



############################################################################################
############################################################################################

########################### E2E-medium o=1 ##############################
# total_experiment_count = 0
# preseqlen_lst = [10, 20, 1, 3, 5]
# lr_lst = [0.00005]
# epoch_lst = [10, 5] # 10 already tried 10, todo try 5.
# for (preseqlen, lr, epoch) in product(preseqlen_lst, lr_lst, epoch_lst):
#
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} " \
#                " --submit yes  --epoch {} --learning_rate {} --mode data2text " \
#                " --objective_mode 1 ".format(preseqlen, epoch, lr)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))

#
# ########################### E2E-large o=1 ##############################
# total_experiment_count = 0
# preseqlen_lst = [10, 20, 1, 3, 5]
# lr_lst = [0.00005]
# epoch_lst = [10, 5] # 10 already tried 10, todo try 5.
# for (preseqlen, lr, epoch) in product(preseqlen_lst, lr_lst, epoch_lst):
#
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} " \
#                " --submit yes  --epoch {} --learning_rate {} --mode data2text --notes large_earlystop5 --bsz 5 " \
#                " --objective_mode 1 ".format(preseqlen, epoch, lr)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))




# ########################## DART-large o=1 ##############################
# total_experiment_count = 0
# preseqlen_lst = [40, 20, 30, 10]
# lr_lst = [0.00005, 0.00003]
# epoch_lst = [5]
# for (epoch, lr, preseqlen) in product(epoch_lst, lr_lst, preseqlen_lst):
#
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} " \
#                " --submit yes  --epoch {} --learning_rate {} --bsz 2 --gradient_accumulation_steps 3" \
#                " --mode triples --notes large_earlystop5 --objective_mode 1 ".format(preseqlen, epoch, lr)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))


########################## WebNLG-large o=1 ##############################
# total_experiment_count = 0
# preseqlen_lst = [20, 30, 10, 5]
# lr_lst = [0.00005, 0.00003]
# epoch_lst = [5]
# for (epoch, lr, preseqlen) in product(epoch_lst, lr_lst, preseqlen_lst):
#
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} " \
#                " --submit yes  --epoch {} --learning_rate {} --bsz 2 --gradient_accumulation_steps 3" \
#                " --mode webnlg --notes large_earlystop5 --objective_mode 1 ".format(preseqlen, epoch, lr)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))


# ########################### E2E-large o=1 (second tune) ##############################
# total_experiment_count = 0
# preseqlen_lst = [10, 1, 5]
# lr_lst = [0.00003, 0.00001]
# epoch_lst = [10, 5] # 10 already tried 10, todo try 5.
# for (preseqlen, lr, epoch) in product(preseqlen_lst, lr_lst, epoch_lst):
#
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} " \
#                " --submit yes  --epoch {} --learning_rate {} --mode data2text --notes large_earlystopsmall --bsz 5 " \
#                " --objective_mode 1 ".format(preseqlen, epoch, lr)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))


########################## WebNLG-large o=1 (second tune) ##############################
# total_experiment_count = 0
# preseqlen_lst = [20, 30, 10]
# lr_lst = [0.00001] #[0.00005,0.00008, 0.00003]
# epoch_lst = [5] # [10]
# for (epoch, lr, preseqlen) in product(epoch_lst, lr_lst, preseqlen_lst):
#
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} " \
#                " --submit yes  --epoch {} --learning_rate {} --bsz 2 --gradient_accumulation_steps 3" \
#                " --mode webnlg --notes large_earlystop5 --objective_mode 1 ".format(preseqlen, epoch, lr)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))


# ########################## DART-large o=1 (second tune) ##############################
# total_experiment_count = 0
# preseqlen_lst = [40, 20, 30]
# lr_lst = [0.00001] #[0.00001, 0.00003, 0.00005]
# epoch_lst = [5] #[10]
# for (epoch, lr, preseqlen) in product(epoch_lst, lr_lst, preseqlen_lst):
#
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} " \
#                " --submit yes  --epoch {} --learning_rate {} --bsz 2 --gradient_accumulation_steps 3" \
#                " --mode triples --notes large_earlystop5 --objective_mode 1 ".format(preseqlen, epoch, lr)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))



############################################################################
###################### FOR 100 ####################################
# total_experiment_count = 0
# num_label_lst = [4] #list(range(5))
# max_steps_lst = [600, 200, 400]
# prompt_lst = ['elephant', 'active'] #['keep', 'divide','banana', 'beautiful']#['elephant', 'active']#['summarize', 'table-to-text:'] #'debate'
# lr_lst = [0.00005]
# warmup_lst = [100]
# random_seed_lst = [200, 9]
# for (label, max_steps, seed, prompt, lr, ws) in product(num_label_lst, max_steps_lst, random_seed_lst, prompt_lst, lr_lst, warmup_lst):
#     note = 'Alowdata_{}_{}'.format(label, 100)
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen 51 --notes {} --max_steps {} --seed {} " \
#                "--submit yes --lowdata_token {} --warmup_steps {}  --objective_mode 0 --learning_rate {} ".format(note, max_steps, seed, prompt, ws, lr, )
#     print(command1)
#     # os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))


############################ FOR 100 without initialization trick. #####################################################
# total_experiment_count = 0
# num_label_lst = list(range(5))
# max_steps_lst = [600, 200, 400]
# preseqlen_lst = [1, 6]
# random_seed_lst = [200, 9]
# lr_lst = [0.00005, 0.00003]
# warmup_lst = [100, 0]
# for (label, max_steps, seed, preseqlen, lr, ws) in product(num_label_lst, max_steps_lst, random_seed_lst, preseqlen_lst, lr_lst, warmup_lst):
#     note = 'Blowdata_{}_{}'.format(label, 100)
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} --notes {} --max_steps {} --seed {} " \
#                "--submit yes --warmup_steps {} --use_lowdata_token no --objective_mode 0 --learning_rate {} ".format(preseqlen, note, max_steps, seed, ws, lr)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))



########################### E2E-medium o=1 ##############################
# total_experiment_count = 0
# preseqlen_lst = [10, 20, 1, 5, 40]
# lr_lst = [0.00005, 0.00003, 0.00008]
# epoch_lst = [10, 5, 15]
# seed_lst = [101] #101] #22 # 9
# for (seed, preseqlen, lr, epoch) in product(seed_lst, preseqlen_lst, lr_lst, epoch_lst):
#
#     if preseqlen != 1 and epoch == 15:
#         continue
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} " \
#                " --submit yes  --epoch {} --learning_rate {} --mode data2text " \
#                " --objective_mode 1 --notes AAearlystop --seed {} ".format(preseqlen, epoch, lr, seed)
#     print(command1)
#     # os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))


########################### WebNLG-medium o=1 ##############################
# total_experiment_count = 0
# preseqlen_lst = [10, 20, 1, 5, 40]
# lr_lst = [0.00005, 0.00003, 0.00008]
# epoch_lst = [10, 5]
# seed_lst = [101]#[22] # 9
# for (seed, preseqlen, lr, epoch) in product(seed_lst, preseqlen_lst, lr_lst, epoch_lst):
#
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} " \
#                " --submit yes  --epoch {} --learning_rate {} --mode webnlg " \
#                " --objective_mode 1 --bsz 5  --notes AAearlystop --seed {} ".format(preseqlen, epoch, lr, seed)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))

########################### Triples-medium o=1 ##############################
# total_experiment_count = 0
# preseqlen_lst = [10, 20, 1, 5, 40]
# lr_lst = [0.00005, 0.00003, 0.00008]
# epoch_lst = [10, 5]
# seed_lst = [101]#9, 101, 22
# for (seed, preseqlen, lr, epoch) in product(seed_lst, preseqlen_lst, lr_lst, epoch_lst):
#
#     command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} " \
#                " --submit yes  --epoch {} --learning_rate {} --mode triples " \
#                " --objective_mode 1 --bsz 5  --notes AAearlystop --seed {} ".format(preseqlen, epoch, lr, seed)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))


########################### WebNLG-medium o=0 ##############################
total_experiment_count = 0
preseqlen_lst = [10, 20, 1, 5, 40]
lr_lst = [0.00005, 0.00003, 0.00008]
epoch_lst = [10, 5]
seed_lst = [101]#[22] # 9
for (seed, preseqlen, lr, epoch) in product(seed_lst, preseqlen_lst, lr_lst, epoch_lst):

    command1 = "python train_e2e.py --optim_prefix yes --preseqlen {} " \
               " --submit yes  --epoch {} --learning_rate {} --mode webnlg " \
               " --objective_mode 0 --bsz 5  --notes AAearlystop --seed {} ".format(preseqlen, epoch, lr, seed)
    print(command1)
    os.system(command1)
    total_experiment_count += 1
print('submitted {} jobs in total'.format(total_experiment_count))