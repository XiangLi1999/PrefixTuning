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

############################################################################
###################### FOR 100 ####################################
# total_experiment_count = 0
# num_label_lst = [101, 22]
# epoch_lst = [20, 30]
# prompt_lst = ['summarize',]
# random_seed_lst = [200, 9, 42]
# for (label, epoch, seed, prompt) in product(num_label_lst, epoch_lst, random_seed_lst, prompt_lst):
#     note = 'Alowdata_{}_{}'.format(label, 100)
#     command1 = "python train_bart.py  --mode xsum --notes {} --epoch {} --seed {} " \
#                "--submit yes --lowdata_token {} --warmup_steps 0 --bsz 5 --warmup_steps 100 ".format(note, epoch, seed, prompt ) # have or not have warmup steps.
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))


# set the seed to 400 for finetune.
# prompt_lst = ['summarize']
# total_experiment_count = 0
# for (label, epoch, seed, prompt) in product(num_label_lst, epoch_lst, random_seed_lst, prompt_lst):
#     note = 'Alowdata_{}_{}'.format(label, 100)
#     command1 = "python train_bart.py --mode xsum --notes {} --epoch {} --seed {} " \
#                "--submit yes --lowdata_token {} --warmup_steps 0 --tuning_mode finetune --bsz 2 " \
#                "--gradient_accumulation_steps 3 ".format(note, epoch, seed, prompt)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))



############################################################################
###################### FOR 200 ####################################
# total_experiment_count = 0
# num_label_lst = [101, 22, 8]
# epoch_lst = [10] #[20, 30] # TODO[10]
# prompt_lst = ['summarize',]
# random_seed_lst = [200, 9, 42]
# for (label, epoch, seed, prompt) in product(num_label_lst, epoch_lst, random_seed_lst, prompt_lst):
#     note = 'Blowdata_{}_{}'.format(label, 200)
#     command1 = "python train_bart.py  --mode xsum --notes {} --epoch {} --seed {} " \
#                "--submit yes --lowdata_token {}  --bsz 5 --warmup_steps 100 ".format(note, epoch, seed, prompt ) # have or not have warmup steps.
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))


# set the seed to 400 for finetune.
# prompt_lst = ['summarize']
# epoch_lst = [20, 10]
# total_experiment_count = 0
# for (label, epoch, seed, prompt) in product(num_label_lst, epoch_lst, random_seed_lst, prompt_lst):
#     note = 'Blowdata_{}_{}'.format(label, 200)
#     command1 = "python train_bart.py --mode xsum --notes {} --epoch {} --seed {} " \
#                "--submit yes --lowdata_token {} --warmup_steps 0 --tuning_mode finetune --bsz 2 " \
#                "--gradient_accumulation_steps 3 ".format(note, epoch, seed, prompt)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))

############################################################################
###################### FOR 50 ####################################
# total_experiment_count = 0
# num_label_lst = [101, 22, 8]
# epoch_lst = [20, 30, 40]
# prompt_lst = ['summarize',]
# random_seed_lst = [200, 9, 42]

# for (label, epoch, seed, prompt) in product(num_label_lst, epoch_lst, random_seed_lst, prompt_lst):
#     note = 'Clowdata_{}_{}'.format(label, 50)
#     command1 = "python train_bart.py  --mode xsum --notes {} --epoch {} --seed {} " \
#                "--submit yes --lowdata_token {}  --bsz 5 --warmup_steps 100 ".format(note, epoch, seed, prompt ) # have or not have warmup steps.
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))


# prompt_lst = ['summarize']
# epoch_lst = [20, 30, 40]
# total_experiment_count = 0
# for (label, epoch, seed, prompt) in product(num_label_lst, epoch_lst, random_seed_lst, prompt_lst):
#     note = 'Clowdata_{}_{}'.format(label, 50)
#     command1 = "python train_bart.py --mode xsum --notes {} --epoch {} --seed {} " \
#                "--submit yes --lowdata_token {} --warmup_steps 0 --tuning_mode finetune --bsz 2 " \
#                "--gradient_accumulation_steps 3 ".format(note, epoch, seed, prompt)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))





############################################################################
###################### FOR 500 ####################################
total_experiment_count = 0
num_label_lst = [101, 22, 8]
epoch_lst = [20, 10, 5]
prompt_lst = ['summarize',]
random_seed_lst = [200, 9, 42]
# # for (label, epoch, seed, prompt) in product(num_label_lst, epoch_lst, random_seed_lst, prompt_lst):
# #     note = 'Flowdata_{}_{}'.format(label, 500)
# #     command1 = "python train_bart.py  --mode xsum --notes {} --epoch {} --seed {} " \
# #                "--submit yes --lowdata_token {}  --bsz 5 --warmup_steps 100 --learning_rate 0.00005".format(note, epoch, seed, prompt ) # have or not have warmup steps.
# #     print(command1)
# #     os.system(command1)
# #     total_experiment_count += 1
# # print('submitted {} jobs in total'.format(total_experiment_count))
#
#
prompt_lst = ['summarize']
epoch_lst = [5, 3]
total_experiment_count = 0
for (label, epoch, seed, prompt) in product(num_label_lst, epoch_lst, random_seed_lst, prompt_lst):
    note = 'Jlowdata_{}_{}'.format(label, 500)
    command1 = "python train_bart.py --mode xsum --notes {} --epoch {} --seed {} " \
               "--submit yes --lowdata_token {} --warmup_steps 0 --tuning_mode finetune --bsz 2 " \
               "--gradient_accumulation_steps 3 ".format(note, epoch, seed, prompt)
    print(command1)
    os.system(command1)
    total_experiment_count += 1
print('submitted {} jobs in total'.format(total_experiment_count))




############################################################################
###################### FOR 50  (second trial) ####################################
# total_experiment_count = 0
# num_label_lst = [101, 22, 8]
# epoch_lst = [20, 30, 40]
# prompt_lst = ['summarize',]
# random_seed_lst = [200, 9, 42]
#
# for (label, epoch, seed, prompt) in product(num_label_lst, epoch_lst, random_seed_lst, prompt_lst):
#     note = 'Glowdata_{}_{}'.format(label, 50)
#     command1 = "python train_bart.py  --mode xsum --notes {} --epoch {} --seed {} " \
#                "--submit yes --lowdata_token {}  --bsz 5 --warmup_steps 100 ".format(note, epoch, seed, prompt ) # have or not have warmup steps.
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))
#
#
# prompt_lst = ['summarize']
# epoch_lst = [20, 30, 40]
# total_experiment_count = 0
# for (label, epoch, seed, prompt) in product(num_label_lst, epoch_lst, random_seed_lst, prompt_lst):
#     note = 'Glowdata_{}_{}'.format(label, 50)
#     command1 = "python train_bart.py --mode xsum --notes {} --epoch {} --seed {} " \
#                "--submit yes --lowdata_token {} --warmup_steps 0 --tuning_mode finetune --bsz 2 " \
#                "--gradient_accumulation_steps 3 ".format(note, epoch, seed, prompt)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))



############################################################################
###################### FOR 200 (second trial) ####################################
# total_experiment_count = 0
# num_label_lst = [101, 22, 8]
# epoch_lst = [10, 20, 30] # TODO[10]
# prompt_lst = ['summarize',]
# random_seed_lst = [200, 9, 42]
# for (label, epoch, seed, prompt) in product(num_label_lst, epoch_lst, random_seed_lst, prompt_lst):
#     note = 'Hlowdata_{}_{}'.format(label, 200)
#     command1 = "python train_bart.py  --mode xsum --notes {} --epoch {} --seed {} " \
#                "--submit yes --lowdata_token {}  --bsz 5 --warmup_steps 100 ".format(note, epoch, seed, prompt ) # have or not have warmup steps.
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))
#
#
prompt_lst = ['summarize']
epoch_lst = [5, 3]
total_experiment_count = 0
for (label, epoch, seed, prompt) in product(num_label_lst, epoch_lst, random_seed_lst, prompt_lst):
    note = 'Jlowdata_{}_{}'.format(label, 200)
    command1 = "python train_bart.py --mode xsum --notes {} --epoch {} --seed {} " \
               "--submit yes --lowdata_token {} --warmup_steps 0 --tuning_mode finetune --bsz 2 " \
               "--gradient_accumulation_steps 3 ".format(note, epoch, seed, prompt)
    print(command1)
    os.system(command1)
    total_experiment_count += 1
print('submitted {} jobs in total'.format(total_experiment_count))


############################################################################
###################### FOR 100 (second trial) ####################################
# total_experiment_count = 0
# num_label_lst = [8, 101, 22]
# epoch_lst = [20, 30]
# prompt_lst = ['summarize',]
# random_seed_lst = [200, 9, 42]
# # for (label, epoch, seed, prompt) in product(num_label_lst, epoch_lst, random_seed_lst, prompt_lst):
# #     note = 'Ilowdata_{}_{}'.format(label, 100)
# #     command1 = "python train_bart.py  --mode xsum --notes {} --epoch {} --seed {} " \
# #                "--submit yes --lowdata_token {} --bsz 5 --warmup_steps 100 ".format(note, epoch, seed, prompt ) # have or not have warmup steps.
# #     print(command1)
# #     os.system(command1)
# #     total_experiment_count += 1
# # print('submitted {} jobs in total'.format(total_experiment_count))
#
#
# prompt_lst = ['summarize']
# epoch_lst = [10] #[20, 30]
# total_experiment_count = 0
# for (label, epoch, seed, prompt) in product(num_label_lst, epoch_lst, random_seed_lst, prompt_lst):
#     note = 'Ilowdata_{}_{}'.format(label, 100)
#     command1 = "python train_bart.py --mode xsum --notes {} --epoch {} --seed {} " \
#                "--submit yes --lowdata_token {} --warmup_steps 0 --tuning_mode finetune --bsz 2 " \
#                "--gradient_accumulation_steps 3 ".format(note, epoch, seed, prompt)
#     print(command1)
#     os.system(command1)
#     total_experiment_count += 1
# print('submitted {} jobs in total'.format(total_experiment_count))

# DONE : python train_bart.py --mode xsum --notes Alowdata_22_100 --epoch 30 --seed 42  --lowdata_token summarize --warmup_steps 0 --tuning_mode finetune --bsz 2 --gradient_accumulation_steps 3
# rouge1=32.403
# rouge2=10.649
# rougeL=24.813
# {'rouge1': 32.4118, 'rouge2': 10.6607, 'rougeL': 24.8023}

# python train_bart.py --mode xsum --notes lowdata_8_100 --preseqlen 20 --bsz 2 --gradient_accumulation_steps 3  --epoch 20 --tuning_mode finetune
