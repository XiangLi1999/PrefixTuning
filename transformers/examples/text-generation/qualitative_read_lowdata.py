import os, sys
import glob
from collections import defaultdict

def list_to_line(lst, line):
    for name in lst:
        # print(name)
        line_result = open(name, "r").readlines()[line]
        print(line_result)
    print()
    return

hyperparam_dict = {('data2textprefixtunet=table', 100): '100st=200',
                   ('data2textfinetunet=table', 100): '100st=100',
                   ('data2textprefixtunet=table', 50): '50st=200',
                   ('data2textfinetunet=table', 50): '50st=100',
                   ('data2textprefixtunet=table', 200): '200st=200',
                   ('data2textfinetunet=table', 200): '200st=200',
                   ('data2textprefixtunet=table', 500): '500st=400',
                   ('data2textfinetunet=table', 500): '500st=400',
                   }

file_lst = glob.glob('e2e_results_conv/*lowdata*t=table-to-text:*beam')
print(file_lst)
name_dict_pt = defaultdict(list)
name_dict_ft = defaultdict(list)
for name in file_lst:
    if 'finetune' not in name and 'prefixtune' not in name:
        continue

    if 'Alowdata' not in name and 'Blowdata' not in name  and  'Clowdata'not in name  and 'Dlowdata'  not in name  \
            and 'Elowdata' not in name and 'Flowdata' not in name and  'Glowdata' not in name :
        continue

    base_name = os.path.basename(name)

    base_name = base_name.split('_')
    # print(base_name)

    train_label = base_name[14] # seeded dataset
    model_name =base_name[0] + base_name[-3].split('-')[0] #table or summary
    steps = base_name[15] # max_steps
    seed = base_name[10] # seeded training.
    train_num = int(base_name[15].split('st=')[0])

    if (model_name, train_num) not in hyperparam_dict:
        continue
    else:
        if steps != hyperparam_dict[(model_name, train_num)] :
            continue

    if train_label != str(0):
        continue

    print(train_label, model_name, steps, seed, train_num)

    if base_name[0] == 'data2textfinetune':
        name_dict_ft[train_num].append(name)
    else:
        name_dict_pt[train_num].append(name)

# print(name_dict)

list_to_line(['e2e_results_conv/data2textfinetune_y_51_act_cat_b=10-e=5_d=0.0_'
              'u=no_lr=5e-05_w=0.0_s=200_r=n_m=512_Clowdata_4_50st=100_ev=50_ws'
              '=0_t=table-to-text:-checkpoint-50_valid_src'], 300)
print(50, '-'*20)
list_to_line(name_dict_pt[50], 300)

print(100, '-'*20)
list_to_line(name_dict_pt[100], 300)

print(200, '-'*20)
list_to_line(name_dict_pt[200], 300)

print(500, '-'*20)
list_to_line(name_dict_pt[500], 300)

print('-'*300)
print(50, '-'*20)
list_to_line(name_dict_ft[50], 300)

print(100, '-'*20)
list_to_line(name_dict_ft[100], 300)

print(200, '-'*20)
list_to_line(name_dict_ft[200], 300)

print(500, '-'*20)
list_to_line(name_dict_ft[500], 300)

    #
    # result_dict[model_name][train_num][steps].append(
    #     (float(bleu), float(nist), float(meteor), float(rougel), float(cider)))


# print(result_dict)
# print(result_dict['data2textfinetune_n_20'][100])
# prone down hyperparam for data building.