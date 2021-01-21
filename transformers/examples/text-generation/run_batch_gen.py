import glob
import os
import sys
# temp = glob.glob("/u/scr/xlisali/contrast_LM/transformers/examples/control/save_e2e_models/*")
# temp = glob.glob("/u/scr/xlisali/contrast_LM/transformers/examples/control/*tune_y_*512")
#
# temp = glob.glob("/u/scr/xlisali/contrast_LM/transformers/examples/control/save_e2e_models_infix/*")

# temp = glob.glob("/u/scr/xlisali/contrast_LM/transformers/examples/control/save_e2e_models_infix/*")

# temp = glob.glob("/u/scr/xlisali/contrast_LM/transformers/examples/control/triples_models/*")

# temp = glob.glob("/u/scr/xlisali/contrast_LM/transformers/examples/control/webnlg_models/*")

# temp = glob.glob("/u/scr/xlisali/contrast_LM/transformers/examples/control/e2e_lowdata_models/*")


# temp = glob.glob("/u/scr/xlisali/contrast_LM/transformers/examples/control/e2e_lowdata_models_new/*")

# temp = glob.glob("/u/scr/xlisali/contrast_LM/transformers/examples/control/e2e_lowdata_models_finetune/*")

# temp = glob.glob("/u/scr/xlisali/contrast_LM/transformers/examples/control/e2e_lowdata_models_new/*")

# temp = glob.glob("/u/scr/xlisali/contrast_LM/transformers/examples/control/save_e2e_models_convcheck/*")

# temp = glob.glob("/u/scr/xlisali/contrast_LM/transformers/examples/control/classification_models/*")

temp = glob.glob("/u/scr/xlisali/contrast_LM/transformers/examples/control/save_e2e_models_convcheck/*large*o=1/*checkpoint*")




count_total = 0
for idx, elem in enumerate(temp):
    if elem[-3:] == '.sh'  or elem[-3:] == 'out':
        continue


    # check if the model finished training.
    # if len(glob.glob(elem+'/pytorch_model.bin')) > 0:
    #     print('yes')
    # else:
    #     continue

    sub_model_name = os.path.basename(elem)




    # os.system("python gen.py data2text yes yes {}".format(elem))

    if False:

        checkpoint_path = glob.glob(os.path.join(elem, '*checkpoint*'))
        assert len(checkpoint_path) == 1
        checkpoint_path = checkpoint_path[0]
        sub_model_name = os.path.basename(checkpoint_path)
        elem = checkpoint_path

    # if 'finetune' not in sub_model_name:
    #     #     continue
    #
    #     # if '1000' in sub_model_name or '5000' in sub_model_name:
    #     #     continue

    if ('lowdata_0_100' in  sub_model_name) or ('lowdata_1_100' in  sub_model_name) or ('lowdata_2_100' in  sub_model_name):
        print('BAD')
        continue


    # print(elem)
    count_total += 1
    if 'webnlg' in sub_model_name:
        os.system("python gen.py webnlg yes yes {} yes".format(elem))

    elif 'triples' in sub_model_name:
        os.system("python gen.py triples yes yes {} yes".format(elem))

    elif 'data2text' in sub_model_name:
        os.system("python gen.py data2text yes yes {} yes".format(elem))

    elif 'classify-sentiment' in sub_model_name:
        os.system("python gen.py classify-sentiment yes yes {} yes".format(elem))

    elif 'classify-topic' in sub_model_name:
        pass

    # print("python gen.py data2text yes yes {} yes".format(elem))


print(count_total)
