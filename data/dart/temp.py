import json, sys, os
from ast import literal_eval

with open(sys.argv[1], 'r') as f:
    result = json.load(f)
    print(len(result))



def read_triples_files(path, tokenizer_bos, src_tgt, new_path_out, new_path_src):
    file_dict = {}
    SAVE_REF = False


    out_handle = open(new_path_out, 'w')

    if SAVE_REF:
        src_handle = []
        for i in range(3):
            src_handle.append(open(new_path_src+str(i), 'w'))

    with open(path) as f:
        lines_dict = json.load(f)

    print(len(lines_dict))
    full_rela_lst = []
    full_src_lst = []
    for k, example in enumerate(lines_dict):
        rela_lst = []
        temp_triples = ''
        for i, tripleset in enumerate(example['tripleset']):
            subj, rela, obj = tripleset
            rela = rela.lower()
            rela_lst.append(rela)
            if i > 0:
                temp_triples += ' | '
            temp_triples += '{} : {} : {}'.format(subj, rela, obj)

        temp_triples = ' {} {}'.format(temp_triples, tokenizer_bos)

        ident_ = (temp_triples, tuple(rela_lst))

        assert ident_ in src_tgt

        sys_result = src_tgt[ident_]
        out_handle.write("{}\n".format(sys_result))

        temp_sents = [x['text'] for x in example['annotations'] if x['text'].strip()]
        # print(len(temp_sents), temp_sents, k)

        if SAVE_REF:
            for j in range(3):
                if j >= len(temp_sents):
                    src_handle[j].write("\n")
                else:
                    src_handle[j].write("{}\n".format(temp_sents[j]))

        # for (j, sent) in enumerate(temp_sents[:3]):
        #     # print(j, src_handle[j])
        #     src_handle[j].write("{}\n".format(sent))

    out_handle.close()

    if SAVE_REF:
        for i in range(3):
            src_handle[i].close()

    return


def src_tgt_pair(src_path, tgt_path):
    temp_src = []
    with open(src_path, 'r') as f_src:
        for line in f_src:
            temp_src.append(literal_eval(line.strip()))
    temp_tgt = []
    with open(tgt_path, 'r') as f_tgt:
        for line in f_tgt:
            temp_tgt.append(line.strip())

    assert len(temp_tgt) == len(temp_src)
    print(temp_src[0])
    print(temp_src[1])
    print(temp_src[3])

    bos_token = temp_src[0][0].split()[-1]
    print(bos_token)

    return {src:tgt for (src, tgt) in zip(temp_src, temp_tgt)}, bos_token




path = sys.argv[1]
dict_src_tgt, bos = src_tgt_pair(sys.argv[3], sys.argv[2])

read_triples_files(path, bos, dict_src_tgt, '../../evaluation/our_beam/{}'.format(os.path.basename(sys.argv[2])), 'check_reference')
