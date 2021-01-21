import os, sys
import random
import numpy as np


data_dir = 'xsum'
train_path_src = os.path.join(data_dir, 'train.source')
train_path_tgt = os.path.join(data_dir, 'train.target')
dev_path_src = os.path.join(data_dir, 'val.source')
dev_path_tgt = os.path.join(data_dir, 'val.target')
test_path_src = os.path.join(data_dir, 'test.source')
test_path_tgt = os.path.join(data_dir, 'test.target')
# 100, 200, 500, 1k

def select(path_src, path_tgt, num, outp_src, outp_tgt):
    src_full = []
    with open (path_src) as src:
        for line in src:
            src_full.append(line.strip())

    tgt_full = []
    with open(path_tgt) as tgt:
        for line in tgt:
            tgt_full.append(line.strip())

    assert len(src_full) == len(tgt_full)
    print(len(src_full), len(tgt_full))
    # a = np.random.choice(len(src_full), 10)
    # print(a)
    temp_lst = np.random.choice(len(src_full), num, replace=False)

    src_out = open(outp_src, 'w')
    tgt_out = open(outp_tgt, 'w')

    for idx in temp_lst:
        src_out.write(src_full[idx] + '\n')
        tgt_out.write(tgt_full[idx] + '\n')

    src_out.close()
    tgt_out.close()

    return

if __name__ == '__main__':
    num_ = int(sys.argv[1])
    seed_ = int(sys.argv[2])
    np.random.seed(seed_)
    data_dir2 = 'lowdata_xsum/xsum_{}_{}'.format(num_, seed_)
    os.mkdir(data_dir2)

    # data_dir2 = 'lowdata_xsum/xsum_small_test'

    train_path_src2 = os.path.join(data_dir2, 'train.source')
    train_path_tgt2 = os.path.join(data_dir2, 'train.target')
    dev_path_src2 = os.path.join(data_dir2, 'val.source')
    dev_path_tgt2 = os.path.join(data_dir2, 'val.target')
    test_path_src2 = os.path.join(data_dir2, 'test.source')
    test_path_tgt2 = os.path.join(data_dir2, 'test.target')
    select(train_path_src, train_path_tgt, num_, train_path_src2, train_path_tgt2)
    select(dev_path_src, dev_path_tgt, int(num_*0.3), dev_path_src2, dev_path_tgt2)

    # os.mkdir(data_dir2)
    # select(test_path_src, test_path_tgt, 1500, test_path_src2, test_path_tgt2)





