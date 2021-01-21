import sys, os
from collections import Counter
import numpy as np
def process_url(url):
    url = url.strip('/')
    keywords = url.split('/')[-3:]
    # print('old', keywords)
    if keywords[-1].isdigit():
        if 'www' in keywords[0]:
            del keywords[0]
        keywords = keywords[:-1]
    else:
        detailed_topic = keywords[-1].split('-')[:-1]
        if 'www' in keywords[0]:
            del keywords[0]
        # print(detailed_topic, keywords)
        if len(detailed_topic) > 0:
            keywords = keywords[:-1] + [detailed_topic[0]]
        else:
            keywords = keywords[:-1]
    # print(tuple(keywords))
    return tuple([x for x in keywords if x ])

def count_topic(topic_lst):
    count = Counter()
    for topic in topic_lst:
        count[topic] += 1
    print(count)

    # gather topics that has count > 10
    topic_selected_lst = []
    for key, val in count.items():
        if val > 10:
            topic_selected_lst.append(key)
            print(key, val)


def get_file_handle():
    pass


if __name__ == '__main__':
    topic_lst = []
    with open('xsum_topic/{}.topic'.format(sys.argv[1]), 'r') as f:
        for line in f:
            line = line.strip()
            topic = process_url(line)
            topic_lst.append(topic)

    # count_topic(topic_lst)
    # two options.
    # 1. Train on (news, *) and eval on (sports)
    # 2. Train on (news, uk), (news, business), (news, world) and eval on (news, other*)

    option = 2




    if option == 1:

        data_dir = 'xsum_topic-news-sports2'
        train_path_src = os.path.join(data_dir, 'train.source')
        train_path_tgt = os.path.join(data_dir, 'train.target')
        train_path_topic = os.path.join(data_dir, 'train.topic')
        dev_path_src = os.path.join(data_dir, 'val.source')
        dev_path_tgt = os.path.join(data_dir, 'val.target')
        dev_path_topic = os.path.join(data_dir, 'val.topic')
        test_path_src = os.path.join(data_dir, 'test.source')
        test_path_tgt = os.path.join(data_dir, 'test.target')
        test_path_topic = os.path.join(data_dir, 'test.topic')

        source_path = 'xsum_topic/{}.source'.format(sys.argv[1])
        source_lst = []
        with open(source_path, 'r') as f:
            for line in f :
                source_lst.append(line)

        target_path = 'xsum_topic/{}.target'.format(sys.argv[1])
        target_lst = []
        with open(target_path, 'r') as f:
            for line in f:
                target_lst.append(line)

        assert len(target_lst) == len(source_lst)
        assert len(target_lst) == len(topic_lst)

        max_num = None

        if sys.argv[3] == 'train':
            out_source = open(train_path_src, 'w') # train
            out_target = open(train_path_tgt, 'w')
            out_topic = open(train_path_topic, 'w')
            print('writing to train')

        elif sys.argv[3] == 'test':
            out_source = open(test_path_src, 'w') # test
            out_target = open(test_path_tgt, 'w')
            out_topic = open(test_path_topic, 'w')
            print('writing to test')
            max_num = 8000

        elif sys.argv[3] == 'val':
            out_source = open(dev_path_src, 'w')  # dev
            out_target = open(dev_path_tgt, 'w')
            out_topic = open(dev_path_topic, 'w')
            print('writing to val')




        final_lst_topic = []
        final_lst_source = []
        final_lst_target =  []
        for topic, src, tgt in zip(topic_lst, source_lst, target_lst):
            if topic[0] == sys.argv[2]:

                if max_num is None:
                    out_topic.write(str(topic) + '\n')
                    out_source.write(src)
                    out_target.write(tgt)

                else:
                    final_lst_topic.append(str(topic))
                    final_lst_source.append(src)
                    final_lst_target.append(tgt)

        if max_num is not None:
            assert len(final_lst_topic) == len(final_lst_target)
            assert len(final_lst_topic) == len(final_lst_source)
            print('the max number is {}'.format(max_num))
            cand_lst = np.random.choice(len(final_lst_topic), max_num, replace=False)

            for cand in cand_lst:
                out_topic.write(final_lst_topic[cand] + '\n')
                out_source.write(final_lst_source[cand])
                out_target.write(final_lst_target[cand])






            # elif topic[0] == 'sport':
            #     out_topic_val.write(str(topic) + '\n')
            #     out_source_val.write(src)
            #     out_target_val.write(tgt)


        out_source.close()
        out_topic.close()
        out_target.close()
        # out_topic_val.close()
        # out_source_val.close()
        # out_target_val.close()

    elif option == 2:

        data_dir = 'xsum_news2'
        train_path_src = os.path.join(data_dir, 'train.source')
        train_path_tgt = os.path.join(data_dir, 'train.target')
        train_path_topic = os.path.join(data_dir, 'train.topic')
        dev_path_src = os.path.join(data_dir, 'val.source')
        dev_path_tgt = os.path.join(data_dir, 'val.target')
        dev_path_topic = os.path.join(data_dir, 'val.topic')
        test_path_src = os.path.join(data_dir, 'test.source')
        test_path_tgt = os.path.join(data_dir, 'test.target')
        test_path_topic = os.path.join(data_dir, 'test.topic')

        source_path = 'xsum_topic/{}.source'.format(sys.argv[1])
        source_lst = []
        with open(source_path, 'r') as f:
            for line in f :
                source_lst.append(line)

        target_path = 'xsum_topic/{}.target'.format(sys.argv[1])
        target_lst = []
        with open(target_path, 'r') as f:
            for line in f:
                target_lst.append(line)

        assert len(target_lst) == len(source_lst)
        assert len(target_lst) == len(topic_lst)

        max_num = None

        if sys.argv[3] == 'train':
            out_source = open(train_path_src, 'w') # train
            out_target = open(train_path_tgt, 'w')
            out_topic = open(train_path_topic, 'w')
            print('writing to train')

        elif sys.argv[3] == 'test':
            out_source = open(test_path_src, 'w') # test
            out_target = open(test_path_tgt, 'w')
            out_topic = open(test_path_topic, 'w')
            print('writing to test')
            max_num = 8000

        elif sys.argv[3] == 'val':
            out_source = open(dev_path_src, 'w')  # dev
            out_target = open(dev_path_tgt, 'w')
            out_topic = open(dev_path_topic, 'w')
            print('writing to val')


        final_lst_topic = []
        final_lst_source = []
        final_lst_target =  []
        for topic, src, tgt in zip(topic_lst, source_lst, target_lst):
            if topic[0] == 'news':
                if topic in [('news', 'uk'), ('news', 'business'), ('news', 'world'),]:
                    if sys.argv[2] == 'yes':

                        if max_num is None:
                            out_topic.write(str(topic) + '\n')
                            out_source.write(src)
                            out_target.write(tgt)

                        else:
                            final_lst_topic.append(str(topic))
                            final_lst_source.append(src)
                            final_lst_target.append(tgt)

                else:
                    if sys.argv[2] == 'no':
                        if max_num is None:
                            out_topic.write(str(topic) + '\n')
                            out_source.write(src)
                            out_target.write(tgt)

                        else:
                            final_lst_topic.append(str(topic))
                            final_lst_source.append(src)
                            final_lst_target.append(tgt)

            # elif topic[0] == 'sport':
            #     out_topic_val.write(str(topic) + '\n')
            #     out_source_val.write(src)
            #     out_target_val.write(tgt)

        if max_num is not None:
            assert len(final_lst_topic) == len(final_lst_target)
            assert len(final_lst_topic) == len(final_lst_source)
            print('the max number is {}'.format(max_num))
            cand_lst = np.random.choice(len(final_lst_topic), max_num, replace=False)

            for cand in cand_lst:
                out_topic.write(final_lst_topic[cand] + '\n')
                out_source.write(final_lst_source[cand])
                out_target.write(final_lst_target[cand])

        out_source.close()
        out_topic.close()
        out_target.close()
        # out_topic_val.close()
        # out_source_val.close()
        # out_target_val.close()









