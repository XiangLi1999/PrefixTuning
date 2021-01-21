

from __future__ import absolute_import, division, print_function

_DOCUMENT = "document"
_SUMMARY = "summary"
_ID = "id"

_REMOVE_LINES = set(
    [
        "Share this with\n",
        "Email\n",
        "Facebook\n",
        "Messenger\n",
        "Twitter\n",
        "Pinterest\n",
        "WhatsApp\n",
        "Linkedin\n",
        "LinkedIn\n",
        "Copy this link\n",
        "These are external links and will open in a new window\n",
    ]
)

def _generate_examples(split_path, split_name, data_dir):
        """Yields examples."""

        with open(split_path, "r", encoding="utf-8") as f:
            split_ids = json.load(f)

        for i in split_ids[split_name]:
            with open(os.path.join(data_dir, i + ".summary"), "r", encoding="utf-8") as f:
                text = "".join([line for line in f.readlines() if line not in _REMOVE_LINES and line.strip()])
                # Each file follows below format:
                # [SN]URL[SN]
                # http://somelink
                #
                # [SN]TITLE[SN]
                # some intro
                #
                # [SN]FIRST-SENTENCE[SN]
                # some intro
                #
                # [SN]RESTBODY[SN]
                # text line.
                # another text line.
                # "another text line."

                # According to the following issue, FIRST-SENTENCE
                # is the reference summary and TITLE is unused:
                # https://github.com/EdinburghNLP/XSum/issues/22
                segs = text.split("[SN]")
                yield i, {_DOCUMENT: segs[8].strip(), _SUMMARY: segs[6].strip(), _ID: i, 'url':segs[2]}



import json
import os

import datasets
import sys 
data_dir = 'xsum_topic'
train_path_src = os.path.join(data_dir, 'train.source')
train_path_tgt = os.path.join(data_dir, 'train.target')
train_path_topic = os.path.join(data_dir, 'train.topic')
dev_path_src = os.path.join(data_dir, 'val.source')
dev_path_tgt = os.path.join(data_dir, 'val.target')
dev_path_topic = os.path.join(data_dir, 'val.topic')
test_path_src = os.path.join(data_dir, 'test.source')
test_path_tgt = os.path.join(data_dir, 'test.target')
test_path_topic = os.path.join(data_dir, 'test.topic')
# 100, 200, 500, 1k

count = 0

if sys.argv[2] == 'validation':
    src_path = open(dev_path_src, 'w')
    tgt_path = open(dev_path_tgt, 'w')
    topic_path = open(dev_path_topic, 'w')

elif sys.argv[2] == 'train':
    src_path = open(train_path_src, 'w')
    tgt_path = open(train_path_tgt, 'w')
    topic_path = open(train_path_topic, 'w')

elif sys.argv[2] == 'test':
    src_path = open(test_path_src, 'w')
    tgt_path = open(test_path_tgt, 'w')
    topic_path = open(test_path_topic, 'w')
for i in _generate_examples(sys.argv[1], sys.argv[2], sys.argv[3]):
    count += 1
    document = i[1]['document']
    document = document.replace('\n', '').strip()
    # print(document)
    summary = i[1]['summary'].strip()
    url = i[1]['url'].strip()

    if document:
        src_path.write(document + '\n')
        # print(summary)
        # print(dev_path_tgt)
        tgt_path.write(summary + '\n')
        topic_path.write(url + '\n')

src_path.close()
tgt_path.close()
topic_path.close()
print(count)

