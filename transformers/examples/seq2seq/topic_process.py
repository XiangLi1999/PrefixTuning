import nltk, os, sys
from nltk.tokenize import RegexpTokenizer


data_dir = 'xsum'
train_path_src = os.path.join(data_dir, 'train.source')
train_path_tgt = os.path.join(data_dir, 'train.target')
dev_path_src = os.path.join(data_dir, 'val.source')
dev_path_tgt = os.path.join(data_dir, 'val.target')
test_path_src = os.path.join(data_dir, 'test.source')
test_path_tgt = os.path.join(data_dir, 'test.target')
# 100, 200, 500, 1k

src_lst = []
with open(dev_path_src,'r') as f:
    for line in f:
        src_lst.append(line)

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
# nltk.download('wordnet')
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def lemmatize_stemming(text):
    return stemmer.stem(lemmatizer.lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

processed_docs = []
for x in src_lst:
    processed_docs.append(preprocess(x))

print(processed_docs[:2])

dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=200, id2word=dictionary, passes=2, workers=2)


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
for i in _generate_examples(sys.argv[1], sys.argv[2], sys.argv[3]):
    count += 1
    src_path.write(i[1]['document'])
    tgt_path.write(i[1]['summary'])
    topic_path.write(i[1]['url'])

src_path.close()
tgt_path.close()
topic_path.close()
print(count)
