#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *
from segment import Segmenter
from vocab import get_vocab, VOCAB_SIZE
from quatrains import get_quatrains
from gensim import models
from numpy.random import uniform

_w2v_path = os.path.join(data_dir, 'word2vec.npy')

def _gen_embedding(ndim):  # 生成ndim维度的词向量
    print "Generating %d-dim word embedding ..." %ndim
    int2ch, ch2int = get_vocab()  # 得到词库
    ch_lists = []
    quatrains = get_quatrains()  # 得到所有符合要求规则的四行诗的诗句
    for idx, poem in enumerate(quatrains):  # 对于四行诗中的每一首诗
        for sentence in poem['sentences']:  # 对于诗中的每一句诗
            ch_lists.append(filter(lambda ch: ch in ch2int, sentence))  # 检查诗句的每一行中哪些在ch2int词典中
        if 0 == (idx+1)%10000:
            print "[Word2Vec] %d/%d poems have been processed." %(idx+1, len(quatrains))
    print "Hold on. This may take some time ..."
    model = models.Word2Vec(ch_lists, size = ndim, min_count = 5)  # ch_list是词库，ndim是要生成的词向量的维度
    embedding = uniform(-1.0, 1.0, [VOCAB_SIZE, ndim])  # 平均分布的矩阵，每一行代表一个词向量，每一个词向量维度ndim
    for idx, ch in enumerate(int2ch):
        if ch in model.wv:  # 如果int2ch中的该词在model生成的词向量中
            embedding[idx,:] = model.wv[ch]  # embedding中的该行代表ch对应的词向量
    np.save(_w2v_path, embedding)
    print "Word embedding is saved."

# 固定代码套路，一个生成gen方法，一个得到get方法
def get_word_embedding(ndim):
    if not os.path.exists(_w2v_path):
        _gen_embedding(ndim)
    return np.load(_w2v_path)


if __name__ == '__main__':
    embedding = get_word_embedding(128)
    print "Size of embedding: (%d, %d)" %embedding.shape


