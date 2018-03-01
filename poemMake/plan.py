#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *
from segment import Segmenter
from quatrains import get_quatrains
from rank_words import get_word_ranks
from data_utils import *
import jieba
from gensim import models
from random import shuffle, random, randint

_model_path = os.path.join(data_dir, 'kw_model.bin')


class Planner:

    def __init__(self):
        self.ranks = get_word_ranks()
        if not os.path.exists(_model_path):
            self._train()
        else:
            self.model = models.Word2Vec.load(_model_path)

    def _train(self):
        print "Start training Word2Vec for planner ..."
        quatrains = get_quatrains()
        segmenter = Segmenter()  # 对诗句分段和取其中的每个词不一样
        seg_lists = []
        for idx, quatrain in enumerate(quatrains):
            seg_list = []
            for sentence in quatrain['sentences']:
                seg_list.extend(filter(lambda seg: seg in self.ranks,
                        segmenter.segment(sentence)))
            seg_lists.append(seg_list)
            if 0 == (idx+1)%10000:
                print "[Plan Word2Vec] %d/%d quatrains has been processed." %(idx+1, len(quatrains))
        print "Hold on. This may take some time ..."
        self.model = models.Word2Vec(seg_lists, size = 512, min_count = 5)  # 代表一个词向量类，生成的是词向量模型
        self.model.save(_model_path)

    def expand(self, words, num):
        positive = filter(lambda w: w in self.model.wv, words)
        similars = self.model.wv.most_similar(positive = positive) \
                if len(positive) > 0 else []
        words.extend(pair[0] for pair in similars[:min(len(similars), num-len(words))])
        if len(words) < num:
            _prob_sum = sum(1./(i+1) for i in range(len(self.ranks)))
            _prob_sum -= sum(1./(self.ranks[word]+1) for word in words)
            while len(words) < num:
                prob_sum = _prob_sum
                for word, rank in self.ranks.items():
                    if word in words:
                        continue
                    elif prob_sum * random() < 1./(rank+1):
                        words.append(word)
                        break
                    else:
                        prob_sum -= 1./(rank+1)
        shuffle(words)

    def plan(self, text):
        def extract(sentence):
            return filter(lambda x: x in self.ranks, jieba.lcut(sentence))  # 对一句诗进行分词，而且词必须在词库中
        #  把每一句诗分词之后累加起来之后按照每个词的rank值排序得到结果列表，列表是按照rank排好序的从小到大的
        keywords = sorted(reduce(lambda x,y:x+y, map(extract, split_sentences(text)), []),
            cmp = lambda x,y: cmp(self.ranks[x], self.ranks[y]))  # cmp在python3中不再存在，而是由operator模块中的比较大小的方法取代
        words = [keywords[idx] for idx in \
                filter(lambda i: 0 == i or keywords[i] != keywords[i-1], range(len(keywords)))] # 去除前后相同的词
        if len(words) < 4:
            self.expand(words, 4)
        else:
            while len(words) > 4:
                words.pop()
        return words  # 一定是四个词，四个和给定text每行最相关的四个词

if __name__ == '__main__':
    planner = Planner()
    kw_train_data = get_kw_train_data()
    for row in kw_train_data:
        num = randint(1,3)
        uprint(row[1:])
        print "num = %d" %num
        guess = row[1:num+1]
        planner.expand(guess, 4)
        uprintln(guess)
        assert len(guess) == 4
        print

