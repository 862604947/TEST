#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *
from corpus import get_all_corpus
from vocab import get_vocab


def is_quatrain(poem):
    if poem['source'] == 'qsc_tab.txt':
        return False
    else:
        sentences = poem['sentences']
        return len(sentences) == 4 and \
                (len(sentences[0]) == 5 or len(sentences[0]) == 7) and \
                reduce(lambda x, sentence: x and len(sentence) == len(sentences[0]),
                        sentences[1:], True)


def get_quatrains():  # 返回每个字符都在字库ch2int中的四行诗的诗句
    _, ch2int = get_vocab()
    def quatrain_filter(poem):
        if not is_quatrain(poem):
            return False
        else:
            for sentence in poem['sentences']:
                for ch in sentence:
                    if ch not in ch2int:
                        return False
            return True
    return filter(quatrain_filter, get_all_corpus())  # get_all_corpus()方法返回的是所有诗句文件数据中的诗的记录，每一行代表一首诗的名、作者、朝代、诗句


if __name__ == '__main__':
    quatrains = get_quatrains()
    print "Size of quatrains: %d" % len(quatrains)

