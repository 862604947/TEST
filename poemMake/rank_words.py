#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *
from segment import Segmenter, get_sxhy_dict
from quatrains import get_quatrains


stopwords_raw = os.path.join(raw_dir, 'stopwords.txt')

rank_path = os.path.join(data_dir, 'word_ranks.json')


def get_stopwords():
    stopwords = set()
    with codecs.open(stopwords_raw, 'r', 'utf-8') as fin:
        line = fin.readline()
        while line:
            stopwords.add(line.strip())
            line = fin.readline()
    return stopwords


def _text_rank(adjlist):  # 模仿PageRank算法实现text_rank
    damp = 0.85   # d值默认为0.85
    scores = dict((word,1.0) for word in adjlist)  # rank值
    try:
        for i in range(100000):  # 最大迭代次数位100000
            print "[TextRank] Start iteration %d ..." %i,
            sys.stdout.flush()
            cnt = 0   # 记录该次迭代是否有rank被更新了。
            new_scores = dict()  #记录该论迭代的rank值
            for word in adjlist:  # 实际计算根据公式
                new_scores[word] = (1-damp)+damp*sum(adjlist[other][word]*scores[other] \
                        for other in adjlist[word])
                if scores[word] != new_scores[word]:
                    cnt += 1
            print "Done (%d/%d)" %(cnt, len(scores))
            if 0 == cnt:
                break  # cnt如果为0说明没有rank改变则需要停止
            else:
                scores = new_scores  # 否则继续迭代更新权重值
        print "TextRank is done."
    except KeyboardInterrupt:
        print "\nTextRank is interrupted."
    sxhy_dict = get_sxhy_dict()
    def _compare_words(a, b):
        if a[0] in sxhy_dict and b[0] not in sxhy_dict:
            return -1
        elif a[0] not in sxhy_dict and b[0] in sxhy_dict:
            return 1
        else:
            return cmp(b[1], a[1])
    words = sorted([(word,score) for word,score in scores.items()],
            cmp = _compare_words)
    with codecs.open(rank_path, 'w', 'utf-8') as fout:
        json.dump(words, fout)


def _rank_all_words():
    segmenter = Segmenter()  # 诗句分段器
    stopwords = get_stopwords()  # 停用词列表
    print "Start TextRank over the selected quatrains ..."
    quatrains = get_quatrains()  # 四行诗集合
    adjlist = dict()
    for idx, poem in enumerate(quatrains):  # 对于每首诗
        if 0 == (idx+1)%10000:
            print "[TextRank] Scanning %d/%d poems ..." %(idx+1, len(quatrains))
        for sentence in poem['sentences']:  # 对于每一句诗
            segs = filter(lambda word: word not in stopwords,
                    segmenter.segment(sentence))  # 得到不再停用词中的词段
            for seg in segs:  # 对于每个词段
                if seg not in adjlist:
                    adjlist[seg] = dict()  # 每个词段生成一个字典dict
            for i, seg in enumerate(segs):  # 对于每个词段
                for _, other in enumerate(segs[i+1:]): # 去和后面的每个词段比较，实际是源于text_rank需要的网状结构图
                    if seg != other:  # 精巧的code
                        adjlist[seg][other] = adjlist[seg][other]+1 \
                                if other in adjlist[seg] else 1.0
                        adjlist[other][seg] = adjlist[other][seg]+1 \
                                if seg in adjlist[other] else 1.0
    for word in adjlist:
        w_sum = sum(weight for other, weight in adjlist[word].items())  # 求该word对应的所有词的权重综合
        for other in adjlist[word]:
            adjlist[word][other] /= w_sum  # 求该word中每个value对应的权重平均值
    print "[TextRank] Weighted graph has been built."
    _text_rank(adjlist)

# 文件数据处理的典型套路代码结构模式。代码喜欢用filter()/reduce()/lambda/os.path/codecs.open()/json.load()/json.dump()/dict
def get_word_ranks():
    if not os.path.exists(rank_path):
        _rank_all_words()
    with codecs.open(rank_path, 'r', 'utf-8') as fin:
        ranks = json.load(fin)
    return dict((pair[0], idx) for idx, pair in enumerate(ranks))


if __name__ == '__main__':
    ranks = get_word_ranks()
    print "Size of word_ranks: %d" % len(ranks)

