#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *
from rank_words import get_stopwords
from data_utils import kw_train_path
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer  # 两个类都是为了计算TF-IDF值的，
#TF-IDF为每一个文档d和由关键词w[1]...w[k]组成的查询串q计算一个权值，用于表示查询串q与文档d的匹配度
from sklearn.metrics import silhouette_score


def get_cluster_labels(texts, tokenizer, n_clusters):  # 典型的对文档进行聚类的流程
    print "Clustering %d texts into %d groups ..." %(len(texts), n_clusters)
    vectorizer = CountVectorizer(tokenizer = tokenizer,
            stop_words = get_stopwords())
    transformer = TfidfTransformer()
    km = KMeans(n_clusters = n_clusters)  # 构造聚类器
    # vectorizer.fit_transform(texts)将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在第i个文本下的词频
    tfidf = transformer.fit_transform(vectorizer.fit_transform(texts)) #TfidfTransformer是统计vectorizer中每个词语的tf-idf权值返回权值矩阵
    km.fit(tfidf)  #  对文档texts中每个单词对每篇文档的tfidf权值矩阵实行聚类
    return km.labels_.tolist()  # 获取聚类标签


def _eval_cluster(texts, tokenizer, n_clusters):
    vectorizer = CountVectorizer(tokenizer = tokenizer,
            stop_words = get_stopwords())
    transformer = TfidfTransformer()
    km = KMeans(n_clusters = n_clusters)
    tfidf = transformer.fit_transform(vectorizer.fit_transform(texts))
    km.fit(tfidf)
    return silhouette_score(tfidf,
            km.labels_.tolist(),
            sample_size = 1000)  # 评估聚类效果，评分结果在[-1, +1]之间，评分结果越高，聚类结果越好


if __name__ == '__main__':
    texts = []
    with codecs.open(kw_train_path, 'r', 'utf-8') as fin:
        line = fin.readline()
        while line:
            texts.append(line.strip())
            line = fin.readline()
    for n in range(2, 30):
        score = _eval_cluster(texts,
                tokenizer = lambda x: x.split('\t'),
                n_clusters = n)
        print "n_clusters = %d, score = %f" %(n, score)

