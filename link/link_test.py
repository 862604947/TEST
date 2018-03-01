__author__ = 'samsung'
"""
链接图片生成部分和生词生成部分的测试应用部分，训练时两部分单独训练
"""
import pictureDis.label_images as li
from poemMake.data_utils import *
from poemMake.plan import Planner
from poemMake.generate import Generator
import argparse
import tensorflow as tf
import sys

sys.setdefaultencoding('utf8')

if __name__=='__main__':
    planner = Planner()
    generator = Generator()
    while True:
        FLAGS,unparsed = li.parser.parse_known_args()
        pic_feature_results = li.main(sys.argv[:1]+unparsed)

        line = " ".join(pic_feature_results)
        if None == line.lower():
            break
        elif len(pic_feature_results)>0 :
            keywords = planner.plan(line)
            print("Keywords:\t",)
            for word in keywords:
                print(word)
            print('\n')
            print("Poem Generated:\n")
            sentences = generator.generate(keywords)
            print('\t'+sentences[0]+u'，'+sentences[1]+u'。')
            print('\t'+sentences[2]+u'，'+sentences[3]+u'。')