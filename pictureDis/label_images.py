__author__ = 'samsung'
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simple image classification with Inception.

Run image classification with your model.

This script is usually used with retrain.py found in this same
directory.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. You are required
to pass in the graph file and the txt file.

It outputs human readable strings of the top 5 predictions along with
their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Example usage:
python label_image.py --graph=retrained_graph.pb
  --labels=retrained_labels.txt
  --image=flower_photos/daisy/54377391_15648e8d18.jpg

NOTE: To learn to use this file and retrain.py, please see:

https://codelabs.developers.google.com/codelabs/tensorflow-for-poets
"""

import argparse
import sys

import tensorflow as tf

parser = argparse.ArgumentParser()  # argparse详解https://www.cnblogs.com/arkenstone/p/6250782.html
parser.add_argument(
    '--image', required=True, type=str, help='Absolute path to image file.')
parser.add_argument(
    '--num_top_predictions',
    type=int,
    default=4,
    help='Display this many predictions.')
parser.add_argument(
    '--graph',
    required=True,
    type=str,
    help='Absolute path to graph file (.pb)')
parser.add_argument(
    '--labels',
    required=True,
    type=str,
    help='Absolute path to labels file (.txt)')  # 标签文本文件
parser.add_argument(
    '--output_layer',
    type=str,
    default='final_result:0',
    help='Name of the result operation')
parser.add_argument(
    '--input_layer',
    type=str,
    default='DecodeJpeg/contents:0',
    help='Name of the input operation')

def trans_eng2chi():
  trans = {}
  trans["desert"] = ["沙漠"]
  trans["desert_mountains"] = ["沙漠","高山"]
  trans["desert_mountains_sunset"] = ["沙漠","高山","夕阳"]
  trans["desert_sea"] = ["沙漠","大海"]
  trans["desert_sunset"] = ["沙漠","夕阳"]
  trans["desert_sunset_trees"] = ["沙漠","夕阳","树"]
  trans["desert_trees"] = ["沙漠","树"]
  trans["mountains"] = ["高山"]
  trans["mountains sea"] = ["高山","大海"]
  trans["mountains sea trees"] = ["高山","大海","树"]
  trans["mountains sunset"] = ["高山","夕阳"]
  trans["mountains sunset trees"] = ["高山","夕阳","树"]
  trans["mountains trees"] = ["高山","树"]
  trans["sea"] = ["大海"]
  trans["sea sunset"] = ["大海","夕阳"]
  trans["sea sunset trees"] = ["大海","夕阳","树"]
  trans["sea trees"] = ["大海","树"]
  trans["sunset"] = ["夕阳"]
  trans["sunset trees"] = ["夕阳","树"]
  trans["trees"] = ["树"]
  return trans


def load_image(filename):  # 加载测试用的图片
  """Read in the image_data to be classified."""
  return tf.gfile.FastGFile(filename, 'rb').read()


def load_labels(filename):  # 加载之前生成的output_labels.txt中的标签，加载这个是为了从概率值转化为类别标签
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):  # 加载之前生成的图模型
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def run_graph(image_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  with tf.Session() as sess:
    # Feed the image_data as input to the graph.
    #   predictions will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})

    # Sort to show labels in order of confidence
    trans_dict = trans_eng2chi()
    result = []
    top_k = predictions.argsort()[-num_top_predictions:][::-1]  # 取得前top_k个
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))
      result.extend(trans_dict[human_string])
    return list(set(result))

def main(argv):
  """Runs inference on an image."""
  if argv[1:]:  # 如果argv中有多余一个参数的情况则说明有异常
    raise ValueError('Unused Command Line Args: %s' % argv[1:])

  if not tf.gfile.Exists(FLAGS.image):
    tf.logging.fatal('image file does not exist %s', FLAGS.image)

  if not tf.gfile.Exists(FLAGS.labels):
    tf.logging.fatal('labels file does not exist %s', FLAGS.labels)

  if not tf.gfile.Exists(FLAGS.graph):
    tf.logging.fatal('graph file does not exist %s', FLAGS.graph)

  # load image
  image_data = load_image(FLAGS.image)

  # load labels
  labels = load_labels(FLAGS.labels)

  # load graph, which is stored in the default session
  load_graph(FLAGS.graph)

  result = run_graph(image_data, labels, FLAGS.input_layer, FLAGS.output_layer,
            FLAGS.num_top_predictions)
  return result  #返回的是中文类别列表，包含多个不重复的中文特征词，可以直接作为诗词部分的测试输入。

if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()  #假设unparsed为[]空
  tf.app.run(main=main, argv=sys.argv[:1]+unparsed)  # 只是把执行的python文件名传给了main（）函数

# 奇怪的地方：main函数并没有FLAGS作为其传参，但是可以直接使用FLAGS变量